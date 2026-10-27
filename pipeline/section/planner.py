"""[plan] OCR結果から章・節の区切りを判定し、VLMに入れる範囲(チャンク)をJSONとして保存する

保存先: chunks/{報告書名}/
  chunk_0001.json … VLMへの入力1回分(対象ページの画像の場所、OCR結果(座標・確信度を含む全項目)、範囲、図)
  structure.json  … 判定した見出しの一覧と根拠、目次、ページの種類
  plan.md         … 人が確認するための一覧表
  plan.json       … チャンクの一覧(最後に書き出し、これがあれば判定が完了している)
"""

import json
import shutil
from datetime import datetime

from ..base import Step
from .analyzer import DocumentStructure, StructureAnalyzer
from .chunker import Chunk, Chunker
from .figures import FigureRegistry
from .llm_judge import LlmBoundaryJudge
from .ocr_page import OcrPage
from .report import Report
from .validator import PLAN_FORMAT, PlanValidator

CHUNK_FORMAT = "capture_vlm.chunk/1"


class SectionPlanner(Step):
    name = "区切りの判定"

    def run(self) -> None:
        validator = PlanValidator(self.config)
        for report in self.targets.reports:
            plan_path = validator.plan_dir(report) / "plan.json"
            if plan_path.exists() and not self.config.overwrite and not validator.problems(report):
                self.log(f"{report.name}: 判定済みのためスキップ")
                continue
            missing = self._missing_inputs(report)
            if missing:
                self.log(f"[スキップ] {report.name}: 画像またはOCR結果が揃っていません({len(missing)}件)")
                for m in missing[:5]:
                    self.log(f"  - {m}")
                continue
            self._plan(report)

    def _missing_inputs(self, report: Report) -> list[str]:
        from ..ocr_runner import NdlOcrRunner
        missing = []
        for ref in report.pages:
            if not ref.image_path.exists():
                missing.append(ref.image_path.name)
            if not NdlOcrRunner._is_current(ref.json_path):
                missing.append(f"{ref.json_path.name}(無いか古い形式。--only ocr --overwrite で作り直してください)")
        return missing

    def _plan(self, report: Report) -> None:
        cfg = self.config
        self.log(f"{report.name}: {len(report.pdfs)}個のPDF / {len(report.pages)}ページ")
        pages = [OcrPage.load(ref) for ref in report.pages]

        judge = None
        if cfg.boundary_llm != "off":
            from ..vlm_converter import get_client
            judge = LlmBoundaryJudge(lambda: get_client(cfg))
        structure = StructureAnalyzer(cfg, judge, log=lambda m: self.log("  " + m)).analyze(pages)
        figures = FigureRegistry(pages)
        chunker = Chunker(cfg, pages, structure, figures)
        chunks = chunker.build()

        out = PlanValidator(cfg).plan_dir(report)
        if out.exists():
            shutil.rmtree(out)  # 古い判定結果が混ざらないよう作り直す
        out.mkdir(parents=True)
        fig_dir = cfg.output_dir / report.name / "figures"
        n_saved = figures.save_crops(fig_dir)

        items = []
        for c in chunks:
            path = out / f"chunk_{c.id}.json"
            path.write_text(json.dumps(self._chunk_json(report, c, chunker, {p.ref.global_no: p for p in pages}),
                                       ensure_ascii=False, indent=2), encoding="utf-8")
            items.append({"chunk_id": c.id, "file": path.name, "kind": c.kind, "title": c.title,
                          "pages": [c.start.page, c.end_page]})
        (out / "structure.json").write_text(json.dumps(self._structure_json(structure), ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        (out / "plan.md").write_text(self._plan_md(report, structure, chunks), encoding="utf-8")
        unassigned = {f.id for f in figures.figures} - {f.id for c in chunks for f in c.figures}
        plan = {
            "format": PLAN_FORMAT, "status": "complete", "report": report.name,
            "pdfs": [p.name for p in report.pdfs], "page_count": len(report.pages),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "settings": {"section_level": cfg.section_level, "max_pages_per_chunk": cfg.max_pages_per_chunk,
                         "boundary_llm": cfg.boundary_llm, "model_id": cfg.model_id if judge else None},
            "chunks": items,
        }
        (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

        self.log(f"  → チャンク {len(chunks)}個 / 見出し {len(structure.headings)}個 / 図 {len(figures.figures)}個(切り出し {n_saved}個)")
        if unassigned:
            self.log(f"  [警告] どのチャンクにも入らなかった図: {sorted(unassigned)}")
        if structure.unlocated_toc:
            self.log(f"  [注意] 本文で位置を特定できなかった目次の項目: {len(structure.unlocated_toc)}件(plan.md を確認してください)")
        self.log(f"  確認用の一覧: {out / 'plan.md'}")

    @staticmethod
    def _chunk_json(report: Report, c: Chunk, chunker: Chunker, pages: dict[int, OcrPage]) -> dict:
        scope = chunker.scope_lines(c)
        def line_text(no, order):
            if order is None:
                return None
            l = next((x for x in pages[no].lines if x.order == order), None)
            return {"line_id": l.id, "text": l.text} if l else None
        return {
            "format": CHUNK_FORMAT,
            "report": report.name,
            "chunk_id": c.id,
            "kind": c.kind,
            "title": c.title,
            "level": c.level,
            "breadcrumb": c.breadcrumb,
            "continued": c.continued,
            "scope": {
                # shared_with_previous / next: そのページを前後のチャンクと共有している(ページ途中で区切った)
                "start": {"page": c.start.page, "line": line_text(c.start.page, c.start.order),
                          "shared_with_previous": c.start.order is not None
                          and chunker._content_before(c.start.page, c.start.order)},
                "end": {"page": c.end_page, "before_line": line_text(c.end_page, c.end_before),
                        "shared_with_next": c.end_before is not None},
            },
            "headings": [h.to_dict() for h in c.headings],
            "figures": [f.to_dict() for f in c.figures],
            "pages": [
                {
                    "global_page": no,
                    "pdf": pages[no].ref.pdf_stem,
                    "page_no": pages[no].ref.page_no,
                    "printed_page": pages[no].folio,
                    "kind": chunker.structure.page_kinds.get(no),
                    "image": pages[no].ref.image_path.name,
                    "ocr_json": pages[no].ref.json_path.name,
                    "in_scope_line_ids": scope.get(no, []),
                    "ocr": pages[no].data,  # 座標・確信度などを含むOCR結果をそのまま保存(後の学習用)
                }
                for no in c.page_range
            ],
        }

    @staticmethod
    def _structure_json(s: DocumentStructure) -> dict:
        return {
            "headings": [h.to_dict() for h in s.headings],
            "toc_pages": s.toc_pages,
            "toc_entries": [{"title": e.title, "printed_page": e.printed_page, "level": e.level,
                             "located_page": e.located_page, "located_line": e.located_line} for e in s.toc_entries],
            "page_kinds": s.page_kinds,
            "pillar_changes": s.pillar_changes,
            "llm_checked_pages": s.llm_pages,
        }

    @staticmethod
    def _plan_md(report: Report, s: DocumentStructure, chunks: list[Chunk]) -> str:
        lines = [f"# 区切りの判定結果: {report.name}", "",
                 f"PDF: {', '.join(p.name for p in report.pdfs)} / 全{len(report.pages)}ページ", "",
                 "## チャンク", "",
                 "| ID | 種類 | ページ(通し) | 開始 | 終了 | 見出し | 図 |", "|---|---|---|---|---|---|---|"]
        for c in chunks:
            start = "先頭" if c.start.order is None else f"行{c.start.order}から"
            end = "末尾" if c.end_before is None else f"行{c.end_before}の手前"
            pages = f"{c.start.page}" if c.start.page == c.end_page else f"{c.start.page}-{c.end_page}"
            crumb = " > ".join(c.breadcrumb + [c.title])
            lines.append(f"| {c.id} | {c.kind} | {pages} | {start} | {end} | {crumb} | {len(c.figures)} |")
        lines += ["", "## 見出し", "", "| 通しページ | 階層 | 見出し | 根拠 | 点数 |", "|---|---|---|---|---|"]
        for h in s.headings:
            lines.append(f"| {h.global_page} | {h.level if h.level < 9 else '?'} | {h.text} | {', '.join(h.sources)} | {h.score:.1f} |")
        if s.unlocated_toc:
            lines += ["", "## 本文で位置を特定できなかった目次の項目", ""]
            lines += [f"- {e.title}(印刷 {e.printed_page}ページ)" for e in s.unlocated_toc]
        kinds = {}
        for no, k in s.page_kinds.items():
            kinds.setdefault(k, []).append(no)
        lines += ["", "## ページの種類", ""] + [f"- {k}: {v}" for k, v in sorted(kinds.items())]
        return "\n".join(lines) + "\n"