"""[vlm] チャンクごとに、複数ページの画像とOCR結果をVLMに入れてMarkdownにする(section モード)

最初に、対象のすべての報告書でチャンクが揃っていることを確認してから処理を始める。
表は本文とは別に、切り出した画像と表の中のOCR結果から1つずつHTMLにする(tables/T0001.html)。
"""

import json
import re

from ..base import PipelineError, Step
from ..vlm_converter import MarkdownCleaner, get_client
from .prompts import SectionPromptBuilder, build_table_prompt
from .table_html import build_table_html
from .validator import PlanValidator

FIG_REF_RE = re.compile(r"figures/(F\d{4})\.png")
TABLE_REF_RE = re.compile(r"tables/(T\d{4})\.html")


class SectionVlmConverter(Step):
    name = "VLM → Markdown(節ごと)"

    def run(self) -> None:
        validator = PlanValidator(self.config)
        reports = self.targets.reports
        self._check_all_ready(validator, reports)

        tasks, table_tasks = [], []
        for report in reports:
            plan_dir = validator.plan_dir(report)
            plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
            out_dir = self.config.output_dir / report.name / "chunks"
            out_dir.mkdir(parents=True, exist_ok=True)
            for item in plan["chunks"]:
                out = out_dir / f"chunk_{item['chunk_id']}.md"
                if self.config.overwrite or not out.exists():
                    tasks.append((report.name, plan_dir / item["file"], out))
                chunk = json.loads((plan_dir / item["file"]).read_text(encoding="utf-8"))
                for table in chunk.get("tables", []):
                    # 古い plan(表を XML にしていたころ)の chunk JSON には "html" が無いので、画像の場所から求める
                    rel = table.get("html") or table["image"].rsplit(".", 1)[0] + ".html"
                    html = self.config.output_dir / report.name / rel
                    if self.config.overwrite or not html.exists():
                        table_tasks.append((report.name, chunk, table, html))
        if not tasks and not table_tasks:
            self.log("Markdown化待ちのチャンク・表はありません")
            return

        client = get_client(self.config)
        if tasks:
            self.log(f"{len(tasks)}個のチャンクをMarkdownにします")
        self._convert_chunks(client, tasks)
        if table_tasks:
            self.log(f"{len(table_tasks)}個の表をHTMLにします")
        self._convert_tables(client, table_tasks)

    def _convert_chunks(self, client, tasks) -> None:
        builder, cleaner = SectionPromptBuilder(self.config), MarkdownCleaner()
        for i, (report_name, chunk_path, out) in enumerate(tasks, 1):
            chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
            pages = [p["global_page"] for p in chunk["pages"]]
            span = f"{pages[0]}" if pages[0] == pages[-1] else f"{pages[0]}-{pages[-1]}"
            self.log(f"[{i}/{len(tasks)}] {report_name} チャンク{chunk['chunk_id']}(通し{span}ページ) {chunk['title']}")
            try:
                images = [self.config.image_dir / p["image"] for p in chunk["pages"]]
                result = client.generate(images, builder.build(chunk), max_pixels=self.config.section_max_image_pixels)
                md = cleaner.clean(result.text)
                self.log(f"  出力 {result.num_tokens} トークン")
                md += self._check_refs(chunk, md, "figures", FIG_REF_RE, "図")
                md += self._check_refs(chunk, md, "tables", TABLE_REF_RE, "表")
                if result.truncated:
                    self.log("  [警告] 出力が上限に達して途中で打ち切られました"
                             "(config.py の max_new_tokens を増やすか、繰り返しが起きていないか確認してください)")
                    md += "\n<!-- warning: output truncated -->\n"
                out.write_text(md, encoding="utf-8")
            except Exception as e:
                self.log(f"  [エラー] {e}")

    def _check_all_ready(self, validator: PlanValidator, reports) -> None:
        """1つでも揃っていない報告書があれば、VLMを始めずに止める"""
        if not reports:
            raise PipelineError("対象の報告書がありません")
        errors = []
        for r in reports:
            for p in validator.problems(r):
                errors.append(f"  {r.name}: {p}")
        if errors:
            raise PipelineError("チャンクが揃っていないため、VLMの処理を始められません。\n" + "\n".join(errors) +
                                "\n  python main.py --only plan を実行してください。")
        total = sum(len(json.loads((validator.plan_dir(r) / "plan.json").read_text(encoding="utf-8"))["chunks"])
                    for r in reports)
        self.log(f"確認OK: {len(reports)}冊 / {total}個のチャンクが揃っています")

    def _convert_tables(self, client, tasks) -> None:
        """表を1つずつ、切り出した画像と表の中のOCR結果からHTMLにさせ、整えて保存する"""
        for i, (report_name, chunk, table, out) in enumerate(tasks, 1):
            page = next(p for p in chunk["pages"] if p["global_page"] == table["page"])
            self.log(f"[{i}/{len(tasks)}] {report_name} 表{table['id']}(通し{table['page']}ページ) {table['caption']}")
            try:
                image = self.config.output_dir / report_name / table["image"]
                result = client.generate(image, build_table_prompt(table, page["ocr"]),
                                         max_pixels=self.config.table_max_image_pixels)
                self.log(f"  出力 {result.num_tokens} トークン")
                html, ok = build_table_html(table, report_name, page, result.text)
                if not ok:
                    self.log("  [警告] 出力を表として読めませんでした(HTMLの <pre class='raw'> に出力をそのまま残します)")
                if result.truncated:
                    self.log("  [警告] 出力が上限に達して途中で打ち切られました")
                    html += "<!-- warning: output truncated -->\n"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(html, encoding="utf-8")
            except Exception as e:
                self.log(f"  [エラー] {e}")

    def _check_refs(self, chunk: dict, md: str, key: str, ref_re: re.Pattern, label: str) -> str:
        """一覧の図・表がすべて書き込まれたかを確認し、抜けていれば印を残す"""
        expected = {f["id"] for f in chunk.get(key, [])}
        found = set(ref_re.findall(md))
        notes = ""
        missing = sorted(expected - found)
        unknown = sorted(found - expected)
        if missing:
            self.log(f"  [警告] 出力に無い{label}: {', '.join(missing)}")
            notes += f"\n<!-- warning: missing {key}: {', '.join(missing)} -->\n"
        if unknown:
            self.log(f"  [警告] 一覧に無い{label}のID: {', '.join(unknown)}")
            notes += f"\n<!-- warning: unknown {key[:-1]} ids: {', '.join(unknown)} -->\n"
        return notes