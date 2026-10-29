"""複数の手がかりを総合して、報告書の見出し(章・節など)を確定する

  1. 目次      : 見出しの一覧と階層、印刷上のページ番号
  2. 書式ルール: 見出しの書き方・文字の大きさ・OCRの種別による点数
  3. 柱        : ページ上部の章題が変わったページ
  4. VLM       : 1〜3で判断がつかない箇所を、ページ画像とOCR結果で確認(Config.boundary_llm)
"""

import difflib
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..config import Config
from .headings import HeadingCandidate, HeadingRuleDetector, default_rank, style_of
from .llm_judge import LlmBoundaryJudge
from .ocr_page import Line, OcrPage, normalize
from .page_map import PageMapper
from .toc import TocEntry, TocParser


@dataclass
class Heading:
    page: OcrPage
    line_id: int
    order: int
    text: str            # 見出しの表記(VLMが読み直した場合はその表記)
    level: int
    score: float
    sources: list[str]
    style: str | None
    ocr_text: str = ""   # OCRが読み取った表記

    @property
    def global_page(self) -> int:
        return self.page.ref.global_no

    def to_dict(self) -> dict:
        d = {"page": self.global_page, "line_id": self.line_id, "text": self.text, "level": self.level,
             "score": round(self.score, 2), "sources": self.sources, "style": self.style}
        if self.ocr_text and self.ocr_text != self.text:
            d["ocr_text"] = self.ocr_text
        return d


@dataclass
class DocumentStructure:
    headings: list[Heading]
    toc_entries: list[TocEntry]
    toc_pages: list[int]
    page_kinds: dict[int, str]
    pillar_changes: list[int]
    llm_pages: list[int]
    unlocated_toc: list[TocEntry] = field(default_factory=list)


class StructureAnalyzer:
    def __init__(self, config: Config, judge: LlmBoundaryJudge | None, log=print):
        self.config = config
        self.judge = judge
        self.log = log
        self.rules = HeadingRuleDetector()
        self.toc = TocParser()

    def analyze(self, pages: list[OcrPage]) -> DocumentStructure:
        cfg = self.config
        by_no = {p.ref.global_no: p for p in pages}

        # --- 1. 目次 ---
        toc_pages = self.toc.find_toc_pages(pages)
        toc_nos = {p.ref.global_no for p in toc_pages}
        entries = self.toc.parse(toc_pages)
        mapper = PageMapper(pages)

        # --- 2. 書式ルール(目次ページの各行は見出しに似ているので除く。表題「目次」だけは残す) ---
        cands: dict[tuple[int, int], HeadingCandidate] = {}
        for p in pages:
            for c in self.rules.detect(p):
                if p.ref.global_no in toc_nos and c.style != "special":
                    continue
                if self.toc.is_sub_toc(p) and c.style != "special":
                    continue
                cands[c.key] = c

        # --- 目次の見出しを本文中の行に対応づける ---
        unlocated = self._locate_toc(entries, pages, toc_nos, mapper, cands)
        if entries:
            self.log(f"目次: {len(entries)}項目(本文中で位置を特定: {len(entries) - len(unlocated)}項目)")

        # --- 3. 柱の変化 ---
        changes = self._pillar_changes(pages)
        for no in changes:
            for c in cands.values():
                if c.page.ref.global_no == no and c.style and default_rank(c.style) <= 1.2:
                    c.score += 1
                    c.sources.add("pillar")

        # --- 4. VLM(ページ画像とOCR結果) ---
        llm_pages = self._llm_targets(pages, cands, changes, unlocated, mapper, toc_nos)
        if self.judge and llm_pages:
            how = "画像とOCR結果" if self.config.boundary_use_image else "OCR結果"
            self.log(f"VLMで見出しを確認({how}): {len(llm_pages)}ページ")
            for no in llm_pages:
                self._apply_llm(by_no[no], cands, self._nearby_toc(entries, mapper, no))

        headings = self._finalize(cands, entries)
        return DocumentStructure(
            headings=headings, toc_entries=entries, toc_pages=sorted(toc_nos),
            page_kinds={p.ref.global_no: ("toc" if p.ref.global_no in toc_nos else p.kind) for p in pages},
            pillar_changes=changes, llm_pages=llm_pages if self.judge else [], unlocated_toc=unlocated,
        )

    # ---------- 目次との対応づけ ----------
    def _locate_toc(self, entries, pages, toc_nos, mapper, cands) -> list[TocEntry]:
        """
        目次の各項目が本文のどの行かを探す。前の項目より後ろのページから順に探す(順序は逆転しない)。
        探す範囲は、信頼できるノンブルがあればそこから、無ければ直前に見つけた項目の
        「印刷番号と通し番号のずれ」から予想する。
        """
        cfg = self.config
        body_pages = [p for p in pages if p.ref.global_no not in toc_nos]
        unlocated, last = [], (body_pages[0].ref.global_no if body_pages else 1)
        learned_offset = None
        self._learned_offsets: list[tuple[int, int]] = []  # (印刷番号, ずれ)
        for e in entries:
            if e.printed_page is None:
                expected = None
            else:
                expected = mapper.to_global(e.printed_page)
                if expected is None and learned_offset is not None:
                    expected = e.printed_page + learned_offset
            lo = max(last, (expected - 1) if expected else last)
            hi = (expected + 2) if expected else last + (5 if e.printed_page is None else 15)
            best, best_ratio = None, 0.0
            target = e.title.replace(" ", "")
            for p in body_pages:
                no = p.ref.global_no
                if not (lo <= no <= hi):
                    continue
                for line in p.content_lines:
                    norm = line.norm
                    if not norm or line.type == "キャプション":
                        continue
                    head = norm[: len(target) + 2]
                    ratio = difflib.SequenceMatcher(None, target, head).ratio()
                    if (no, line.id) in cands:
                        ratio += 0.1
                    if ratio > best_ratio:
                        best, best_ratio = (p, line), ratio
            if best and best_ratio >= 0.6:
                p, line = best
                e.located_page, e.located_line = p.ref.global_no, line.id
                last = p.ref.global_no
                if e.printed_page is not None:
                    learned_offset = p.ref.global_no - e.printed_page
                    self._learned_offsets.append((e.printed_page, learned_offset))
                key = (p.ref.global_no, line.id)
                c = cands.get(key)
                if c is None:
                    c = HeadingCandidate(p, line, e.style, cfg.heading_accept_score, ["目次に一致"], set())
                    cands[key] = c
                else:
                    c.score += 3
                    c.reasons.append("目次に一致")
                c.sources.add("toc")
                c.toc_level = e.level
            else:
                unlocated.append(e)
        return unlocated

    @staticmethod
    def _pillar_changes(pages: list[OcrPage]) -> list[int]:
        changes, prev = [], ""
        for p in pages:
            cur = p.pillar_text
            if not cur:
                continue
            if prev and difflib.SequenceMatcher(None, prev, cur).ratio() < 0.6:
                changes.append(p.ref.global_no)
            prev = cur
        return changes

    # ---------- VLM ----------
    def _llm_targets(self, pages, cands, changes, unlocated, mapper, toc_nos) -> list[int]:
        cfg = self.config
        mode = cfg.boundary_llm
        if mode == "off" or self.judge is None:
            return []
        text_pages = [p.ref.global_no for p in pages if p.kind == "text" and p.ref.global_no not in toc_nos]
        if mode == "all":
            return text_pages
        targets = set()
        for c in cands.values():  # 点数が中間の候補(目次で確認できたものは除く)
            if cfg.heading_candidate_score <= c.score < cfg.heading_accept_score and "toc" not in c.sources:
                targets.add(c.page.ref.global_no)
        for no in changes:  # 柱が変わったのに大見出しが見つからないページ
            if not any(c.page.ref.global_no == no and c.score >= cfg.heading_accept_score for c in cands.values()):
                targets.add(no)
        for c in cands.values():  # 見出しらしいのに書式も目次も無く、階層が決まらない行(OCRの読み誤りの可能性)
            if c.score >= cfg.heading_accept_score and c.style is None and "toc" not in c.sources:
                targets.add(c.page.ref.global_no)
        for e in unlocated:  # 目次にあるのに本文で見つからない見出し
            g = self._expected_page(mapper, e.printed_page)
            if g:
                targets.update({g, g + 1})
        return sorted(t for t in targets if t in text_pages)

    def _expected_page(self, mapper: PageMapper, printed: int | None) -> int | None:
        """印刷番号から通し番号を予想する(ノンブル → 目次との対応で分かったずれ の順)"""
        if printed is None:
            return None
        g = mapper.to_global(printed)
        if g is not None:
            return g
        learned = getattr(self, "_learned_offsets", [])
        if not learned:
            return None
        _, offset = min(learned, key=lambda t: abs(t[0] - printed))
        return printed + offset

    def _nearby_toc(self, entries: list[TocEntry], mapper: PageMapper, no: int) -> list[TocEntry]:
        return [e for e in entries if (g := self._expected_page(mapper, e.printed_page)) and abs(g - no) <= 1]

    def _apply_llm(self, page: OcrPage, cands, toc_hint) -> None:
        cfg = self.config
        result = self.judge.judge(page, toc_hint)
        if result is None:
            self.log(f"[警告] VLMの応答を解釈できませんでした: 通し{page.ref.global_no}ページ")
            return
        no = page.ref.global_no
        for line_id, decision in result.items():
            level = decision.level
            key = (no, line_id)
            c = cands.get(key)
            if c is None:
                # ルールでは候補にならなかった行。VLMの判断だけで決めず、ルールの点数(長い文などは負)も加える
                line = page.line_by_id(line_id)
                style, score, reasons = HeadingRuleDetector.score_line(page, line)
                c = HeadingCandidate(page, line, style, max(score, 0) + 3 if score >= 0 else score + 3,
                                     reasons + ["VLMが見出しと判断"], set())
                cands[key] = c
            else:
                c.score += 3
                c.reasons.append("VLMが見出しと判断")
            c.sources.add("llm")
            c.llm_level = level
            self._apply_text_fix(c, decision.text)
        for c in cands.values():  # VLMが選ばなかった中間の候補は下げる
            if c.page.ref.global_no == no and "llm" not in c.sources and "toc" not in c.sources \
                    and c.score < cfg.heading_accept_score:
                c.score -= 2

    def _apply_text_fix(self, c: HeadingCandidate, text: str | None) -> None:
        """
        VLMが画像から読み直した表記を記録する。OCRでは書式が分からなかった行(例:「序」が「17」と読まれた)は、
        読み直した表記で書式を判定し直す(「序」なら前付けの見出し、「第2節…」なら節)。
        """
        if not text or normalize(text) == c.line.norm:
            return
        c.llm_text = text
        c.reasons.append(f"VLMが表記を修正: 「{c.line.text.strip()}」→「{text}」")
        if c.style is None:
            fixed = style_of(Line(c.line.id, c.line.order, text, c.line.type, c.line.x0, c.line.y0,
                                  c.line.x1, c.line.y1, c.line.conf))
            if fixed:
                c.style = fixed

    # ---------- 階層の決定 ----------
    def _finalize(self, cands, entries) -> list[Heading]:
        cfg = self.config
        accepted = [c for c in cands.values() if c.score >= cfg.heading_accept_score]
        style_level = self._style_levels(accepted, entries)
        headings = []
        for c in accepted:
            if c.style == "special":
                level = 1
            elif c.toc_level:
                level = c.toc_level
            elif c.style in style_level:
                level = style_level[c.style]
            elif c.llm_level:
                level = c.llm_level
            else:
                level = 9  # 階層が分からない見出しは区切りに使わない
            headings.append(Heading(c.page, c.line.id, c.line.order, (c.llm_text or c.line.text).strip(), level,
                                    c.score, sorted(c.sources), c.style, ocr_text=c.line.text.strip()))
        headings.sort(key=lambda h: (h.global_page, h.order))
        return headings

    @staticmethod
    def _style_levels(accepted: list[HeadingCandidate], entries: list[TocEntry]) -> dict[str, int]:
        """書式ごとの階層。目次があればそこでの階層の多数決、無ければ標準の順位の並びで決める"""
        votes: dict[str, Counter] = defaultdict(Counter)
        for e in entries:
            if e.style and e.style != "special":
                votes[e.style][e.level] += 1
        levels = {s: cnt.most_common(1)[0][0] for s, cnt in votes.items()}
        rest = sorted({c.style for c in accepted if c.style and c.style != "special" and c.style not in levels},
                      key=default_rank)
        used = sorted(set(levels.values()))
        top = max(used) if used else 0
        for style in rest:
            # 目次に出てこない書式は、目次の階層より下位とみなす(目次が無ければ1から順に)
            higher = [s for s in levels if default_rank(s) < default_rank(style)]
            levels[style] = (max(levels[s] for s in higher) + 1) if higher else (top + 1 if used else 1)
            if not used:
                top = levels[style]
        if not entries:  # 目次が無い場合は、標準の順位の並びをそのまま階層にする
            levels = {s: i + 1 for i, s in enumerate(sorted({c.style for c in accepted if c.style and c.style != "special"},
                                                            key=default_rank))}
        return levels