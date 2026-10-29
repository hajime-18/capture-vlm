"""見出しの書式による判定(ルールベース)

各行に点数を付け、Config.heading_accept_score 以上を見出し、
Config.heading_candidate_score 以上を「見出しかもしれない」候補とする(候補はVLMで確認する)。
"""

import re
from dataclasses import dataclass, field

from .ocr_page import Line, OcrPage

NUM = r"[0-9一二三四五六七八九十百〇]+"

# (書式名, 正規表現, 標準の階層順位)。順位は小さいほど上位(章 > 節 > 項 ...)
STYLES: list[tuple[str, re.Pattern, float]] = [
    ("chapter", re.compile(rf"^第{NUM}章"), 1.0),
    ("part", re.compile(rf"^第{NUM}部"), 0.5),
    ("roman", re.compile(r"^(?=[IVX]{1,5}(?:[\.・\s]|[^\x00-\x7F]))[IVX]{1,5}[\.・]?(?=[^\x00-\x7F])"), 1.2),
    ("section", re.compile(rf"^第{NUM}節"), 2.0),
    ("item", re.compile(rf"^第{NUM}項"), 3.0),
    ("arabic_dot", re.compile(r"^\d{1,2}[\.．](?!\d)\S"), 3.1),
    ("arabic_space", re.compile(r"^\d{1,2}(?=[^\x00-\x7F])(?![年月日号次回期点個基棟軒m])"), 3.2),
    ("paren_num", re.compile(r"^[\(（]\d{1,2}[\)）]"), 4.0),
    ("rparen_num", re.compile(r"^\d{1,2}[\)）]"), 4.5),
    ("circled", re.compile(r"^[①-⑳]"), 5.0),
    ("katakana", re.compile(r"^[ア-ン][\.．、\)）]"), 6.0),
    ("paren_kata", re.compile(r"^[\(（][ア-ン][\)）]"), 6.5),
    ("alpha", re.compile(r"^[A-Za-z][\.．\)）](?=[^\x00-\x7F])"), 7.0),
]

# 前付け・後付けなど、番号の無い大見出し(常に最上位として扱う)
SPECIAL_TITLES = {
    "序", "序文", "序言", "はじめに", "刊行にあたって", "例言", "凡例", "目次", "本文目次", "目次(本文)",
    "挿図目次", "挿表目次", "表目次", "図目次", "図版目次", "写真図版目次", "写真目次", "付図目次", "附図目次",
    "報告書抄録", "抄録", "奥付", "あとがき", "おわりに", "結語", "まとめ", "総括",
    "参考文献", "引用文献", "引用・参考文献", "引用参考文献", "参考・引用文献", "註", "注",
    "付編", "附編", "付章", "付篇", "附篇", "付録", "写真図版", "図版", "図面",
}

# 図表のキャプションは見出しではない
CAPTION_RE = re.compile(r"^(第?\d+(図|表)|図版?\d|表\d|写真\d|挿図\d|PL\.?\d|Fig\.?\s*\d|Tab\.?\s*\d)", re.I)


def style_of(line: Line) -> str | None:
    norm = line.norm
    raw = line.text.strip()
    if norm in SPECIAL_TITLES:
        return "special"
    if CAPTION_RE.match(norm):
        return None
    if re.match(r"^[①-⑳]", raw):
        return "circled"
    for name, pattern, _ in STYLES:
        if pattern.match(norm):
            return name
    return None


def default_rank(style: str) -> float:
    if style == "special":
        return 0.0
    return next(rank for name, _, rank in STYLES if name == style)


@dataclass
class HeadingCandidate:
    page: OcrPage
    line: Line
    style: str | None
    score: float
    reasons: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)  # rule / toc / pillar / llm
    toc_level: int | None = None
    llm_level: int | None = None
    llm_text: str | None = None   # VLMが画像から読み取った見出しの正しい表記

    @property
    def key(self) -> tuple[int, int]:
        return (self.page.ref.global_no, self.line.id)


class HeadingRuleDetector:
    def detect(self, page: OcrPage) -> list[HeadingCandidate]:
        out = []
        for line in page.content_lines:
            norm = line.norm
            if not norm or CAPTION_RE.match(norm) or line.type == "キャプション":
                continue
            style, score, reasons = self.score_line(page, line)
            if score > 0 and (style or line.type == "タイトル本文"):
                out.append(HeadingCandidate(page, line, style, score, reasons, {"rule"}))
        return out

    @staticmethod
    def score_line(page: OcrPage, line: Line) -> tuple[str | None, float, list[str]]:
        """書式・大きさ・長さなどから、見出しらしさの点数を付ける(書式に当てはまらない行にも使う)"""
        body_h = page.body_height
        norm = line.norm
        style = style_of(line)
        score, reasons = 0.0, []
        if style == "special":
            score += 3; reasons.append("前付け・後付けの見出し語")
        elif style:
            score += 2; reasons.append(f"見出しの書式({style})")
        if line.type == "タイトル本文":
            score += 2; reasons.append("OCRの種別が見出し")
        ratio = line.height / body_h if body_h else 1.0
        if ratio >= 1.15:
            score += 1; reasons.append(f"文字が大きい(本文の{ratio:.1f}倍)")
        if ratio >= 1.4:
            score += 1
        if len(norm) <= 25:
            score += 1
        elif len(norm) > 40:
            score -= 2; reasons.append("長い(本文の可能性)")
        if norm.endswith(("。", "、", ")")):
            score -= 3; reasons.append("句読点で終わる")
        return style, score, reasons