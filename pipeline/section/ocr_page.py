"""OCR結果(1ページ分)を、区切りの判定に使いやすい形で扱う"""

import json
import re
import statistics
import unicodedata
from dataclasses import dataclass

from .report import PageRef

FIGURE_BLOCK_TYPES = {"図版", "組織図"}   # 図として扱うブロック
NON_CONTENT_TYPES = {"柱", "ノンブル"}     # 本文の内容ではない行


def normalize(text: str) -> str:
    """全角・半角の揺れと空白をなくす(比較用)"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))


def _box(bbox) -> tuple[int, int, int, int]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


@dataclass
class Line:
    id: int
    order: int
    text: str
    type: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float

    @property
    def norm(self) -> str:
        return normalize(self.text)

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def is_content(self) -> bool:
        return self.type not in NON_CONTENT_TYPES


@dataclass
class Block:
    type: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float
    text: str = ""

    @property
    def area(self) -> int:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)


class OcrPage:
    def __init__(self, ref: PageRef, data: dict):
        self.ref = ref
        self.data = data  # 保存用に元のJSONをそのまま持つ
        raw_lines = data["contents"][0] if data.get("contents") else []
        lines = []
        for raw in raw_lines:
            x0, y0, x1, y1 = _box(raw["boundingBox"])
            lines.append(Line(
                id=int(raw["id"]), order=int(raw.get("order", raw["id"])), text=raw.get("text", ""),
                type=raw.get("type", "本文"), x0=x0, y0=y0, x1=x1, y1=y1,
                conf=float(raw.get("confidence", 0) or 0),
            ))
        self.lines = sorted(lines, key=lambda l: l.order)  # 読み順
        self.blocks = []
        for raw in data.get("blocks", []):
            x0, y0, x1, y1 = _box(raw["boundingBox"])
            self.blocks.append(Block(raw.get("type", ""), x0, y0, x1, y1, float(raw.get("confidence", 0)),
                                     raw.get("text", "")))
        size = data.get("page") or {}
        info = data.get("imginfo") or {}
        self.width = int(size.get("width") or info.get("img_width") or 1)
        self.height = int(size.get("height") or info.get("img_height") or 1)

    @classmethod
    def load(cls, ref: PageRef) -> "OcrPage":
        return cls(ref, json.loads(ref.json_path.read_text(encoding="utf-8")))

    # ---------- 行の分類 ----------
    @property
    def content_lines(self) -> list[Line]:
        return [l for l in self.lines if l.is_content]

    @property
    def body_height(self) -> float:
        """本文の文字の高さの代表値(見出しの大きさの比較に使う)"""
        hs = [l.height for l in self.lines if l.type == "本文" and len(l.norm) >= 8]
        hs = hs or [l.height for l in self.content_lines]
        return float(statistics.median(hs)) if hs else 1.0

    def _texts_of(self, kind: str) -> list[str]:
        """行として読まれたもの(柱・ノンブルの種別の行)と、ブロックとして読み足したものの両方"""
        return [l.norm for l in self.lines if l.type == kind] + \
               [normalize(b.text) for b in self.blocks if b.type == kind and b.text]

    @property
    def pillar_text(self) -> str:
        return re.sub(r"\d+", "", "".join(self._texts_of("柱")))

    @property
    def folio(self) -> int | None:
        """ノンブル(印刷されたページ番号)"""
        for text in self._texts_of("ノンブル"):
            m = re.fullmatch(r"[-－‐–—(（]?(\d{1,4})[-－‐–—)）]?", text)
            if m:
                return int(m.group(1))
        return None

    @property
    def figures(self) -> list[Block]:
        return [b for b in self.blocks if b.type in FIGURE_BLOCK_TYPES]

    @property
    def kind(self) -> str:
        """text(本文) / plate(図版が中心) / blank(内容なし)"""
        body_chars = sum(len(l.norm) for l in self.content_lines if l.type != "キャプション")
        fig_ratio = sum(b.area for b in self.figures) / float(self.width * self.height)
        if body_chars == 0 and not self.figures:
            return "blank"
        if fig_ratio >= 0.3 and body_chars < 150:
            return "plate"
        captions = [l for l in self.content_lines if l.type == "キャプション" or re.match(r"^(図版|写真|PL\.?)\d", l.norm)]
        if self.figures and body_chars < 60 and captions:
            return "plate"
        return "text"

    def lines_before(self, order: int) -> list[Line]:
        return [l for l in self.content_lines if l.order < order]

    def line_by_id(self, line_id: int) -> Line | None:
        return next((l for l in self.lines if l.id == line_id), None)