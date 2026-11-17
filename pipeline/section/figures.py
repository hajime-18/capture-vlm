"""図・写真と表に報告書内で通しの番号を振り、切り出して保存する

- 図・写真: F0001, F0002, ... → figures/F0001.png
  Markdownでは、図があった位置に ![キャプション](figures/F0001.png) と書かせる。
- 表: T0001, T0002, ... → tables/T0001.png(切り出し画像)と tables/T0001.html(表の中身。vlm で作る)
  Markdownでは、表があった位置に [キャプション](tables/T0001.html) と書かせる。
これで、元の位置と切り出したものを対応づける。
"""

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .headings import CAPTION_RE
from .ocr_page import TABLE_BLOCK_TYPES, Block, Line, OcrPage

# 図の近くにある表のキャプション、表の近くにある図のキャプションを取り違えないための除外
TABLE_CAPTION_RE = re.compile(r"^(第?\d+表|表\d|Tab\.?\s*\d)", re.I)
FIGURE_CAPTION_RE = re.compile(r"^(第?\d+図|図版?\d|写真\d|挿図\d|PL\.?\d|Fig\.?\s*\d)", re.I)


@dataclass
class Figure:
    id: str
    page: OcrPage
    block: Block
    caption: str

    folder = "figures"

    @property
    def global_page(self) -> int:
        return self.page.ref.global_no

    @property
    def rel_path(self) -> str:
        """切り出し画像の場所(output_md/{報告書名}/ からの相対パス)"""
        return f"{self.folder}/{self.id}.png"

    def to_dict(self) -> dict:
        b = self.block
        return {"id": self.id, "page": self.global_page, "type": b.type,
                "boundingBox": [[b.x0, b.y0], [b.x0, b.y1], [b.x1, b.y0], [b.x1, b.y1]],
                "confidence": b.conf, "caption": self.caption, "image": self.rel_path}


@dataclass
class Table(Figure):
    folder = "tables"

    @property
    def html_path(self) -> str:
        """表の中身(HTML)の場所(output_md/{報告書名}/ からの相対パス)"""
        return f"{self.folder}/{self.id}.html"

    def inner_lines(self) -> list[Line]:
        """表の領域の中にあるOCRの行(行の中心が領域内にあるもの)"""
        b = self.block
        return [l for l in self.page.content_lines
                if b.x0 <= (l.x0 + l.x1) / 2 <= b.x1 and b.y0 <= (l.y0 + l.y1) / 2 <= b.y1]

    def to_dict(self) -> dict:
        return super().to_dict() | {"html": self.html_path, "line_ids": [l.id for l in self.inner_lines()]}


class FigureRegistry:
    """図(tables=False)または表(tables=True)の一覧"""

    def __init__(self, pages: list[OcrPage], tables: bool = False):
        self.figures: list[Figure] = []
        cls, prefix, exclude = (Table, "T", FIGURE_CAPTION_RE) if tables else (Figure, "F", TABLE_CAPTION_RE)
        n = 0
        for page in pages:
            blocks = [b for b in page.blocks if b.type in TABLE_BLOCK_TYPES] if tables else page.figures
            for block in sorted(blocks, key=lambda b: (b.y0, b.x0)):
                n += 1
                self.figures.append(cls(f"{prefix}{n:04d}", page, block, self._caption(page, block, exclude)))

    @staticmethod
    def _caption(page: OcrPage, fig: Block, exclude: re.Pattern) -> str:
        """図の上下にある、横位置の重なるキャプション行のうち最も近いもの(別の種類のキャプションは除く)"""
        best, best_dist = "", float("inf")
        for line in page.content_lines:
            if not (line.type == "キャプション" or CAPTION_RE.match(line.norm)) or exclude.match(line.norm):
                continue
            if min(line.x1, fig.x1) - max(line.x0, fig.x0) <= 0:
                continue
            dist = line.y0 - fig.y1 if line.y0 >= fig.y1 else fig.y0 - line.y1
            if -page.height * 0.02 <= dist <= page.height * 0.08 and dist < best_dist:
                best, best_dist = line.text.strip(), dist
        return best

    def on_page(self, global_no: int) -> list[Figure]:
        return [f for f in self.figures if f.global_page == global_no]

    def save_crops(self, report_dir: Path) -> int:
        """VLM用のページ画像から図を切り出す(OCRとVLM用画像で解像度が違うため座標を換算する)"""
        saved = 0
        cache: dict[Path, Image.Image] = {}
        for f in self.figures:
            img_path = f.page.ref.image_path
            if not img_path.exists():
                continue
            img = cache.setdefault(img_path, Image.open(img_path).convert("RGB"))
            sx, sy = img.width / f.page.width, img.height / f.page.height
            b = f.block
            pad = 4
            box = (max(0, int(b.x0 * sx) - pad), max(0, int(b.y0 * sy) - pad),
                   min(img.width, int(b.x1 * sx) + pad), min(img.height, int(b.y1 * sy) + pad))
            if box[2] > box[0] and box[3] > box[1]:
                out = report_dir / f.rel_path
                out.parent.mkdir(parents=True, exist_ok=True)
                img.crop(box).save(out)
                saved += 1
        return saved
