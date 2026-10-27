"""図・写真に報告書内で通しの番号(F0001, F0002, ...)を振り、切り出して保存する

Markdownでは、図があった位置に ![キャプション](figures/F0001.png) と書かせることで、
元の位置と切り出した画像を対応づける。
"""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .headings import CAPTION_RE
from .ocr_page import Block, OcrPage


@dataclass
class Figure:
    id: str
    page: OcrPage
    block: Block
    caption: str

    @property
    def global_page(self) -> int:
        return self.page.ref.global_no

    @property
    def rel_path(self) -> str:
        return f"figures/{self.id}.png"

    def to_dict(self) -> dict:
        b = self.block
        return {"id": self.id, "page": self.global_page, "type": b.type,
                "boundingBox": [[b.x0, b.y0], [b.x0, b.y1], [b.x1, b.y0], [b.x1, b.y1]],
                "confidence": b.conf, "caption": self.caption, "image": self.rel_path}


class FigureRegistry:
    def __init__(self, pages: list[OcrPage]):
        self.figures: list[Figure] = []
        n = 0
        for page in pages:
            for block in sorted(page.figures, key=lambda b: (b.y0, b.x0)):
                n += 1
                self.figures.append(Figure(f"F{n:04d}", page, block, self._caption(page, block)))

    @staticmethod
    def _caption(page: OcrPage, fig: Block) -> str:
        """図の上下にある、横位置の重なるキャプション行のうち最も近いもの"""
        best, best_dist = "", float("inf")
        for line in page.content_lines:
            if not (line.type == "キャプション" or CAPTION_RE.match(line.norm)):
                continue
            if min(line.x1, fig.x1) - max(line.x0, fig.x0) <= 0:
                continue
            dist = line.y0 - fig.y1 if line.y0 >= fig.y1 else fig.y0 - line.y1
            if -page.height * 0.02 <= dist <= page.height * 0.08 and dist < best_dist:
                best, best_dist = line.text.strip(), dist
        return best

    def on_page(self, global_no: int) -> list[Figure]:
        return [f for f in self.figures if f.global_page == global_no]

    def save_crops(self, out_dir: Path) -> int:
        """VLM用のページ画像から図を切り出す(OCRとVLM用画像で解像度が違うため座標を換算する)"""
        out_dir.mkdir(parents=True, exist_ok=True)
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
                img.crop(box).save(out_dir / f"{f.id}.png")
                saved += 1
        return saved