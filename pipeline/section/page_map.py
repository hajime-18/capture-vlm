"""印刷されたページ番号(ノンブル)と、PDFの通し番号を対応づける

ノンブルは小さな数字なので、OCRで読み誤ることが多い(特に1桁)。
そのため、前後のページと番号が連続している(通し番号が1増えると印刷番号も1増える)ものだけを信頼する。
"""

from collections import Counter

from .ocr_page import OcrPage


class PageMapper:
    def __init__(self, pages: list[OcrPage]):
        raw = {p.ref.global_no: p.folio for p in pages if p.folio is not None}
        self.anchors = [(g, f) for g, f in raw.items() if self._consistent(raw, g, f)]
        offsets = [g - f for g, f in self.anchors]
        self.default_offset = Counter(offsets).most_common(1)[0][0] if offsets else None

    @staticmethod
    def _consistent(raw: dict[int, int], g: int, f: int) -> bool:
        """前後2ページ以内に、番号の差がページの差と一致するノンブルがあるか"""
        return any(raw.get(g + d) == f + d for d in (-2, -1, 1, 2))

    def to_global(self, printed: int) -> int | None:
        """印刷上のページ番号 → 通し番号。近くの信頼できるノンブルのずれ(最頻値)を使う"""
        near = [g - f for g, f in self.anchors if abs(f - printed) <= 15]
        if near:
            return printed + Counter(near).most_common(1)[0][0]
        return printed + self.default_offset if self.default_offset is not None else None