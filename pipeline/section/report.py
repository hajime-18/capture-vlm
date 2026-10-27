"""報告書(1冊)の単位を扱う。1冊が複数のPDFに分割されている場合は1冊にまとめる。"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pypdfium2 as pdfium

from ..config import Config


@dataclass
class PageRef:
    """報告書の中の1ページ。global_no は報告書全体での通し番号(1始まり)"""
    global_no: int
    pdf_stem: str
    page_no: int
    image_path: Path
    json_path: Path

    @property
    def label(self) -> str:
        return f"{self.pdf_stem} {self.page_no}ページ目"


@dataclass
class Report:
    name: str
    pdfs: list[Path]
    pages: list[PageRef] = field(default_factory=list)


class ReportCatalog:
    """PDFのファイル名の規則(Config.report_patterns)から、どのPDFが同じ報告書かを判定する"""

    def __init__(self, config: Config, pdfs: list[Path]):
        self.config = config
        self.reports = self._build(pdfs)

    def _split_name(self, stem: str) -> tuple[str, int]:
        for pattern in self.config.report_patterns:
            m = re.match(pattern, stem)
            if m:
                return m["report"], int(m["order"])
        return stem, 0

    def _build(self, pdfs: list[Path]) -> list[Report]:
        groups: dict[str, list[tuple[int, Path]]] = {}
        for pdf in dict.fromkeys(pdfs):  # 重複を除く
            name, order = self._split_name(pdf.stem)
            groups.setdefault(name, []).append((order, pdf))
        reports = []
        for name, items in sorted(groups.items()):
            ordered = [p for _, p in sorted(items, key=lambda t: (t[0], t[1].name))]
            reports.append(Report(name=name, pdfs=ordered, pages=self._pages(ordered)))
        return reports

    def _pages(self, pdfs: list[Path]) -> list[PageRef]:
        pages, n = [], 0
        for pdf in pdfs:
            doc = pdfium.PdfDocument(pdf)
            try:
                count = len(doc)
            finally:
                doc.close()
            for i in range(1, count + 1):
                n += 1
                stem = self.config.page_stem(pdf.stem, i)
                pages.append(PageRef(
                    global_no=n, pdf_stem=pdf.stem, page_no=i,
                    image_path=self.config.image_dir / f"{stem}.png",
                    json_path=self.config.json_dir / f"{stem}.json",
                ))
        return pages

    def containing(self, pdfs: list[Path]) -> list[Report]:
        """指定したPDFのいずれかを含む報告書"""
        stems = {p.stem for p in pdfs}
        return [r for r in self.reports if stems & {p.stem for p in r.pdfs}]