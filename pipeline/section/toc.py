"""目次ページを見つけて、見出しと印刷上のページ番号の一覧を読み取る"""

import re
from dataclasses import dataclass

from .headings import style_of
from .ocr_page import Line, OcrPage, normalize

TOC_TITLE_RE = re.compile(r"^(本文)?目次(\(本文\))?$|^もくじ$|^contents$", re.I)
SUB_TOC_RE = re.compile(r"^(挿図|挿表|表|図|図版|写真図版|写真|付図|附図|挿入写真|図面)目次$")
# 「第1節 調査の経緯 ………… 12」のような行。末尾の数字がページ番号
ENTRY_RE = re.compile(r"^(?P<title>.*?[^\d\s.・…‥\-ー―─—_])[\s.・…‥\-ー―─—_]*(?P<page>\d{1,4})$")
LEADERS = "・…‥.-ー―─—_ 　"


@dataclass
class TocEntry:
    title: str
    printed_page: int | None   # OCRでページ番号が読めなかった場合は None
    style: str | None
    level: int = 0           # 目次での階層(1始まり)
    toc_page: int = 0        # 目次が載っているページ(通し番号)
    located_page: int | None = None
    located_line: int | None = None


class TocParser:
    def find_toc_pages(self, pages: list[OcrPage]) -> list[OcrPage]:
        """本文の目次ページ(続くページも含む)。挿図目次などは含めない"""
        result = []
        i = 0
        while i < len(pages):
            if self._has_title(pages[i], TOC_TITLE_RE):
                result.append(pages[i])
                j = i + 1
                while j < len(pages) and not self._has_title(pages[j], SUB_TOC_RE) and self._entry_ratio(pages[j]) >= 0.4:
                    result.append(pages[j])
                    j += 1
                i = j
            else:
                i += 1
        return result

    def is_sub_toc(self, page: OcrPage) -> bool:
        return self._has_title(page, SUB_TOC_RE)

    @staticmethod
    def _has_title(page: OcrPage, pattern: re.Pattern) -> bool:
        top = [l for l in page.content_lines if l.y0 < page.height * 0.35][:6]
        return any(pattern.match(l.norm) for l in top)

    def _entry_ratio(self, page: OcrPage) -> float:
        lines = page.content_lines
        if not lines:
            return 0.0
        return sum(1 for l in lines if ENTRY_RE.match(l.norm)) / len(lines)

    def parse(self, toc_pages: list[OcrPage]) -> list[TocEntry]:
        entries: list[tuple[TocEntry, Line, float]] = []
        for page in toc_pages:
            pending = ""  # ページ番号の無い行(2行に折り返した見出しの前半)
            for line in self._ordered(page):
                norm = line.norm
                if TOC_TITLE_RE.match(norm) or SUB_TOC_RE.match(norm):
                    continue
                m = ENTRY_RE.match(norm)
                if not m:
                    title = norm.rstrip(LEADERS)
                    style = style_of(Line(0, 0, title, "本文", 0, 0, 0, 0, 0))
                    if style and len(title) <= 40:
                        # 見出しの書式なのにページ番号が無い(OCRで数字が抜けた)。前後の項目の間で探す
                        col_left = self._column_left(page, line)
                        entries.append((TocEntry(title, None, style, toc_page=page.ref.global_no),
                                        line, (line.x0 - col_left) / max(page.body_height, 1)))
                        pending = ""
                    else:
                        pending = norm if len(norm) <= 40 else ""
                    continue
                title = m["title"].rstrip(LEADERS)
                if pending and style_of(Line(0, 0, title, "本文", 0, 0, 0, 0, 0)) is None:
                    title = pending + title
                pending = ""
                if not title or re.fullmatch(r"[\d\s]+", title):
                    continue
                style = style_of(Line(0, 0, title, "本文", 0, 0, 0, 0, 0))
                col_left = self._column_left(page, line)
                entries.append((TocEntry(title, int(m["page"]), style, toc_page=page.ref.global_no),
                                line, (line.x0 - col_left) / max(page.body_height, 1)))
        self._assign_levels(entries)
        return [e for e, _, _ in entries]

    @staticmethod
    def _ordered(page: OcrPage) -> list[Line]:
        """2段組の目次にも対応するため、左の段→右の段の順に並べる"""
        mid = page.width * 0.5
        return sorted(page.content_lines, key=lambda l: (l.x0 >= mid, l.y0))

    @staticmethod
    def _column_left(page: OcrPage, line: Line) -> int:
        mid = page.width * 0.5
        same = [l.x0 for l in page.content_lines if (l.x0 >= mid) == (line.x0 >= mid)]
        return min(same) if same else 0

    @staticmethod
    def _assign_levels(entries: list[tuple[TocEntry, Line, float]]) -> None:
        """字下げの深さで階層を決める。字下げに差が無ければ書式の標準順位で決める"""
        from .headings import default_rank
        indents = sorted({round(ind) for _, _, ind in entries})
        clusters: list[int] = []
        for ind in indents:  # 1文字分以内の差は同じ字下げとみなす
            if not clusters or ind - clusters[-1] > 1:
                clusters.append(ind)
        if len(clusters) >= 2:
            for e, _, ind in entries:
                e.level = 1 + max(i for i, c in enumerate(clusters) if round(ind) >= c)
            return
        ranks = sorted({default_rank(e.style) for e, _, _ in entries if e.style})
        for e, _, _ in entries:
            e.level = 1 + ranks.index(default_rank(e.style)) if e.style else 1