"""見出しの位置から、VLMに1回で入れる範囲(チャンク)を決める

- 区切りは Config.section_level 以上の見出し(章・節)。
- 次の見出しがページの途中にある場合、そのページは前後両方のチャンクに入れる(重複させて取りこぼしを防ぐ)。
  各チャンクには「どの行から、どの行の手前まで」を scope として記録し、VLMにはその範囲だけを変換させる。
- 図版ページが Config.min_plate_run 枚以上続く部分は、図版だけのチャンクにする。
- Config.max_pages_per_chunk を超える範囲は、下位の見出し → ページ の順で分割する。
"""

from dataclasses import dataclass, field

from ..config import Config
from .analyzer import DocumentStructure, Heading
from .figures import Figure, FigureRegistry
from .ocr_page import Line, OcrPage


def is_before(line: Line, boundary: Line, page: OcrPage) -> bool:
    """
    行が境界の行(次の見出し)より前にあるか。基本はOCRの読み順だが、読み順が乱れることがあるため位置でも確かめる。
    - 読み順が前でも、見出しより下にあり、同じ段にある行は後ろとみなす(例: 見出しの下の表が先に読まれた場合)
    - 読み順が後でも、見出しより上にあり、同じ段にある行は前とみなす
    見出しが中央寄せ・全幅なら全体が同じ段、そうでなければ横位置が重なる行を同じ段とする(2段組への対応)。
    """
    if line.id == boundary.id:
        return False
    center = (boundary.x0 + boundary.x1) / 2
    full = abs(center - page.width / 2) < page.width * 0.1 or (boundary.x1 - boundary.x0) > page.width * 0.6
    same_column = full or (min(line.x1, boundary.x1) - max(line.x0, boundary.x0) > 0) or \
        (abs(line.x0 - boundary.x0) < page.width * 0.05)
    tol = boundary.height * 0.3
    if line.order < boundary.order:
        return not (same_column and line.y0 >= boundary.y1 - tol)
    return same_column and line.y1 <= boundary.y0 + tol


@dataclass(frozen=True)
class Pos:
    """報告書内の位置。order が None ならページの先頭(または末尾)"""
    page: int
    order: int | None = None


@dataclass
class Chunk:
    index: int
    kind: str                         # front / section / special / plate
    title: str
    level: int | None
    breadcrumb: list[str]
    continued: bool
    start: Pos
    end_page: int
    end_before: int | None            # この order の行の手前まで(None ならページ末尾まで)
    headings: list[Heading] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.index:04d}"

    @property
    def page_range(self) -> range:
        return range(self.start.page, self.end_page + 1)


class Chunker:
    def __init__(self, config: Config, pages: list[OcrPage], structure: DocumentStructure, figures: FigureRegistry):
        self.cfg = config
        self.pages = {p.ref.global_no: p for p in pages}
        self.first, self.last = min(self.pages), max(self.pages)
        self.structure = structure
        self.figures = figures
        self.all_headings = [h for h in structure.headings if h.level < 9]
        self.plate_pages = self._plate_runs()

    # ---------- 全体の流れ ----------
    def build(self) -> list[Chunk]:
        pieces = []
        for seg_start, seg_end, heading in self._segments():
            for start, end_page, end_before, plate in self._split_plates(seg_start, seg_end):
                if plate:
                    pieces.extend(self._plate_chunks(start.page, end_page))
                else:
                    pieces.extend(self._split_long(start, end_page, end_before, heading))
        pieces.sort(key=lambda c: (c.start.page, c.start.order if c.start.order is not None else -1))
        for i, c in enumerate(pieces, 1):
            c.index = i
            c.headings = self._headings_in(c)
            c.figures = self._figures_in(c)
        return pieces

    # ---------- 1. 章・節で区切る ----------
    def _segments(self) -> list[tuple[Pos, tuple[int, int | None], Heading | None]]:
        bounds = [h for h in self.all_headings if h.level <= self.cfg.section_level]
        starts: list[tuple[Pos, Heading | None]] = []
        if not bounds or not (bounds[0].global_page == self.first and not self._content_before(self.first, bounds[0].order)):
            starts.append((Pos(self.first), None))
        starts += [(Pos(h.global_page, h.order), h) for h in bounds]

        segments = []
        carry: tuple[Pos, Heading | None] | None = None  # 見出しだけで本文が無い区間は次に繋げる
        for i, (pos, heading) in enumerate(starts):
            nxt = starts[i + 1][0] if i + 1 < len(starts) else None
            end = self._end_of(nxt)
            start, head = carry if carry else (pos, heading)
            if nxt is not None and self._only_headings(start, end, bounds):
                carry = (start, head)
                continue
            carry = None
            segments.append((start, end, head))
        return segments

    def _end_of(self, nxt: Pos | None) -> tuple[int, int | None]:
        """次の区切りの手前まで。次の見出しの前に同じページの本文があれば、そのページも含める(重複)"""
        if nxt is None:
            return (self.last, None)
        if nxt.order is not None and self._content_before(nxt.page, nxt.order):
            return (nxt.page, nxt.order)
        return (nxt.page - 1, None)

    def _line(self, page_no: int, order: int) -> Line | None:
        return next((l for l in self.pages[page_no].lines if l.order == order), None)

    def _content_before(self, page_no: int, order: int) -> bool:
        page = self.pages[page_no]
        boundary = self._line(page_no, order)
        if boundary is None:
            return bool(page.lines_before(order))
        return any(is_before(l, boundary, page) for l in page.content_lines)

    def _only_headings(self, start: Pos, end: tuple[int, int | None], bounds: list[Heading]) -> bool:
        """範囲内の行がすべて見出し(例: 「第3章」の直後に「第1節」)なら True"""
        head_keys = {(h.global_page, h.line_id) for h in bounds}
        for no, line in self._lines_in(start, end[0], end[1]):
            if (no, line.id) not in head_keys:
                return False
        return True

    def _lines_in(self, start: Pos, end_page: int, end_before: int | None):
        for no in range(start.page, end_page + 1):
            if no not in self.pages:
                continue
            page = self.pages[no]
            first = self._line(no, start.order) if (no == start.page and start.order is not None) else None
            last = self._line(no, end_before) if (no == end_page and end_before is not None) else None
            for line in page.content_lines:
                if first is not None and is_before(line, first, page):
                    continue
                if last is not None and not is_before(line, last, page):
                    continue
                yield no, line

    # ---------- 2. 図版ページを分ける ----------
    def _plate_runs(self) -> set[int]:
        kinds = self.structure.page_kinds
        plate, run = set(), []
        for no in sorted(self.pages):
            if kinds.get(no) == "plate":
                run.append(no)
                continue
            if len(run) >= self.cfg.min_plate_run:
                plate.update(run)
            run = []
        if len(run) >= self.cfg.min_plate_run:
            plate.update(run)
        return plate

    def _split_plates(self, start: Pos, end: tuple[int, int | None]):
        """範囲を、図版ページの連続部分とそれ以外に分ける"""
        end_page, end_before = end
        parts, cur, cur_plate = [], None, None
        for no in range(start.page, end_page + 1):
            is_plate = no in self.plate_pages
            if cur is None or is_plate != cur_plate:
                if cur is not None:
                    parts.append([cur, no - 1, cur_plate])
                cur, cur_plate = no, is_plate
        if cur is not None:
            parts.append([cur, end_page, cur_plate])
        result = []
        for s, e, plate in parts:
            s_pos = start if s == start.page else Pos(s)
            e_before = end_before if e == end_page else None
            result.append((s_pos, e, e_before, plate))
        return result

    def _plate_chunks(self, s: int, e: int) -> list[Chunk]:
        size = max(1, self.cfg.max_pages_per_chunk)
        chunks = []
        for i, p in enumerate(range(s, e + 1, size)):
            heading = next((h for h in self.all_headings if h.global_page == p and h.level <= self.cfg.section_level), None)
            title = heading.text if heading else "図版"
            # 図版は本文の節に属さないことが多いので、上位の見出しは付けない(見出しがある場合はそれに従う)
            crumb = [] if heading is None else self._breadcrumb(Pos(p), heading.level)
            chunks.append(Chunk(0, "plate", title, heading.level if heading else None, crumb,
                                i > 0, Pos(p), min(p + size - 1, e), None))
        return chunks

    # ---------- 3. 長い範囲を分割する ----------
    def _split_long(self, start: Pos, end_page: int, end_before: int | None, heading: Heading | None) -> list[Chunk]:
        size = max(1, self.cfg.max_pages_per_chunk)
        out = []
        cur, cur_head, continued = start, (heading if heading and start == Pos(heading.global_page, heading.order) else None), False
        if cur_head is None and heading is not None:
            continued = True  # 図版で分断された後半など
        subs = [h for h in self.all_headings
                if h.level > self.cfg.section_level and self._after(h, start) and self._before_end(h, end_page, end_before)]
        while end_page - cur.page + 1 > size:
            limit = cur.page + size - 1
            cands = [h for h in subs if self._after(h, cur) and h.global_page <= limit]
            if cands:
                cut = cands[-1]
                e_page, e_before = self._end_of(Pos(cut.global_page, cut.order))
                out.append(self._section_chunk(cur, e_page, e_before, cur_head, continued, heading))
                cur, cur_head, continued = Pos(cut.global_page, cut.order), cut, False
            else:
                out.append(self._section_chunk(cur, limit, None, cur_head, continued, heading))
                cur, cur_head, continued = Pos(limit + 1), None, True
        out.append(self._section_chunk(cur, end_page, end_before, cur_head, continued, heading))
        return out

    @staticmethod
    def _after(h: Heading, pos: Pos) -> bool:
        return (h.global_page, h.order) > (pos.page, pos.order if pos.order is not None else -1)

    @staticmethod
    def _before_end(h: Heading, end_page: int, end_before: int | None) -> bool:
        if h.global_page < end_page:
            return True
        return h.global_page == end_page and (end_before is None or h.order < end_before)

    def _section_chunk(self, start, end_page, end_before, head, continued, section_head) -> Chunk:
        if head is not None:
            kind = "special" if head.style == "special" else "section"
            return Chunk(0, kind, head.text, head.level, self._breadcrumb(start, head.level),
                         False, start, end_page, end_before)
        chain = self._ancestors(start)
        if section_head is None or not chain:
            return Chunk(0, "front", "表紙など(最初の見出しより前)", None, [], False, start, end_page, end_before)
        # 途中から始まる範囲: 直前の最も下位の見出しの「続き」とする
        return Chunk(0, "section", f"{chain[-1].text}(続き)", chain[-1].level,
                     [h.text for h in chain[:-1]], True, start, end_page, end_before)

    def _ancestors(self, pos: Pos) -> list[Heading]:
        """その位置より前にある、上位から順の見出しの並び(例: [第3章 調査の成果, 第2節 遺物, 2.土器])"""
        latest: dict[int, Heading] = {}
        for h in self.all_headings:
            if (h.global_page, h.order) >= (pos.page, pos.order if pos.order is not None else -1):
                break
            latest[h.level] = h
            for lv in [l for l in latest if l > h.level]:
                del latest[lv]
        return [latest[l] for l in sorted(latest)]

    def _breadcrumb(self, pos: Pos, own_level: int) -> list[str]:
        return [h.text for h in self._ancestors(pos) if h.level < own_level]

    # ---------- チャンクに含まれる見出し・図 ----------
    def _in_scope(self, c: Chunk, page_no: int, y: int | None = None, order: int | None = None) -> bool:
        if not (c.start.page <= page_no <= c.end_page):
            return False
        page = self.pages[page_no]
        if order is None and y is not None:  # 図は上端のy座標で、境界の行より後ろかを判断する
            if page_no == c.start.page and c.start.order is not None:
                sl = next((l for l in page.lines if l.order == c.start.order), None)
                if sl and y < sl.y0:
                    return False
            if page_no == c.end_page and c.end_before is not None:
                el = next((l for l in page.lines if l.order == c.end_before), None)
                if el and y >= el.y0:
                    return False
            return True
        if page_no == c.start.page and c.start.order is not None and order < c.start.order:
            return False
        if page_no == c.end_page and c.end_before is not None and order >= c.end_before:
            return False
        return True

    def _headings_in(self, c: Chunk) -> list[Heading]:
        keys = {(no, l.id) for no, l in self._lines_in(c.start, c.end_page, c.end_before)}
        return [h for h in self.all_headings if (h.global_page, h.line_id) in keys]

    def _figures_in(self, c: Chunk) -> list[Figure]:
        return [f for f in self.figures.figures if self._in_scope(c, f.global_page, y=f.block.y0)]

    def scope_lines(self, c: Chunk) -> dict[int, list[int]]:
        """ページごとの、範囲内の行IDの一覧"""
        result: dict[int, list[int]] = {no: [] for no in c.page_range}
        for no, line in self._lines_in(c.start, c.end_page, c.end_before):
            result[no].append(line.id)
        return result