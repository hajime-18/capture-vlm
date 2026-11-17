"""VLMが出力したHTMLの<table>を整えて、1つの表を表すHTMLファイルにする(section モードの表)

出力: output_md/{報告書名}/tables/T0001.html

<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>表3 出土遺物観察表</title>
</head>
<body>
<table id="T0001" data-report="…" data-rows="6" data-cols="8" data-status="ok"
       data-page="47" data-pdf="013_001_055" data-page-no="47" data-printed-page="39" data-image="T0001.png"
       data-x0="188" data-y0="626" data-x1="1239" data-y1="809" data-confidence="0.95">
<caption>表3 出土遺物観察表</caption>
<tr>
  <th data-row="1" data-col="1">番号</th>
  <th data-row="1" data-col="3" colspan="3">法量</th>
  …
</tr>
</table>
</body>
</html>

data-row・data-col は結合セルを展開したときの位置(1始まり)。VLMの出力が表として読めなかった場合は
data-status="parse_error" とし、出力をそのまま表の後の <pre class="raw"> に残す。
"""

import re
from dataclasses import dataclass, field
from html import escape
from html.parser import HTMLParser


@dataclass
class Cell:
    text: str = ""
    header: bool = False
    rowspan: int = 1
    colspan: int = 1


@dataclass
class _Row:
    cells: list[Cell] = field(default_factory=list)
    in_head: bool = False


class _TableParser(HTMLParser):
    """最初の<table>の行とセルを読む(閉じタグの抜けなど、崩れたHTMLも読めるだけ読む)"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[_Row] = []
        self.depth = 0          # <table>の入れ子の深さ
        self.done = False
        self.in_head = False
        self.cell: Cell | None = None

    def handle_starttag(self, tag, attrs):
        if self.done:
            return
        if tag == "table":
            self.depth += 1
            return
        if self.depth != 1:
            return
        if tag == "thead":
            self.in_head = True
        elif tag in ("tbody", "tfoot"):
            self.in_head = False
        elif tag == "tr":
            self._close_cell()
            self.rows.append(_Row(in_head=self.in_head))
        elif tag in ("td", "th"):
            self._close_cell()
            if not self.rows:
                self.rows.append(_Row(in_head=self.in_head))
            a = dict(attrs)
            self.cell = Cell(header=(tag == "th" or self.in_head),
                             rowspan=_span(a.get("rowspan")), colspan=_span(a.get("colspan")))
            self.rows[-1].cells.append(self.cell)
        elif tag == "br" and self.cell is not None:
            self.cell.text += "\n"

    def handle_endtag(self, tag):
        if self.done:
            return
        if tag == "table":
            self.depth -= 1
            if self.depth == 0:
                self._close_cell()
                self.done = True
        elif self.depth == 1 and tag in ("td", "th", "tr"):
            self._close_cell()
        elif self.depth == 1 and tag == "thead":
            self.in_head = False

    def handle_data(self, data):
        if self.cell is not None and not self.done:
            self.cell.text += data

    def _close_cell(self):
        if self.cell is not None:
            self.cell.text = "\n".join(re.sub(r"\s+", " ", t).strip() for t in self.cell.text.split("\n")).strip()
            self.cell = None


def _span(value) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def parse_html_table(html: str) -> list[list[Cell]] | None:
    """HTMLの最初の<table>を、行ごとのセルの一覧にする。表が無ければ None"""
    p = _TableParser()
    p.feed(html)
    p.close()
    p._close_cell()
    rows = [r.cells for r in p.rows if r.cells]
    return rows or None


def _layout(rows: list[list[Cell]]) -> tuple[list[list[tuple[int, Cell]]], int]:
    """結合セルを考慮して、各セルの列位置(1始まり)と全体の列数を求める"""
    taken: set[tuple[int, int]] = set()
    placed, n_cols = [], 0
    for r, cells in enumerate(rows):
        col, out = 0, []
        for cell in cells:
            while (r, col) in taken:
                col += 1
            out.append((col + 1, cell))
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    taken.add((r + dr, col + dc))
            col += cell.colspan
        n_cols = max(n_cols, col)
        placed.append(out)
    n_cols = max([n_cols] + [c + 1 for _, c in taken])
    return placed, n_cols


def _attrs(attrs: dict) -> str:
    return "".join(f' {k}="{escape(str(v))}"' for k, v in attrs.items() if v is not None)


def build_table_html(table: dict, report: str, page: dict, vlm_output: str) -> tuple[str, bool]:
    """表のHTML(文字列)と、表として読めたかを返す。table は chunk JSON の tables の1項目、page はそのページの項目"""
    xs = [p[0] for p in table["boundingBox"]]
    ys = [p[1] for p in table["boundingBox"]]
    attrs = {"id": table["id"], "data-report": report}
    src = {"data-page": table["page"], "data-pdf": page["pdf"], "data-page-no": page["page_no"],
           "data-printed-page": page["printed_page"], "data-image": table["image"].rsplit("/", 1)[-1],
           "data-x0": min(xs), "data-y0": min(ys), "data-x1": max(xs), "data-y1": max(ys),
           "data-confidence": table["confidence"]}

    rows = parse_html_table(vlm_output)
    body, after = [], []
    if rows is None:
        attrs["data-status"] = "parse_error"
        after.append(f'<pre class="raw">{escape(vlm_output.strip())}</pre>')
    else:
        placed, n_cols = _layout(rows)
        attrs |= {"data-rows": len(rows), "data-cols": n_cols, "data-status": "ok"}
        for r, cells in enumerate(placed, 1):
            body.append("<tr>")
            for col, cell in cells:
                tag = "th" if cell.header else "td"
                a = {"data-row": r, "data-col": col,
                     "rowspan": cell.rowspan if cell.rowspan > 1 else None,
                     "colspan": cell.colspan if cell.colspan > 1 else None}
                text = "<br>".join(escape(t) for t in cell.text.split("\n"))
                body.append(f"  <{tag}{_attrs(a)}>{text}</{tag}>")
            body.append("</tr>")
    caption = escape(table["caption"])
    html = "\n".join([
        "<!DOCTYPE html>", '<html lang="ja">', "<head>", '<meta charset="utf-8">',
        f"<title>{caption or escape(table['id'])}</title>", "</head>", "<body>",
        f"<table{_attrs(attrs | src)}>", f"<caption>{caption}</caption>", *body, "</table>", *after,
        "</body>", "</html>", ""])
    return html, rows is not None
