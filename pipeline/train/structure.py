"""表の構造HTML(文字を含まないHTML)の正規化と、TEDS-Struct による採点

構造HTMLの決まり(study/make_dataset の表構造アノテーションツールと同じ):
  - 使うタグは table thead tbody tr td だけで、属性は rowspan と colspan だけ。セルは常に空。
  - 見出し行は thead、それ以外は tbody。見出しが0行なら thead は出力しない。
  - 改行や空白は入れず、1行で出力する。

正解(自作データ・PubTabNet)とVLMの出力の両方を to_structure_html() で同じ形にそろえてから比べる。
"""

from dataclasses import dataclass

from apted import APTED, Config as AptedConfig

from ..section.table_html import _TableParser


@dataclass
class _Node:
    tag: str
    colspan: int = 1
    rowspan: int = 1
    children: list["_Node"] | None = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


def parse_structure(html: str) -> list[tuple[bool, list[tuple[int, int]]]] | None:
    """最初の<table>を [(見出し行か, [(rowspan, colspan), ...]), ...] にする。表が無ければ None

    <th> は見出しのセルとして扱うが、行が見出しかどうかは <thead> の中かどうかで決める
    (アノテーションと同じ。<thead> を使わずに <th> だけの行は、行のセルがすべて <th> なら見出しとみなす)。
    """
    p = _TableParser()
    p.feed(html)
    p.close()
    rows = []
    for r in p.rows:
        if not r.cells:
            continue
        head = r.in_head or all(c.header for c in r.cells)
        rows.append((head, [(c.rowspan, c.colspan) for c in r.cells]))
    if not rows:
        return None
    # 見出しは表の上から続く行だけ(本文の後の見出し行は本文として扱う)
    n_head = 0
    while n_head < len(rows) and rows[n_head][0]:
        n_head += 1
    return [(i < n_head, cells) for i, (_, cells) in enumerate(rows)]


def to_structure_html(html: str) -> str | None:
    """HTMLの最初の<table>を、文字を含まない構造HTML(1行)にする。表が無ければ None"""
    rows = parse_structure(html)
    if rows is None:
        return None

    def row(cells):
        tds = "".join(
            "<td" + (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "") + "></td>"
            for rs, cs in cells)
        return f"<tr>{tds}</tr>"

    head = [row(c) for h, c in rows if h]
    body = [row(c) for h, c in rows if not h]
    out = "<table>"
    if head:
        out += "<thead>" + "".join(head) + "</thead>"
    if body:
        out += "<tbody>" + "".join(body) + "</tbody>"
    return out + "</table>"


def table_shape(html: str) -> tuple[int, int] | None:
    """(行数, 列数)。列数は結合セルを展開して数える"""
    rows = parse_structure(html)
    if rows is None:
        return None
    taken: set[tuple[int, int]] = set()
    for r, (_, cells) in enumerate(rows):
        col = 0
        for rs, cs in cells:
            while (r, col) in taken:
                col += 1
            for dr in range(rs):
                for dc in range(cs):
                    taken.add((r + dr, col + dc))
            col += cs
    return len(rows), (max(c for _, c in taken) + 1 if taken else 0)


def has_span(html: str) -> bool:
    """結合セルを含むか(PubTabNet の simple / complex の区分)"""
    return 'rowspan="' in html or 'colspan="' in html


def _tree(structure_html: str) -> _Node:
    root = _Node("table")
    rows = parse_structure(structure_html) or []
    groups = {}
    for head, cells in rows:
        tag = "thead" if head else "tbody"
        if tag not in groups:
            groups[tag] = _Node(tag)
            root.children.append(groups[tag])
        tr = _Node("tr", children=[_Node("td", rowspan=rs, colspan=cs) for rs, cs in cells])
        groups[tag].children.append(tr)
    return root


def _count(node: _Node) -> int:
    return 1 + sum(_count(c) for c in node.children)


class _StructConfig(AptedConfig):
    """TEDS-Struct: タグ・rowspan・colspan が違えば置換のコスト1(文字は比べない)"""

    def rename(self, a: _Node, b: _Node) -> int:
        return int(a.tag != b.tag or a.colspan != b.colspan or a.rowspan != b.rowspan)

    def children(self, node: _Node) -> list[_Node]:
        return node.children


def teds_struct(pred_html: str | None, true_html: str) -> float:
    """TEDS-Struct(0〜1、1が完全一致)。pred_html が None(表として読めない)なら 0

    Zhong et al. "Image-based table recognition: data, model, and evaluation" (PubTabNet) の
    TEDS を、セルの文字を無視して計算したもの: 1 - 木の編集距離 / max(2つの木の節点数)
    """
    if pred_html is None:
        return 0.0
    a, b = _tree(pred_html), _tree(true_html)
    n = max(_count(a), _count(b))
    dist = APTED(a, b, _StructConfig()).compute_edit_distance()
    return max(0.0, 1.0 - dist / n)
