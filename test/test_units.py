"""表のXML化(table_xml.py)と、表に関するプロンプト(prompts.py)の単体テスト

GPU・PDFは不要。結果は画面と test/results/units_{日時}.txt に出力する。
  python test/test_units.py
"""

import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.section.prompts import SectionPromptBuilder, build_table_prompt  # noqa: E402
from pipeline.section.table_xml import build_table_xml, parse_html_table  # noqa: E402


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


TABLE = {"id": "T0001", "page": 3, "caption": "表1 遺物観察表", "image": "tables/T0001.png",
         "boundingBox": box(100, 200, 500, 400), "confidence": 0.9,
         "xml": "tables/T0001.xml", "line_ids": [1, 2, 3, 4]}
PAGE = {"global_page": 3, "pdf": "x", "page_no": 3, "printed_page": None,
        "in_scope_line_ids": [0, 1, 2, 3, 4, 5],
        "ocr": {"contents": [[
            {"id": 0, "order": 0, "text": "本文前", "boundingBox": box(100, 50, 400, 80)},
            {"id": 1, "order": 1, "text": "番号", "boundingBox": box(110, 210, 150, 230)},
            {"id": 2, "order": 2, "text": "器種", "boundingBox": box(200, 212, 250, 228)},
            {"id": 3, "order": 3, "text": "1", "boundingBox": box(110, 260, 130, 280)},
            {"id": 4, "order": 4, "text": "土師器", "boundingBox": box(200, 260, 260, 280)},
            {"id": 5, "order": 5, "text": "本文後", "boundingBox": box(100, 450, 400, 480)},
        ]]}}


def parse(xml: str) -> ET.Element:
    return ET.fromstring(xml.split("\n", 1)[1])  # XML宣言を除く


def cell_texts(rows):
    return [[c.text for c in r] for r in rows]


def run() -> list[tuple[str, bool, str]]:
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    # 普通の表
    xml, ok = build_table_xml(TABLE, "r", PAGE, "<table><tr><th>番号</th><th>器種</th></tr>"
                                               "<tr><td>1</td><td>土師器</td></tr></table>")
    root = parse(xml)
    check("普通の表: 行数・列数", ok and root.get("rows") == "2" and root.get("cols") == "2")
    check("値がNoneの属性は出力しない", "printed_page" not in root.find("source").attrib)
    check("<th> は header=true", root.find("row/cell").get("header") == "true")

    # 結合セル
    html = ("<table><tr><th rowspan=2>番号</th><th colspan=2>法量</th></tr><tr><th>口径</th><th>器高</th></tr>"
            "<tr><td>1</td><td>12.0</td><td>3.1</td></tr></table>")
    root = parse(build_table_xml(TABLE, "r", PAGE, html)[0])
    pos = [(c.get("row"), c.get("col")) for c in root.iter("cell")]
    check("結合セルの列位置", pos == [("1", "1"), ("1", "2"), ("2", "2"), ("2", "3"),
                                ("3", "1"), ("3", "2"), ("3", "3")] and root.get("cols") == "3", str(pos))

    # 前置き・コードブロック・閉じタグの抜け・<br>・<thead>
    rows = parse_html_table("以下です\n```html\n<table><thead><tr><td>a<br>b<td>c</thead>"
                            "<tbody><tr><td>1<td>2</table>\n```")
    check("崩れたHTML", cell_texts(rows) == [["a\nb", "c"], ["1", "2"]] and rows[0][0].header, str(cell_texts(rows)))

    # 表が無い出力
    xml, ok = build_table_xml(TABLE, "r", PAGE, "表を読めませんでした")
    check("表が無い出力は parse_error と <raw>", not ok and 'status="parse_error"' in xml and "<raw>" in xml)

    # 入れ子の表(内側の文字は外側のセルにつながる: 現状の仕様)
    rows = parse_html_table("<table><tr><td>x<table><tr><td>in</td></tr></table></td></tr></table>"
                            "<table><tr><td>2nd</td></tr></table>")
    check("入れ子・2つ目の表は無視", len(rows) == 1 and len(rows[0]) == 1, str(cell_texts(rows)))

    # 特殊文字
    xml, ok = build_table_xml(TABLE | {"caption": "表<1>&"}, "r", PAGE, "<table><tr><td>a&amp;b &lt;c&gt;</td></tr></table>")
    check("特殊文字のエスケープ", ok and parse(xml).find("row/cell").text == "a&b <c>")

    check("空の<table>はNone", parse_html_table("<table></table>") is None)

    # 表のプロンプト: 表の中の行だけを、見た目の行ごとに左から並べる
    p = build_table_prompt(TABLE, PAGE["ocr"])
    check("表のプロンプトのOCR行", "[x=10] 番号  [x=100] 器種" in p and "[x=10] 1  [x=100] 土師器" in p
          and "本文前" not in p)

    # 本文のプロンプト: 表の中の行は [表 T0001] にまとめる
    chunk = {"chunk_id": "0001", "kind": "section", "title": "t", "level": 2, "breadcrumb": [], "continued": False,
             "scope": {"start": {"page": 3, "line": None}, "end": {"page": 3, "line": None}},
             "headings": [], "figures": [], "tables": [TABLE], "pages": [PAGE]}
    sec = SectionPromptBuilder._ocr_section(chunk)
    check("本文のOCRで表の行を印にまとめる", sec.splitlines()[1:] == ["[0] 本文前", "[表 T0001]", "[5] 本文後"], sec)
    check("表の一覧", "T0001: 1枚目の画像" in (SectionPromptBuilder._table_section(chunk) or ""))
    return results


def main():
    results = run()
    lines = [f"単体テスト {datetime.now():%Y-%m-%d %H:%M:%S}"]
    for name, ok, detail in results:
        lines.append(f"{'OK' if ok else 'NG'}  {name}" + (f"\n    {detail}" if detail and not ok else ""))
    n_ng = sum(not ok for _, ok, _ in results)
    lines.append(f"合計 {len(results)}件 / NG {n_ng}件")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    out = Path(__file__).parent / "results" / f"units_{datetime.now():%Y%m%d_%H%M%S}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"結果: {out}")
    sys.exit(1 if n_ng else 0)


if __name__ == "__main__":
    main()
