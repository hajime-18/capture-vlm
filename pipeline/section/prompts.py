"""section モードでVLMに渡す指示文を組み立てる"""

import json

from ..config import Config

COMMON_RULES = """## 変換ルール
1. 本文の文字はOCR結果を基本とし、誤認識・脱字・改行の乱れは画像を見て正しく修正すること。
2. 「表」の一覧にある表は、中身を書かずに位置だけを示すこと(表の中身は別に処理する)。一覧に無い表(遺構一覧表、遺物観察表など)はMarkdownの表形式で再現し、行と列の対応を崩さないこと。
3. 遺跡名・遺構番号(SK01, SD02 等)・遺物番号・数値・単位・時代名・土層名は画像通り正確に記載すること。
4. ページ番号・柱(ページ上部の章題などの繰り返し表記)は出力しないこと。
5. 前のページから次のページへ続く文や段落は、ページの切れ目で区切らずに1つにつなげること。
6. 画像にもOCR結果にも無い情報は追加しないこと。判読できない文字は「〓」とすること。

## 出力形式
- 出力はMarkdown本文のみとすること。
- 前置き、説明、感想、コードブロックの囲み(```)は一切出力しないこと。
"""

TYPE_LABEL = {"タイトル本文": "見出し", "キャプション": "キャプション", "割注": "注", "頭注": "注"}


class SectionPromptBuilder:
    def __init__(self, config: Config):
        self.config = config

    def build(self, chunk: dict) -> str:
        parts = ["あなたは考古学の発掘調査報告書をデジタル化する専門家です。",
                 "添付の画像(報告書の連続したページ)と、その後に続くOCR結果をもとに、指定した範囲をMarkdown形式に変換してください。",
                 "",
                 self._scope_section(chunk),
                 self._heading_section(chunk),
                 self._figure_section(chunk),
                 self._table_section(chunk),
                 COMMON_RULES,
                 "## OCR結果(範囲内の行のみ。[数字]は行番号、[図 F0001] は図の位置、[表 T0001] は表の位置)",
                 self._ocr_section(chunk)]
        return "\n".join(p for p in parts if p is not None)

    # ---------- 範囲 ----------
    def _scope_section(self, chunk: dict) -> str:
        pages = chunk["pages"]
        n = len(pages)
        first, last = pages[0]["global_page"], pages[-1]["global_page"]
        span = f"通し{first}ページ" if n == 1 else f"通し{first}〜{last}ページ"
        rows = ["## この範囲について",
                f"- 画像は{n}ページ分({span})です。" + ("画像の順番どおりに続いています。" if n > 1 else "")]
        if chunk["breadcrumb"]:
            rows.append(f"- この範囲は「{' > '.join(chunk['breadcrumb'])}」の中にあります。"
                        "これらの上位の見出しは前の範囲で出力済みなので、出力しないでください。")
        if chunk["continued"]:
            rows.append(f"- この範囲は「{chunk['title'].removesuffix('(続き)')}」の続きです。見出しを繰り返さず、本文から書き始めてください。")
        start, end = chunk["scope"]["start"], chunk["scope"]["end"]
        if start["line"] and start.get("shared_with_previous"):
            rows.append(f"- 1枚目の画像は「{start['line']['text'].strip()}」から変換してください。"
                        "それより前の部分は前の範囲で変換済みです。")
        if end["before_line"] and end.get("shared_with_next"):
            rows.append(f"- {n}枚目の画像は「{end['before_line']['text'].strip()}」の手前までを変換してください。"
                        "それ以降は次の範囲で変換します。")
        if chunk["kind"] == "plate":
            rows.append("- この範囲は写真図版・図面のページです。図版ごとに、下の図の一覧のIDを使って"
                        " ![キャプション](figures/F0001.png) の形で、ページ順・図版番号順に出力してください。")
        return "\n".join(rows) + "\n"

    # ---------- 見出しの階層 ----------
    @staticmethod
    def _heading_section(chunk: dict) -> str | None:
        heads = [h for h in chunk["headings"] if h["level"] < 9]
        if not heads:
            return None
        rows = ["## 見出しの階層",
                "この範囲で判定した見出しです。次のMarkdownの見出し記号を使ってください。"]
        for h in heads:
            rows.append(f"- {'#' * min(h['level'], 6)} {h['text']}")
        rows.append("一覧に無い見出しがあれば、この階層に合わせて適切なレベルにしてください。")
        return "\n".join(rows) + "\n"

    # ---------- 図 ----------
    @staticmethod
    def _figure_section(chunk: dict) -> str | None:
        figs = chunk["figures"]
        if not figs:
            return None
        index = {p["global_page"]: i + 1 for i, p in enumerate(chunk["pages"])}
        rows = ["## 図・写真",
                "この範囲には次の図があります。元の図があった位置に、図のIDを使って ![キャプション](figures/F0001.png) と書いてください。",
                "図の内容は説明しないでください。IDは一覧にあるものだけを使い、新しく作らないでください。"]
        for f in figs:
            cap = f"キャプション「{f['caption']}」" if f["caption"] else "キャプションなし(画像から読み取ってください)"
            rows.append(f"- {f['id']}: {index.get(f['page'], '?')}枚目の画像、{cap}")
        return "\n".join(rows) + "\n"

    # ---------- 表 ----------
    @staticmethod
    def _table_section(chunk: dict) -> str | None:
        tables = chunk.get("tables", [])
        if not tables:
            return None
        index = {p["global_page"]: i + 1 for i, p in enumerate(chunk["pages"])}
        rows = ["## 表",
                "この範囲には次の表があります。表の中身は別に処理するため、Markdownの表にはしないでください。",
                "元の表があった位置に、表のIDを使って [キャプション](tables/T0001.html) と1行だけ書いてください。",
                "表のキャプションはこの1行に含め、別の行には書かないでください。IDは一覧にあるものだけを使い、新しく作らないでください。"]
        for t in tables:
            cap = f"キャプション「{t['caption']}」" if t["caption"] else "キャプションなし(画像から読み取り、無ければ「表」としてください)"
            rows.append(f"- {t['id']}: {index.get(t['page'], '?')}枚目の画像、{cap}")
        return "\n".join(rows) + "\n"

    # ---------- OCR結果 ----------
    @staticmethod
    def _ocr_section(chunk: dict) -> str:
        blocks = []
        marks_by_page: dict[int, list[tuple[int, str]]] = {}  # ページ → (上端のy, 印)
        in_table: set[tuple[int, int]] = set()                 # 表の中の行(表の印にまとめるので行は出さない)
        for f in chunk["figures"]:
            marks_by_page.setdefault(f["page"], []).append((f["boundingBox"][0][1], f"[図 {f['id']}]"))
        for t in chunk.get("tables", []):
            marks_by_page.setdefault(t["page"], []).append((t["boundingBox"][0][1], f"[表 {t['id']}]"))
            in_table.update((t["page"], line_id) for line_id in t["line_ids"])
        for i, page in enumerate(chunk["pages"], 1):
            no = page["global_page"]
            in_scope = set(page["in_scope_line_ids"])
            lines = page["ocr"]["contents"][0] if page["ocr"].get("contents") else []
            lines = sorted((l for l in lines if l["id"] in in_scope and (no, l["id"]) not in in_table),
                           key=lambda l: l.get("order", l["id"]))
            marks = sorted(marks_by_page.get(no, []))
            rows = [f"### 画像{i}(通し{no}ページ)"]
            for l in lines:
                y = l["boundingBox"][0][1]
                while marks and marks[0][0] <= y:
                    rows.append(marks.pop(0)[1])
                label = TYPE_LABEL.get(l.get("type", ""), "")
                rows.append(f"[{l['id']}] {label + ': ' if label else ''}{l['text']}")
            rows += [m for _, m in marks]
            if len(rows) == 1:
                rows.append("(文字なし)")
            blocks.append("\n".join(rows))
        return "\n\n".join(blocks)

TABLE_PROMPT = """あなたは考古学の発掘調査報告書をデジタル化する専門家です。
添付の画像は報告書から切り出した表1つです。この表を、行と列の構造を保ったままHTMLの<table>に変換してください。

## 変換ルール
1. 1行を<tr>、見出しのセルを<th>、データのセルを<td>とすること。
2. 複数の行・列にまたがる結合セルは rowspan・colspan で表し、セルを重複して書かないこと。
3. 空のセルも<td></td>として書き、各行の列数を揃えること。
4. セル内の改行は<br>とすること。
5. 文字は下のOCR結果を基本とし、誤認識は画像を見て正しく修正すること。遺構番号・遺物番号・数値・単位・記号(*, -, ( ) など)は画像通り正確に書くこと。
6. 画像に無い情報は追加しないこと。判読できない文字は「〓」とすること。
7. キャプション(表題)と表の外の注記は出力しないこと。

## 出力形式
- <table>から</table>までだけを出力すること。前置き、説明、コードブロックの囲み(```)は一切出力しないこと。

## この表について
- キャプション: {caption}

## OCR結果(表の中の文字を、高さの揃ったものごとに1行にまとめたもの。x=は表の左端からの位置で、列の対応の手がかりにしてください。結合セルの文字は行の間に出ることがあります)
{ocr}
"""


def build_table_prompt(table: dict, page_ocr: dict) -> str:
    """表1つ分の指示文。表の中のOCRの行を、見た目の行ごとに左から並べる"""
    x0, ids = table["boundingBox"][0][0], set(table["line_ids"])
    lines = page_ocr["contents"][0] if page_ocr.get("contents") else []
    boxes = []  # (上端, 下端, 左端, 文字)
    for l in lines:
        if l["id"] in ids:
            xs = [p[0] for p in l["boundingBox"]]
            ys = [p[1] for p in l["boundingBox"]]
            boxes.append((min(ys), max(ys), min(xs), l["text"]))
    rows: list[list[tuple]] = []
    for b in sorted(boxes):
        center = (b[0] + b[1]) / 2
        if rows and rows[-1][0][0] <= center <= rows[-1][0][1]:  # 行の最初の文字と高さが重なれば同じ行
            rows[-1].append(b)
        else:
            rows.append([b])
    text = "\n".join("  ".join(f"[x={b[2] - x0}] {b[3]}" for b in sorted(r, key=lambda b: b[2])) for r in rows)
    return TABLE_PROMPT.format(caption=table["caption"] or "なし", ocr=text or "(文字なし)")
