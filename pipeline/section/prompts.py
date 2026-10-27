"""section モードでVLMに渡す指示文を組み立てる"""

import json

from ..config import Config

COMMON_RULES = """## 変換ルール
1. 本文の文字はOCR結果を基本とし、誤認識・脱字・改行の乱れは画像を見て正しく修正すること。
2. 表(遺構一覧表、遺物観察表など)はMarkdownの表形式で再現し、行と列の対応を崩さないこと。
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
                 COMMON_RULES,
                 "## OCR結果(範囲内の行のみ。[数字]は行番号、[図 F0001] は図の位置)",
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

    # ---------- OCR結果 ----------
    @staticmethod
    def _ocr_section(chunk: dict) -> str:
        blocks = []
        figs_by_page: dict[int, list[dict]] = {}
        for f in chunk["figures"]:
            figs_by_page.setdefault(f["page"], []).append(f)
        for i, page in enumerate(chunk["pages"], 1):
            in_scope = set(page["in_scope_line_ids"])
            lines = page["ocr"]["contents"][0] if page["ocr"].get("contents") else []
            lines = sorted((l for l in lines if l["id"] in in_scope), key=lambda l: l.get("order", l["id"]))
            figs = sorted(figs_by_page.get(page["global_page"], []), key=lambda f: f["boundingBox"][0][1])
            rows = [f"### 画像{i}(通し{page['global_page']}ページ)"]
            for l in lines:
                y = l["boundingBox"][0][1]
                while figs and figs[0]["boundingBox"][0][1] <= y:
                    rows.append(f"[図 {figs.pop(0)['id']}]")
                label = TYPE_LABEL.get(l.get("type", ""), "")
                rows.append(f"[{l['id']}] {label + ': ' if label else ''}{l['text']}")
            rows += [f"[図 {f['id']}]" for f in figs]
            if len(rows) == 1:
                rows.append("(文字なし)")
            blocks.append("\n".join(rows))
        return "\n\n".join(blocks)