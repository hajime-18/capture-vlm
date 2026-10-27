"""LLMによる見出しの判定(書式のルールで判断できない箇所の補助)

ページ画像は使わず、OCRの文字列・種別・文字の大きさだけを渡すので、VLMに画像を入れるより軽い。
"""

import json
import re
from typing import Callable, Protocol

from .ocr_page import OcrPage
from .toc import TocEntry

LEVEL_NAMES = {"章": 1, "節": 2, "項": 3, "その他": 4}

PROMPT = """あなたは日本の発掘調査報告書の構成を分析する専門家です。
以下は報告書の1ページ分のOCR結果です。各行は「[行番号] (種別, 文字の大きさ) 文字列」の形式です。
文字の大きさは本文を1.0とした比率です。

このページの中で、章・節・項などの「見出し」になっている行をすべて選んでください。
- 本文中の箇条書き、図表のキャプション、表の中の文字、柱(ページ上部の章題)は見出しではありません。
- 序文・例言・目次・報告書抄録などの表題も見出しです(level は「章」)。
- 見出しが無ければ空のリストを返してください。
{toc_hint}
出力はJSONのみとし、説明やコードブロックの囲みは付けないでください。
{{"headings": [{{"line": 行番号, "level": "章" または "節" または "項" または "その他"}}]}}

## OCR結果
{lines}
"""


class TextGenerator(Protocol):
    def generate_text(self, prompt: str, max_new_tokens: int) -> str: ...


class LlmBoundaryJudge:
    def __init__(self, generator_factory: Callable[[], TextGenerator], max_lines: int = 80):
        """generator_factory は初めて判定するときに呼ぶ(LLMが不要な報告書ではモデルを読み込まない)"""
        self._factory = generator_factory
        self._generator: TextGenerator | None = None
        self.max_lines = max_lines

    @property
    def generator(self) -> TextGenerator:
        if self._generator is None:
            self._generator = self._factory()
        return self._generator

    def judge(self, page: OcrPage, nearby_toc: list[TocEntry]) -> dict[int, int] | None:
        """見出しと判断した行の {行ID: 階層} を返す。応答を解釈できなければ None"""
        prompt = PROMPT.format(toc_hint=self._toc_hint(nearby_toc), lines=self._format_lines(page))
        raw = self.generator.generate_text(prompt, max_new_tokens=512)
        return self._parse(raw, {l.id for l in page.content_lines})

    @staticmethod
    def _toc_hint(entries: list[TocEntry]) -> str:
        if not entries:
            return ""
        items = "\n".join(f"  - {e.title}" for e in entries[:12])
        return f"- 参考: 目次によると、このページ付近には次の見出しがあるはずです。\n{items}\n"

    def _format_lines(self, page: OcrPage) -> str:
        body_h = page.body_height
        rows = []
        for l in page.lines[: self.max_lines]:
            text = l.text.strip()
            text = text if len(text) <= 60 else text[:60] + "…"
            rows.append(f"[{l.id}] ({l.type}, {l.height / body_h:.1f}) {text}")
        return "\n".join(rows)

    @staticmethod
    def _parse(raw: str, valid_ids: set[int]) -> dict[int, int] | None:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        result = {}
        for item in data.get("headings", []):
            try:
                line_id = int(item.get("line"))
            except (TypeError, ValueError):
                continue
            if line_id in valid_ids:
                result[line_id] = LEVEL_NAMES.get(str(item.get("level", "")).strip(), 4)
        return result