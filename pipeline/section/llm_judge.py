"""VLMによる見出しの判定(書式のルールで判断できない箇所の補助)

ページ画像(小さめに縮小)とOCR結果の両方を渡し、どの行が見出しかを判定させる。
画像を見ることで、太字・中央寄せ・余白などの見た目を判断に使えるほか、
OCRが読み誤った見出し(例:「序」→「17」)の正しい表記も答えさせる。
ページ画像が無い場合や Config.boundary_use_image が False の場合は、OCR結果だけで判定する。

VLMに任せるのは「どの行が見出しで、どの階層か」だけで、どこで区切るかはプログラム側(chunker.py)が決める。
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .ocr_page import OcrPage
from .toc import TocEntry

LEVEL_NAMES = {"章": 1, "節": 2, "項": 3, "その他": 4}

INTRO_IMAGE = """あなたは日本の発掘調査報告書の構成を分析する専門家です。
添付の画像は報告書の1ページで、その下はこのページのOCR結果です。
OCR結果の各行は「[行番号] (種別, 文字の大きさ) 文字列」の形式で、文字の大きさは本文を1.0とした比率です。
画像で文字の大きさ・太さ・配置(中央寄せ、字下げ)・前後の余白を確かめて判断してください。"""

INTRO_TEXT = """あなたは日本の発掘調査報告書の構成を分析する専門家です。
以下は報告書の1ページ分のOCR結果です。各行は「[行番号] (種別, 文字の大きさ) 文字列」の形式です。
文字の大きさは本文を1.0とした比率です。"""

PROMPT = """{intro}

このページの中で、章・節・項などの「見出し」になっている行をすべて選んでください。
- 本文中の箇条書き、図表のキャプション、表の中の文字、柱(ページ上部の章題)は見出しではありません。
- 序文・例言・目次・報告書抄録などの表題も見出しです(level は「章」)。
- 見出しが無ければ空のリストを返してください。
- "text" には、その見出しの正しい表記を書いてください。{text_rule}
{toc_hint}
出力はJSONのみとし、説明やコードブロックの囲みは付けないでください。
{{"headings": [{{"line": 行番号, "level": "章" または "節" または "項" または "その他", "text": "見出しの正しい表記"}}]}}

## OCR結果
{lines}
"""

TEXT_RULE_IMAGE = "OCRの文字が画像と違う場合(読み誤り)は、画像のとおりに直してください。"
TEXT_RULE_TEXT = "OCRの文字をそのまま書いてください。"


@dataclass
class JudgeResult:
    level: int
    text: str | None  # VLMが答えた見出しの正しい表記(無ければ None)


class Generator(Protocol):
    def generate(self, images, prompt: str, max_pixels: int | None = None,
                 max_new_tokens: int | None = None, quiet: bool = False): ...

    def generate_text(self, prompt: str, max_new_tokens: int) -> str: ...


class LlmBoundaryJudge:
    def __init__(self, generator_factory: Callable[[], Generator], use_image: bool = True,
                 image_pixels: int = 800_000, max_lines: int = 80, max_new_tokens: int = 384):
        """generator_factory は初めて判定するときに呼ぶ(判定が不要な報告書ではモデルを読み込まない)"""
        self._factory = generator_factory
        self._generator: Generator | None = None
        self.use_image = use_image
        self.image_pixels = image_pixels
        self.max_lines = max_lines
        self.max_new_tokens = max_new_tokens

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = self._factory()
        return self._generator

    def judge(self, page: OcrPage, nearby_toc: list[TocEntry]) -> dict[int, JudgeResult] | None:
        """見出しと判断した行の {行ID: JudgeResult} を返す。応答を解釈できなければ None"""
        with_image = self.use_image and page.ref.image_path.exists()
        prompt = PROMPT.format(
            intro=INTRO_IMAGE if with_image else INTRO_TEXT,
            text_rule=TEXT_RULE_IMAGE if with_image else TEXT_RULE_TEXT,
            toc_hint=self._toc_hint(nearby_toc), lines=self._format_lines(page),
        )
        if with_image:
            raw = self.generator.generate([Path(page.ref.image_path)], prompt, max_pixels=self.image_pixels,
                                          max_new_tokens=self.max_new_tokens, quiet=True).text
        else:
            raw = self.generator.generate_text(prompt, max_new_tokens=self.max_new_tokens)
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
    def _parse(raw: str, valid_ids: set[int]) -> dict[int, JudgeResult] | None:
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
            if line_id not in valid_ids:
                continue
            text = str(item.get("text") or "").strip()
            result[line_id] = JudgeResult(
                level=LEVEL_NAMES.get(str(item.get("level", "")).strip(), 4),
                text=text if 0 < len(text) <= 60 else None,
            )
        return result