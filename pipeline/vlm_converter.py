"""[3] ページ画像 + OCR JSON → Markdown (page モード)。VLMクライアントは section モードとも共有する"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .base import Step
from .config import Config
from .prompt import build_prompt
from .target import TargetDocuments


@dataclass
class GenerationResult:
    text: str
    num_tokens: int      # 生成したトークン数
    truncated: bool      # 上限に達して途中で打ち切られたか


class QwenVlmClient:
    """Qwen3.8 の読み込みと推論だけを担当する"""

    def __init__(self, config: Config):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForMultimodalLM as AutoVLM
        except ImportError:  # 古いtransformers向け
            from transformers import AutoModelForImageTextToText as AutoVLM

        self._torch = torch
        self.max_new_tokens = config.max_new_tokens
        self.repetition_penalty = config.repetition_penalty
        self.max_image_pixels = config.max_image_pixels
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAが使えません。torchのインストールを確認してください")

        print(f"  モデルを読み込み中: {config.model_id} (量子化: {config.quantization or 'なし'})")
        self.processor = AutoProcessor.from_pretrained(config.model_id)
        self.model = AutoVLM.from_pretrained(
            config.model_id,
            dtype=self._compute_dtype(),
            device_map="auto",
            max_memory=self._max_memory(config),
            quantization_config=self._quantization_config(config.quantization),
            attn_implementation=config.attn_implementation,
        ).eval()
        self.context_length = self._context_length()
        print(f"  コンテキスト長: {self.context_length or '不明'} / 最大出力: {self.max_new_tokens} トークン")

    def _context_length(self) -> int | None:
        """モデルが扱える最大トークン数(入力+出力)を設定ファイルから読む"""
        cfg = self.model.config
        for c in (getattr(cfg, "text_config", None), cfg):
            n = getattr(c, "max_position_embeddings", None) if c is not None else None
            if n:
                return int(n)
        return None

    def _compute_dtype(self):
        """bf16はAmpere(計算能力8.0)以降のみ高速。それより古いGPU(Pascal, Volta等)ではfp16を使う"""
        torch = self._torch
        majors = [torch.cuda.get_device_capability(i)[0] for i in range(torch.cuda.device_count())]
        return torch.bfloat16 if min(majors) >= 8 else torch.float16

    def _max_memory(self, config: Config) -> dict:
        """空き容量を調べ、十分に空いているGPUだけにモデルを配置する"""
        GiB = 1024 ** 3
        mem = {}
        for i in range(self._torch.cuda.device_count()):
            free, total = self._torch.cuda.mem_get_info(i)
            name = self._torch.cuda.get_device_name(i)
            free_gib, total_gib = free / GiB, total / GiB
            if free_gib < config.min_gpu_free_gib:
                print(f"    GPU {i} ({name}): 空き {free_gib:.1f}/{total_gib:.1f} GiB → 使用しない")
                continue
            limit = min(config.gpu_memory_limit_gib, free_gib - config.gpu_reserve_gib)
            mem[i] = f"{limit:.1f}GiB"
            print(f"    GPU {i} ({name}): 空き {free_gib:.1f}/{total_gib:.1f} GiB → {limit:.1f} GiB まで使用")
        if not mem:
            raise RuntimeError(f"空きが {config.min_gpu_free_gib} GiB 以上あるGPUがありません")
        mem["cpu"] = config.cpu_memory_limit
        return mem

    def _quantization_config(self, mode: str | None):
        if mode is None:
            return None
        from transformers import BitsAndBytesConfig
        skip = ["visual", "lm_head"]  # 画像エンコーダと出力層は精度を保つため量子化しない
        if mode == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self._compute_dtype(),
                llm_int8_skip_modules=skip,
            )
        if mode == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
        raise ValueError(f"未対応の量子化指定です: {mode}")

    def _load_image(self, image_path: Path, max_pixels: int | None = None):
        """画素数が上限を超える画像は縮小する(画像トークンが増えすぎるとOOMになるため)"""
        from PIL import Image
        limit = max_pixels or self.max_image_pixels
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if w * h > limit:
            scale = (limit / (w * h)) ** 0.5
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    def generate(self, images: Path | list[Path], prompt: str, max_pixels: int | None = None,
                 max_new_tokens: int | None = None, quiet: bool = False) -> GenerationResult:
        """画像(0枚以上)と指示文から生成する。画像は指示文より前に、渡した順で並べる"""
        paths = [images] if isinstance(images, Path) else list(images)
        loaded = [self._load_image(p, max_pixels) for p in paths]
        content = [{"type": "image", "image": img} for img in loaded] + [{"type": "text", "text": prompt}]
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
            enable_thinking=False,  # 思考過程を出力させない
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        max_new = self._allowed_new_tokens(input_len, max_new_tokens or self.max_new_tokens)
        if not quiet:
            sizes = ", ".join(f"{i.size[0]}x{i.size[1]}" for i in loaded) or "なし"
            print(f"    画像 {len(loaded)}枚 ({sizes}px) / 入力 {input_len} トークン")
        try:
            with self._torch.inference_mode():
                ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,  # 再現性のため貪欲法
                    repetition_penalty=self.repetition_penalty,
                )
            new_tokens = ids[0][input_len:]
            n = len(new_tokens)
            ended = n > 0 and new_tokens[-1].item() in self._eos_ids()
            return GenerationResult(
                text=self.processor.decode(new_tokens, skip_special_tokens=True),
                num_tokens=n,
                truncated=(n >= max_new and not ended),
            )
        finally:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()

    def generate_text(self, prompt: str, max_new_tokens: int) -> str:
        """画像を使わない生成(区切りの判定で使う)"""
        return self.generate([], prompt, max_new_tokens=max_new_tokens, quiet=True).text

    def _allowed_new_tokens(self, input_len: int, wanted: int) -> int:
        """入力と出力の合計がコンテキスト長を超えないようにする"""
        if self.context_length is None:
            return wanted
        room = self.context_length - input_len
        if room < wanted:
            print(f"    [注意] コンテキスト長の残りが {room} トークンのため、出力上限を下げます")
        return max(1, min(wanted, room))

    def _eos_ids(self) -> set[int]:
        eos = self.model.generation_config.eos_token_id
        return set(eos) if isinstance(eos, (list, tuple)) else {eos}


_CLIENT: QwenVlmClient | None = None


def get_client(config: Config) -> QwenVlmClient:
    """モデルの読み込みは時間がかかるので、1回の実行の中では使い回す(区切りの判定とMarkdown化で共有)"""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = QwenVlmClient(config)
    return _CLIENT


class MarkdownCleaner:
    """Markdown以外の混入物(思考過程・コードフェンス)を取り除く"""

    _THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
    _FENCE = re.compile(r"^```(?:markdown|md)?\s*\n(.*?)\n```$", re.DOTALL)

    def clean(self, text: str) -> str:
        text = self._THINK.sub("", text).replace("</think>", "").strip()
        m = self._FENCE.match(text)
        if m:
            text = m.group(1).strip()
        return text + "\n"


class VlmMarkdownConverter(Step):
    name = "VLM → Markdown"

    def __init__(self, config: Config, targets: TargetDocuments):
        super().__init__(config, targets)
        self.cleaner = MarkdownCleaner()

    def run(self) -> None:
        tasks = self._pending_tasks()
        if not tasks:
            self.log("Markdown化待ちのページはありません")
            return
        client = get_client(self.config)  # 必要な時だけ読み込む
        for i, (img, js, out) in enumerate(tasks, 1):
            self.log(f"[{i}/{len(tasks)}] {img.name}")
            try:
                ocr_text = json.dumps(json.loads(js.read_text(encoding="utf-8")), ensure_ascii=False)
                result = client.generate(img, build_prompt(ocr_text))
                md = self.cleaner.clean(result.text)
                self.log(f"  出力 {result.num_tokens} トークン")
                if result.truncated:
                    self.log("  [警告] 出力が上限に達して途中で打ち切られました"
                             "(config.py の max_new_tokens を増やすか、繰り返しが起きていないか確認してください)")
                    md += "\n<!-- warning: output truncated -->\n"  # 後から検索できるよう印を残す
                out.write_text(md, encoding="utf-8")
            except Exception as e:
                self.log(f"  [エラー] {e}")

    def _pending_tasks(self) -> list[tuple[Path, Path, Path]]:
        cfg = self.config
        cfg.page_md_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for img in sorted(cfg.image_dir.glob("*.png")):
            if not self.targets.includes(img.stem):
                continue
            js = cfg.json_dir / f"{img.stem}.json"
            out = cfg.page_md_dir / f"{img.stem}.md"
            if js.exists() and (cfg.overwrite or not out.exists()):
                tasks.append((img, js, out))
        return tasks