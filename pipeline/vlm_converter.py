"""[3] ページ画像 + OCR JSON → Markdown (Qwen3.8)"""

import json
import re
from pathlib import Path

from .base import Step
from .config import Config
from .prompt import build_prompt
from .target import TargetDocuments


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

    def _load_image(self, image_path: Path):
        """画素数が上限を超える画像は縮小する(画像トークンが増えすぎるとOOMになるため)"""
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if w * h > self.max_image_pixels:
            scale = (self.max_image_pixels / (w * h)) ** 0.5
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    def generate(self, image_path: Path, prompt: str) -> str:
        image = self._load_image(image_path)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
            enable_thinking=False,  # 思考過程を出力させない
        ).to(self.model.device)
        print(f"    画像 {image.size[0]}x{image.size[1]}px / 入力 {inputs['input_ids'].shape[-1]} トークン")
        try:
            with self._torch.inference_mode():
                ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                          do_sample=False)  # 再現性のため貪欲法
            new_tokens = ids[0][inputs["input_ids"].shape[-1]:]
            return self.processor.decode(new_tokens, skip_special_tokens=True)
        finally:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()


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
        client = QwenVlmClient(self.config)  # 必要な時だけ読み込む
        for i, (img, js, out) in enumerate(tasks, 1):
            self.log(f"[{i}/{len(tasks)}] {img.name}")
            try:
                ocr_text = json.dumps(json.loads(js.read_text(encoding="utf-8")), ensure_ascii=False)
                raw = client.generate(img, build_prompt(ocr_text))
                out.write_text(self.cleaner.clean(raw), encoding="utf-8")
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