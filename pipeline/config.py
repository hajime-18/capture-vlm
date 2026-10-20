"""パイプライン全体の設定"""

import re
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # capture_vlm/


@dataclass
class Config:
    # --- ディレクトリ ---
    pdf_dir: Path = BASE_DIR / "pdf"
    image_dir: Path = BASE_DIR / "images"
    json_dir: Path = BASE_DIR / "ocr_json"
    output_dir: Path = BASE_DIR / "output_md"

    # --- PDF → 画像 ---
    render_dpi: int = 200

    # --- NDLOCR-Lite ---
    ndlocr_dir: Path = BASE_DIR / "ndlocr-lite"  # git submodule として配置したNDLOCR-Lite
    ocr_device: str = "cuda"                     # "cuda" または "cpu"

    # --- VLM ---
    model_id: str = "Qwen/Qwen3.5-9B"             # 32GB級のGPUが空いていれば "Qwen/Qwen3.8-27B"
    max_new_tokens: int = 8192                   # 1ページあたりの最大出力トークン数
    repetition_penalty: float = 1.0              # 同じ行の繰り返しが起きる場合は 1.05 程度にする
    max_image_pixels: int = 1_600_000            # VLMに渡す画像の最大画素数(A4で約145dpi相当)
    attn_implementation: str = "sdpa"            # メモリ効率の良いattention
    quantization: str | None = "4bit"            # "4bit" / "8bit" / None(量子化なし)
    gpu_memory_limit_gib: float = 14.0           # 1枚あたりに載せる上限
    gpu_reserve_gib: float = 2.0                 # 推論時の作業領域として各GPUに残す量
    min_gpu_free_gib: float = 6.0                # 空きがこれ未満のGPUは使わない(小容量GPU・使用中GPUを除外)
    cpu_memory_limit: str = "64GiB"              # GPUに載らない分を置くCPUメモリの上限

    # --- 共通 ---
    overwrite: bool = False

    # ページ画像のファイル名: {PDF名}_p0001
    page_name_re: re.Pattern = field(default=re.compile(r"^(?P<doc>.+)_p(?P<page>\d{4})$"))

    @property
    def ndlocr_script(self) -> Path:
        return self.ndlocr_dir / "src" / "ocr.py"

    @property
    def page_md_dir(self) -> Path:
        return self.output_dir / "pages"

    @staticmethod
    def page_stem(doc_name: str, page_no: int) -> str:
        return f"{doc_name}_p{page_no:04d}"