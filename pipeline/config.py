"""パイプライン全体の設定"""

import re
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # capture_vlm/


@dataclass
class Config:
    # --- 処理方式 ---
    mode: str = "section"                        # "section"(節ごと) / "page"(1ページずつ)。--mode で上書き可

    # --- ディレクトリ ---
    pdf_dir: Path = BASE_DIR / "pdf"
    image_dir: Path = BASE_DIR / "images"
    json_dir: Path = BASE_DIR / "ocr_json"
    output_dir: Path = BASE_DIR / "output_md"
    chunk_dir: Path = BASE_DIR / "chunks"        # 節ごとの区切り(範囲JSON)の保存先

    # --- PDF → 画像(VLM入力専用) ---
    render_dpi: int = 200

    # --- NDLOCR-Lite ---
    ndlocr_dir: Path = BASE_DIR / "ndlocr-lite"  # git submodule として配置したNDLOCR-Lite
    ocr_pdf_dpi: int = 200                       # NDLOCR-LiteがPDFを読み取るときの解像度
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

    # --- 節ごとの処理(section モード) ---
    section_level: int = 2                       # 区切る見出しの階層(1=章, 2=節)
    max_pages_per_chunk: int = 3                 # 1回にVLMへ入れる最大ページ数(重複ページを含む)
    min_plate_run: int = 2                       # 図版ページがこの枚数以上続けば図版だけでまとめる
    section_max_image_pixels: int = 1_200_000    # 複数ページを入れるため、1枚あたりの画素数を抑える
    boundary_llm: str = "ambiguous"              # 区切り判定でのLLMの使い方: "off" / "ambiguous"(迷う箇所だけ) / "all"
    heading_accept_score: float = 4.0            # 見出しと認める点数
    heading_candidate_score: float = 2.0         # 見出し候補として扱う点数(この間はLLMで確認)
    # 1冊の報告書が複数のPDFに分かれている場合のファイル名の規則(report=報告書名, order=順番)
    report_patterns: tuple = (
        r"^(?P<report>.+)\((?P<order>\d+)\)$",          # eriseki_005(1), eriseki_005(2), ...
        r"^(?P<report>.+?)_(?P<order>\d{3})_\d{3}$",      # 014_001_004, 014_005_008, ...(開始_終了ページ)
    )

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