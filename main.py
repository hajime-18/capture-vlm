"""
発掘調査報告書 PDF → Markdown 一括変換

  python main.py                  # 全段階を実行
  python main.py --only pdf ocr   # 指定した段階だけ実行 (pdf / ocr / plan / vlm / merge)
  python main.py --mode page      # 1ページずつ処理する(省略時は config.py の mode)
  python main.py --overwrite      # 出力済みでも作り直す
  python main.py --ocr-device cpu # OCRをCPUで実行する
  python main.py --pdf "eriseki_005(1).pdf" 014_001_004 # PDFを指定して実行(括弧を含む名前は引用符で囲む)
  python main.py --pdf "eriseki_005*"                   # ワイルドカードで指定
"""

import argparse
import os
import sys
from pathlib import Path

# venvの外で実行されるのを防ぐ(別環境のtransformers等が使われてしまうため)
if sys.prefix == sys.base_prefix:
    venv_python = Path(__file__).resolve().parent / "venv" / "bin" / "python"
    sys.exit(f"venvが有効になっていません。\n  source venv/bin/activate を実行するか、\n  {venv_python} main.py で実行してください。")

# メモリの断片化によるOOMを減らす(torchの読み込み前に設定する必要がある)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pipeline import Config, Pipeline
from pipeline.base import PipelineError
from pipeline.pipeline import ALL_STEP_NAMES, MODES


def parse_args():
    parser = argparse.ArgumentParser(description="発掘調査報告書 PDF → Markdown パイプライン")
    parser.add_argument("--only", nargs="+", choices=ALL_STEP_NAMES, help="実行する段階")
    parser.add_argument("--pdf", nargs="+", metavar="NAME",
                        help="処理するPDF(ファイル名・拡張子なし・ワイルドカード・パス)。省略時は pdf/ 内のすべて")
    parser.add_argument("--mode", choices=list(MODES),
                        help="section: 節ごとにまとめて処理 / page: 1ページずつ処理(省略時は config.py の mode)")
    parser.add_argument("--overwrite", action="store_true", help="出力済みのファイルも作り直す")
    parser.add_argument("--ocr-device", choices=["cuda", "cpu"], default="cuda", help="OCRに使うデバイス")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(overwrite=args.overwrite, ocr_device=args.ocr_device)
    if args.mode:
        config.mode = args.mode
    try:
        Pipeline(config, steps=args.only, pdf_names=args.pdf).run()
    except (FileNotFoundError, PipelineError, ValueError) as e:  # 利用者が対処できるものはメッセージだけ表示
        sys.exit(str(e))


if __name__ == "__main__":
    main()