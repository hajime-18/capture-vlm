"""
発掘調査報告書 PDF → Markdown 一括変換

  python main.py                  # 全段階を実行
  python main.py --only pdf ocr   # 指定した段階だけ実行 (pdf / ocr / plan / vlm / merge)
  python main.py --mode page      # 1ページずつ処理する(省略時は config.py の mode)
  python main.py --overwrite      # 出力済みでも作り直す
  python main.py --ocr-device cpu # OCRをCPUで実行する
  python main.py --pdf "eriseki_005(1).pdf" 014_001_004 # PDFを指定して実行(括弧を含む名前は引用符で囲む)
  python main.py --pdf "eriseki_005*"                   # ワイルドカードで指定
  python main.py --clean                                # 生成物をすべて削除する(pdf/ は残す)
  python main.py --clean --pdf "eriseki_005*"           # 指定したPDF(報告書)の生成物だけ削除する

  # 表の構造認識の学習と評価(train モード: data → train → eval → report)
  python main.py --mode train --train-data pubtabnet --run ptn          # PubTabNetで学習し、学習前(base)と比べる
  python main.py --mode train --train-data own --init ptn --run ptn_own # 学習済みの ptn から続けて自作データで学習
  python main.py --mode train --only eval report --models base ptn ptn_own  # 評価と比較表だけ
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
from pipeline.cleaner import OutputCleaner
from pipeline.pipeline import ALL_STEP_NAMES, MODES
from pipeline.target import TargetDocuments
from pipeline.train.data import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description="発掘調査報告書 PDF → Markdown パイプライン")
    parser.add_argument("--only", nargs="+", choices=ALL_STEP_NAMES, help="実行する段階")
    parser.add_argument("--pdf", nargs="+", metavar="NAME",
                        help="処理するPDF(ファイル名・拡張子なし・ワイルドカード・パス)。省略時は pdf/ 内のすべて")
    parser.add_argument("--mode", choices=list(MODES),
                        help="section: 節ごとにまとめて処理 / page: 1ページずつ処理(省略時は config.py の mode)")
    parser.add_argument("--overwrite", action="store_true", help="出力済みのファイルも作り直す")
    parser.add_argument("--ocr-device", choices=["cuda", "cpu"], default="cuda", help="OCRに使うデバイス")
    train = parser.add_argument_group("train モード(表の構造認識の学習と評価)")
    train.add_argument("--run", metavar="NAME", help="学習の実行名(省略時は日時などから自動で付ける)")
    train.add_argument("--init", metavar="BASE|RUN", help="重みの初期値: base(学習前のモデル)か学習済みの実行名")
    train.add_argument("--train-data", nargs="+", choices=list(DATASETS), help="学習に使うデータ")
    train.add_argument("--eval-data", nargs="+", choices=list(DATASETS), help="評価に使うデータ")
    train.add_argument("--models", nargs="+", metavar="BASE|RUN",
                       help="評価するモデル(省略時は base と今回の実行)")
    parser.add_argument("--clean", action="store_true",
                        help="images/ ocr_json/ chunks/ output_md/ の生成物を確認のうえ削除して終了する(pdf/ は削除しない)")
    args = parser.parse_args()
    if args.clean and (args.only or args.overwrite):
        parser.error("--clean は --only・--overwrite と同時に指定できません")
    train_opts = [args.run, args.init, args.train_data, args.eval_data, args.models]
    if args.mode == "train" and (args.pdf or args.clean):
        parser.error("train モードでは --pdf・--clean は使えません")
    if args.mode != "train" and any(train_opts):
        parser.error("--run・--init・--train-data・--eval-data・--models は --mode train のときだけ使えます")
    return args


def main():
    args = parse_args()
    config = Config(overwrite=args.overwrite, ocr_device=args.ocr_device)
    if args.mode:
        config.mode = args.mode
    for key, value in (("train_run", args.run), ("train_init", args.init), ("train_datasets", args.train_data),
                       ("eval_datasets", args.eval_data), ("eval_models", args.models)):
        if value:
            config.__setattr__(key, tuple(value) if isinstance(value, list) else value)
    try:
        if args.clean:
            targets = TargetDocuments(config, args.pdf)
            targets.expand_to_reports()
            OutputCleaner(config, targets).run()
            return
        Pipeline(config, steps=args.only, pdf_names=args.pdf).run()
    except (FileNotFoundError, PipelineError, ValueError) as e:  # 利用者が対処できるものはメッセージだけ表示
        sys.exit(str(e))


if __name__ == "__main__":
    main()