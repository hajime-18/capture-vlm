"""train モードの通しの動作確認(自作データだけで 学習 → 学習前後の評価 → 比較表)

記録は本番の experiments/ を汚さないよう test/results/train_smoke/ に保存する。
結果の要約は test/results/train_smoke_{日時}.txt に出力する。
  CUDA_VISIBLE_DEVICES=1 python test/train_smoke.py [--samples N] [--pubtabnet]
"""

import argparse
import contextlib
import io
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pipeline import Config, Pipeline  # noqa: E402
from pipeline.train import data as train_data  # noqa: E402


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=0, help="学習に使う件数を絞る(0=すべて)")
    ap.add_argument("--pubtabnet", action="store_true", help="PubTabNet の評価用データも先頭20件だけ評価する")
    args = ap.parse_args()

    out_dir = ROOT / "test" / "results"
    log_path = out_dir / f"train_smoke_{datetime.now():%Y%m%d_%H%M%S}.txt"
    datasets = ("own", "pubtabnet") if args.pubtabnet else ("own",)
    config = Config(mode="train", experiment_dir=out_dir / "train_smoke", train_datasets=("own",),
                    eval_datasets=datasets, train_run=f"smoke_{datetime.now():%Y%m%d_%H%M%S}",
                    grad_accum_steps=4, train_log_steps=1)

    # 件数を絞る(長い表を優先して残し、メモリが足りるかを確かめる)
    original_load = train_data.TableDatasets.load

    def load(self, name, split):
        s = original_load(self, name, split)
        if split == "train" and args.samples:
            s = sorted(s, key=lambda x: -len(x.target))[:args.samples]
        if split == "test" and name == "pubtabnet":
            s = s[:20]
        return s
    train_data.TableDatasets.load = load

    with open(log_path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(Tee(sys.stdout, f)):
        print(f"train_smoke: {datetime.now().isoformat(timespec='seconds')} samples={args.samples or 'all'}")
        Pipeline(config, steps=["data", "train", "eval", "report"]).run()
        import torch
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} 最大使用量: {torch.cuda.max_memory_allocated(i) / 1024 ** 3:.2f} GiB")
    print(f"結果: {log_path}")


if __name__ == "__main__":
    main()
