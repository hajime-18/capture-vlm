"""実験の記録(学習の実行ごと・評価ごと)

experiments/
  runs/{実行名}/
    run.json            重みの初期値・学習データ(件数とハッシュ)・ハイパーパラメータ・環境・結果
    train_ids.json      学習に使った表のID(データセットごと)
    train_log.jsonl     学習の経過(ステップごとの損失・学習率)
    adapter/            学習したLoRAの重み
  evals/{モデル}/{データセット}_{評価設定のハッシュ}/
    eval.json           評価したモデル・データ・設定と、集計した指標
    predictions.jsonl   表ごとの出力と点数(途中で止まっても続きから評価できる)
  results.csv           すべての評価の一覧(1行 = 1モデル × 1データセット)
  results.md            学習前(base)と学習後の比較表(report 段階で作る)
"""

import csv
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

from ..base import PipelineError
from ..config import BASE_DIR, Config
from .data import sha256_file

BASE = "base"  # 学習前のモデル(LoRAなし)を表す名前

RESULT_COLUMNS = [
    "evaluated_at", "model", "base_model", "init", "train_data", "eval_data", "eval_split", "n",
    "teds_struct", "teds_struct_simple", "teds_struct_complex", "exact_match", "shape_match", "parse_ok",
    "truncated", "sec_per_table", "eval_set_sha256", "settings_sha256", "eval_dir",
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def git_info() -> dict:
    """コードの版(結果を再現するため)。未コミットの変更があれば dirty=True"""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=BASE_DIR, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def environment() -> dict:
    import torch
    info = {"python": platform.python_version(), "host": platform.node(), "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    for mod in ("transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            info[mod] = None
    return info


class Experiments:
    def __init__(self, config: Config):
        self.config = config
        self.root = config.experiment_dir

    # --- 学習の実行 ---
    def run_dir(self, name: str) -> Path:
        return self.root / "runs" / name

    def adapter_dir(self, name: str) -> Path:
        return self.run_dir(name) / "adapter"

    def load_run(self, name: str) -> dict:
        path = self.run_dir(name) / "run.json"
        if not path.exists():
            runs = ", ".join(self.finished_runs()) or "(なし)"
            raise PipelineError(f"学習の実行が見つかりません: {name}\n  学習済みの実行: {runs}")
        run = read_json(path)
        if run.get("status") != "done" or not (self.adapter_dir(name) / "adapter_model.safetensors").exists():
            raise PipelineError(f"学習が完了していない実行です: {name}(状態: {run.get('status')})")
        return run

    def finished_runs(self) -> list[str]:
        out = []
        for p in sorted((self.root / "runs").glob("*/run.json")):
            if read_json(p).get("status") == "done":
                out.append(p.parent.name)
        return out

    def new_run_name(self, init: str, datasets: list[str]) -> str:
        return f"{datetime.now():%Y%m%d_%H%M%S}_{'+'.join(datasets)}_from-{init}"

    def init_record(self, init: str) -> dict:
        """重みの初期値の記録。init は base(学習前のモデル)か、学習済みの実行名"""
        if init == BASE:
            return {"type": BASE, "base_model": self.config.model_id, "quantization": self.config.quantization}
        parent = self.load_run(init)
        adapter = self.adapter_dir(init) / "adapter_model.safetensors"
        return {"type": "run", "run": init, "base_model": parent["init"]["base_model"],
                "base_revision": parent["init"].get("base_revision"),
                "quantization": parent["init"]["quantization"],
                "adapter_sha256": sha256_file(adapter),
                "parent_init": parent["init"], "parent_train_data": parent["train_data"]}

    def describe_model(self, model: str) -> dict:
        """results.csv の model・base_model・init・train_data 列の値"""
        if model == BASE:
            return {"model": BASE, "base_model": self.config.model_id, "init": "", "train_data": ""}
        run = self.load_run(model)
        init = run["init"]
        return {"model": model, "base_model": init["base_model"],
                "init": init.get("run", BASE),
                "train_data": "+".join(f"{d['name']}({d['n']})" for d in run["train_data"])}

    # --- 評価 ---
    def eval_dir(self, model: str, dataset: str, settings_sha: str) -> Path:
        key = f"base__{self.config.model_id.replace('/', '--')}" if model == BASE else f"run__{model}"
        return self.root / "evals" / key / f"{dataset}_{settings_sha[:10]}"

    def append_result(self, row: dict) -> None:
        path = self.root / "results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    def results(self) -> list[dict]:
        path = self.root / "results.csv"
        if not path.exists():
            return []
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
