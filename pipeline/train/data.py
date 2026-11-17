"""学習・評価用の表データ(表画像 + 構造HTML)

データセットは2つ:
  own        study/make_dataset/dataset(表構造アノテーションツールの出力。ShareGPT形式の jsonl)
             学習用と評価用には、ページ(または報告書)単位のハッシュで分ける。
             データが増えても、既存の表の振り分けは変わらない。
  pubtabnet  PubTabNet(Hugging Face の apoidea/pubtabnet-html。リビジョンを固定して取得)
             train から学習用、val から評価用を、乱数の種を固定して抜き出し、
             train_data/pubtabnet/ に own と同じ形式で保存する(data 段階)。
"""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from ..base import PipelineError
from ..config import Config
from .structure import to_structure_html

DATASETS = ("pubtabnet", "own")
SPLITS = ("train", "test")

PUBTABNET_REPO = "apoidea/pubtabnet-html"
PUBTABNET_REVISION = "0d58324674ff93825c972f7fdfd2887e0ae0a247"  # 取得するデータを固定する
PUBTABNET_SHARD_ROWS = 15_650                                    # train の1ファイルあたりの件数(目安)


@dataclass
class TableSample:
    id: str
    image: Path
    target: str      # 正解の構造HTML(正規化済み)
    dataset: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def samples_digest(samples: list[TableSample]) -> str:
    """データの中身(ID・正解・画像のファイル名)のハッシュ。同じデータで比べているかの確認に使う"""
    key = "\n".join(f"{s.id}\t{s.image.name}\t{s.target}" for s in sorted(samples, key=lambda s: s.id))
    return sha256_text(key)


def read_sharegpt(jsonl: Path, dataset: str) -> list[TableSample]:
    """ShareGPT形式(make_dataset の table_structure.jsonl と同じ)の jsonl を読む"""
    samples = []
    for n, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        answer = next(m["content"] for m in d["messages"] if m["role"] == "assistant")
        target = to_structure_html(answer)
        if target is None:
            raise PipelineError(f"{jsonl}:{n} の正解を表として読めません: {d.get('id')}")
        samples.append(TableSample(d["id"], jsonl.parent / d["images"][0], target, dataset))
    return samples


def write_sharegpt(jsonl: Path, rows: list[dict], prompt: str) -> None:
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "messages": [{"role": "user", "content": "<image>" + prompt},
                             {"role": "assistant", "content": r["target"]}],
                "images": [r["image"]], "id": r["id"]}, ensure_ascii=False) + "\n")


class TableDatasets:
    """データセット名と分割(train / test)から、表の一覧を返す"""

    def __init__(self, config: Config):
        self.config = config

    def load(self, name: str, split: str) -> list[TableSample]:
        if name == "own":
            return self._own(split)
        if name == "pubtabnet":
            jsonl = self.config.pubtabnet_dir / f"{split}.jsonl"
            if not jsonl.exists():
                raise PipelineError(f"PubTabNet が準備されていません: {jsonl}\n"
                                    "  python main.py --mode train --only data を実行してください。")
            return read_sharegpt(jsonl, "pubtabnet")
        raise ValueError(f"データセットは {list(DATASETS)} のいずれかです: {name}")

    # --- own ---
    def _own(self, split: str) -> list[TableSample]:
        jsonl = self.config.own_dataset_dir / "table_structure.jsonl"
        if not jsonl.exists():
            raise PipelineError(f"自作の学習データが見つかりません: {jsonl}\n"
                                "  config.py の own_dataset_dir を確認してください。")
        return [s for s in read_sharegpt(jsonl, "own") if self.own_split_of(s.id) == split]

    def own_split_of(self, table_id: str) -> str:
        """表ID(例: eriseki_025_p0062_t01)の振り分け先。同じページ(報告書)の表は同じ側にする"""
        m = self.config.page_name_re.match(table_id.rsplit("_t", 1)[0])
        if self.config.own_split_group == "report" and m:
            group = m["doc"]
        elif self.config.own_split_group == "page" and m:
            group = m.group(0)
        else:
            group = table_id
        h = int(hashlib.sha256(f"{self.config.train_seed}:{group}".encode()).hexdigest(), 16)
        return "test" if (h % 10_000) / 10_000 < self.config.own_test_ratio else "train"


class PubTabNetPreparer:
    """PubTabNet の parquet を取得し、指定した件数を抜き出して own と同じ形式で保存する"""

    def __init__(self, config: Config, log):
        self.config = config
        self.log = log
        self.root = config.pubtabnet_dir

    def params(self) -> dict:
        c = self.config
        return {"repo": PUBTABNET_REPO, "revision": PUBTABNET_REVISION,
                "train_shards": c.pubtabnet_train_shards, "train_samples": c.pubtabnet_train_samples,
                "test_samples": c.pubtabnet_test_samples, "seed": c.train_seed, "prompt": c.table_structure_prompt}

    def run(self) -> None:
        info = self.root / "source.json"
        if info.exists():
            old = json.loads(info.read_text(encoding="utf-8"))
            if old["params"] == self.params():
                self.log(f"PubTabNet は準備済みです(学習 {old['counts']['train']}件 / 評価 {old['counts']['test']}件)")
                return
            if not self.config.overwrite:
                raise PipelineError(
                    "config.py の PubTabNet の設定が、準備済みのデータと違います。\n"
                    f"  準備済み: {old['params']}\n  現在    : {self.params()}\n"
                    "  作り直す場合は --overwrite を付けてください(評価用データが変わるため、"
                    "以前の評価結果とは比べられなくなります)。")
        files = [f"data/train-{i:05d}-of-00032.parquet" for i in range(self.config.pubtabnet_train_shards)]
        files.append("data/validation-00000-of-00001.parquet")
        paths = [self._download(f) for f in files]
        counts, digests = {}, {}
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        for split, sources, n in (("train", paths[:-1], self.config.pubtabnet_train_samples),
                                  ("test", paths[-1:], self.config.pubtabnet_test_samples)):
            rows = self._sample(sources, n, split)
            write_sharegpt(self.root / f"{split}.jsonl", rows, self.config.table_structure_prompt)
            counts[split] = len(rows)
            digests[split] = samples_digest(read_sharegpt(self.root / f"{split}.jsonl", "pubtabnet"))
            self.log(f"{split}: {len(rows)}件を保存しました")
        info.write_text(json.dumps({"params": self.params(), "files": files, "counts": counts,
                                    "digests": digests}, ensure_ascii=False, indent=1), encoding="utf-8")

    def _download(self, filename: str) -> Path:
        from huggingface_hub import hf_hub_download
        local = self.root / "raw" / Path(filename).name
        if local.exists():
            return local
        self.log(f"ダウンロード中: {PUBTABNET_REPO}/{filename}")
        path = hf_hub_download(PUBTABNET_REPO, filename, repo_type="dataset", revision=PUBTABNET_REVISION,
                               local_dir=self.root / "raw_hf")
        local.parent.mkdir(parents=True, exist_ok=True)
        Path(path).rename(local)
        return local

    def _sample(self, sources: list[Path], n: int, split: str) -> list[dict]:
        """全件から n 件を乱数の種を固定して選び、画像を保存する。表として読めないものは除く"""
        import pyarrow.parquet as pq
        files = [pq.ParquetFile(p) for p in sources]
        index = [(fi, rg, i) for fi, f in enumerate(files)
                 for rg in range(f.num_row_groups) for i in range(f.metadata.row_group(rg).num_rows)]
        rng = random.Random(f"{self.config.train_seed}:{split}")
        chosen = sorted(rng.sample(index, min(n, len(index))))
        rows = []
        for fi, rg in sorted({(fi, rg) for fi, rg, _ in chosen}):
            table = files[fi].read_row_group(rg, columns=["image", "html_table"]).to_pylist()
            for _, _, i in (c for c in chosen if c[:2] == (fi, rg)):
                r = table[i]
                target = to_structure_html(r["html_table"])
                if target is None:
                    continue
                stem = Path(r["image"]["path"]).stem
                image = Path("images") / f"{stem}.png"
                (self.root / image).write_bytes(r["image"]["bytes"])
                rows.append({"id": stem, "image": str(image), "target": target})
        return rows
