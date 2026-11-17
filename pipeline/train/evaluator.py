"""[data] 学習・評価データの準備 / [eval] 評価 / [report] 学習前後の比較表

評価は、学習前のモデル(base)と学習済みの実行を、同じ評価用データ・同じ設定で行う。
評価用データの中身と評価設定のハッシュを記録し、比較表ではそれが一致するものだけを並べる。
"""

import json
import time
from collections import defaultdict

from ..base import PipelineError, Step
from .data import DATASETS, PubTabNetPreparer, TableDatasets, samples_digest, sha256_text
from .experiment import BASE, Experiments, environment, git_info, now, read_json, write_json
from .trainer import model_revision
from .structure import has_span, table_shape, teds_struct, to_structure_html


class TrainDataPreparer(Step):
    name = "学習データの準備"

    def run(self) -> None:
        cfg = self.config
        used = set(cfg.train_datasets) | set(cfg.eval_datasets)
        unknown = used - set(DATASETS)
        if unknown:
            raise PipelineError(f"データセットは {list(DATASETS)} のいずれかです: {sorted(unknown)}")
        if "pubtabnet" in used:
            PubTabNetPreparer(cfg, self.log).run()
        loader = TableDatasets(cfg)
        for d in sorted(used):
            train, test = loader.load(d, "train"), loader.load(d, "test")
            self.log(f"{d}: 学習用 {len(train)}件 / 評価用 {len(test)}件")


class TableEvaluator(Step):
    name = "評価"

    def run(self) -> None:
        cfg = self.config
        exp = Experiments(cfg)
        models = list(cfg.eval_models) or [BASE] + ([cfg.train_run] if cfg.train_run else [])
        models = list(dict.fromkeys(models))
        bases = {exp.describe_model(m)["base_model"] for m in models}
        if len(bases) > 1:
            raise PipelineError(f"ベースモデルが違う実行は、1回の実行で評価できません: {sorted(bases)}")
        cfg.model_id = bases.pop()

        loader = TableDatasets(cfg)
        sets = {d: loader.load(d, "test") for d in cfg.eval_datasets}
        settings = {"base_model": cfg.model_id, "quantization": cfg.quantization, "prompt": cfg.table_structure_prompt,
                    "max_image_pixels": cfg.train_max_image_pixels, "max_new_tokens": cfg.eval_max_new_tokens,
                    "repetition_penalty": cfg.repetition_penalty, "decoding": "greedy"}
        settings_sha = sha256_text(json.dumps(settings, sort_keys=True, ensure_ascii=False))

        client = None
        for model in models:
            for dataset, samples in sets.items():
                if not samples:
                    self.log(f"{dataset}: 評価用データがありません")
                    continue
                eval_sha = samples_digest(samples)
                out = exp.eval_dir(model, dataset, sha256_text(settings_sha + eval_sha))
                info = out / "eval.json"
                if info.exists() and read_json(info).get("status") == "done" and not cfg.overwrite:
                    self.log(f"{model} × {dataset}: 評価済み(TEDS-S {read_json(info)['metrics']['teds_struct']:.4f})")
                    continue
                if client is None:
                    from ..vlm_converter import get_client
                    client = get_client(cfg)
                client.use_adapter(None if model == BASE else model,
                                   None if model == BASE else exp.adapter_dir(model))
                self._evaluate(exp, client, model, dataset, samples, settings, settings_sha, eval_sha, out)

    def _evaluate(self, exp, client, model, dataset, samples, settings, settings_sha, eval_sha, out) -> None:
        cfg = self.config
        pred_path = out / "predictions.jsonl"
        if cfg.overwrite:
            pred_path.unlink(missing_ok=True)
        done = {}
        if pred_path.exists():
            for line in pred_path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                done[r["id"]] = r
        desc = exp.describe_model(model)
        meta = {"status": "running", "started_at": now(), **desc, "eval_data": dataset, "eval_split": "test",
                "n": len(samples), "eval_set_sha256": eval_sha, "settings": settings, "settings_sha256": settings_sha,
                "base_revision": model_revision(cfg.model_id), "environment": environment(), "git": git_info()}
        write_json(out / "eval.json", meta)
        self.log(f"{model} × {dataset}: {len(samples)}件" + (f"(評価済み {len(done)}件から続ける)" if done else ""))

        for i, s in enumerate(samples, 1):
            if s.id in done:
                continue
            t0 = time.time()
            res = client.generate(s.image, cfg.table_structure_prompt, max_pixels=cfg.train_max_image_pixels,
                                  max_new_tokens=cfg.eval_max_new_tokens, quiet=True)
            pred = to_structure_html(res.text)
            score = teds_struct(pred, s.target)
            r = {"id": s.id, "teds_struct": score, "exact": pred == s.target,
                 "shape_ok": pred is not None and table_shape(pred) == table_shape(s.target),
                 "parse_ok": pred is not None, "complex": has_span(s.target),
                 "truncated": res.truncated, "num_tokens": res.num_tokens, "sec": round(time.time() - t0, 2),
                 "prediction": pred, "target": s.target, "raw": res.text}
            with open(pred_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done[s.id] = r
            self.log(f"  [{i}/{len(samples)}] {s.id}: TEDS-S {score:.3f}" + (" (打ち切り)" if res.truncated else ""))

        rows = [done[s.id] for s in samples]
        metrics = _metrics(rows)
        meta |= {"status": "done", "finished_at": now(), "metrics": metrics}
        write_json(out / "eval.json", meta)
        exp.append_result({**desc, "evaluated_at": meta["finished_at"], "eval_data": dataset, "eval_split": "test",
                           "n": len(rows), **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()},
                           "eval_set_sha256": eval_sha, "settings_sha256": settings_sha,
                           "eval_dir": str(out.relative_to(exp.root))})
        self.log(f"  TEDS-S {metrics['teds_struct']:.4f} / 完全一致 {metrics['exact_match']:.3f} / "
                 f"行数・列数の一致 {metrics['shape_match']:.3f}")


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _metrics(rows: list[dict]) -> dict:
    return {
        "teds_struct": _mean(r["teds_struct"] for r in rows),
        "teds_struct_simple": _mean(r["teds_struct"] for r in rows if not r["complex"]),
        "teds_struct_complex": _mean(r["teds_struct"] for r in rows if r["complex"]),
        "exact_match": _mean(float(r["exact"]) for r in rows),
        "shape_match": _mean(float(r["shape_ok"]) for r in rows),
        "parse_ok": _mean(float(r["parse_ok"]) for r in rows),
        "truncated": sum(1 for r in rows if r["truncated"]),
        "sec_per_table": _mean(r["sec"] for r in rows),
    }


class ResultReporter(Step):
    name = "比較表"

    def run(self) -> None:
        exp = Experiments(self.config)
        rows = exp.results()
        if not rows:
            self.log("評価結果がまだありません")
            return
        groups = defaultdict(dict)  # (評価データ, データのハッシュ, 設定のハッシュ) → モデル → 最新の行
        for r in rows:
            groups[(r["eval_data"], r["eval_set_sha256"], r["settings_sha256"])][r["model"]] = r
        lines = ["# 表の構造認識: 学習前後の比較", "",
                 f"{now()} 作成。TEDS-S は TEDS-Struct(1が完全一致)。Δ は同じ評価データ・同じ設定の base との差。", ""]
        cols = ["model", "init", "train_data", "n", "teds_struct", "Δ", "teds_struct_simple", "teds_struct_complex",
                "exact_match", "shape_match", "parse_ok", "truncated"]
        ordered = sorted(groups.items(), key=lambda g: (g[0][0], min(r["evaluated_at"] for r in g[1].values())))
        for (data, eval_sha, settings_sha), by_model in ordered:
            base = by_model.get(BASE)
            lines += [f"## {data}(評価データ {eval_sha[:8]} / 設定 {settings_sha[:8]})", "",
                      "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            for m, r in sorted(by_model.items(), key=lambda kv: (kv[0] != BASE, kv[1]["evaluated_at"])):
                delta = f"{float(r['teds_struct']) - float(base['teds_struct']):+.4f}" if base and m != BASE else ""
                lines.append("| " + " | ".join(delta if c == "Δ" else (r.get(c) or "") for c in cols) + " |")
            if base is None:
                lines += ["", "(この評価データ・設定での base の評価がありません)"]
            lines.append("")
        text = "\n".join(lines)
        path = exp.root / "results.md"
        path.write_text(text, encoding="utf-8")
        print(text)
        self.log(f"保存しました: {path}")
