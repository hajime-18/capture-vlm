"""[train] 表の構造認識の学習(QLoRA: 4bit量子化したモデルにLoRAを付けて学習する)

入力は推論時とまったく同じ形(画像 + 指示文、思考なし)にし、正解の構造HTMLの部分だけで損失を計算する。
重みの初期値(学習前のモデル / 学習済みの実行の続き)・学習データ・設定は experiments/runs/{実行名}/ に記録する。
"""

import json
import math
import random
import re
import time
from pathlib import Path

from ..base import PipelineError, Step
from .data import TableDatasets, TableSample, samples_digest, sha256_file
from .experiment import BASE, Experiments, environment, git_info, now, read_json, write_json


class TableTrainer(Step):
    name = "学習"

    def run(self) -> None:
        cfg = self.config
        exp = Experiments(cfg)
        datasets = list(cfg.train_datasets)
        if not cfg.train_run:
            cfg.train_run = exp.new_run_name(cfg.train_init, datasets)
        name = cfg.train_run
        run_dir = exp.run_dir(name)
        if (run_dir / "run.json").exists():
            status = read_json(run_dir / "run.json").get("status")
            if status == "done" and not cfg.overwrite:
                self.log(f"学習済みです: {name}(作り直す場合は --overwrite)")
                return
            if status != "done" and not cfg.overwrite:
                raise PipelineError(f"途中で止まった実行があります: {run_dir}(状態: {status})\n"
                                    "  最初からやり直す場合は --overwrite を付けてください。")
        if cfg.train_init == name:
            raise PipelineError("重みの初期値に、これから学習する実行自身は指定できません")

        init = exp.init_record(cfg.train_init)
        if init["base_model"] != cfg.model_id:
            self.log(f"初期値の実行のベースモデル {init['base_model']} を使います(config.py の model_id とは異なります)")
            cfg.model_id = init["base_model"]
        loader = TableDatasets(cfg)
        per_dataset = {d: loader.load(d, "train") for d in datasets}
        for d, s in per_dataset.items():
            if not s:
                raise PipelineError(f"学習データがありません: {d}")

        import torch
        _seed_all(cfg.train_seed)
        model, processor, client = self._load_model(exp, init)
        init["base_revision"] = init.get("base_revision") or model_revision(cfg.model_id)

        samples = [s for d in datasets for s in per_dataset[d]]
        encoded, skipped = self._encode_all(processor, samples)
        if not encoded:
            raise PipelineError("train_max_length 以内の学習データがありません")

        run_dir.mkdir(parents=True, exist_ok=True)
        for old in ("train_log.jsonl",):
            (run_dir / old).unlink(missing_ok=True)
        write_json(run_dir / "train_ids.json", {d: [s.id for s in per_dataset[d]] for d in datasets})
        record = {
            "run": name, "status": "running", "started_at": now(), "finished_at": None,
            "init": init,
            "train_data": [{"name": d, "n": len(per_dataset[d]),
                            "n_used": sum(1 for s, _ in encoded if s.dataset == d),
                            "sha256": samples_digest(per_dataset[d]), **self._source(d)} for d in datasets],
            "skipped": skipped,
            "prompt": cfg.table_structure_prompt,
            "hyperparams": self._hyperparams(),
            "environment": environment(), "git": git_info(),
        }
        write_json(run_dir / "run.json", record)
        self.log(f"実行名: {name}")
        self.log(f"重みの初期値: {cfg.train_init} / 学習データ: " +
                 ", ".join(f"{d['name']} {d['n_used']}件" for d in record["train_data"]) +
                 (f"(長すぎて除外 {len(skipped)}件)" if skipped else ""))

        try:
            result = self._train(model, processor, encoded, run_dir)
        except BaseException as e:
            record |= {"status": "failed", "finished_at": now(), "error": f"{type(e).__name__}: {e}"}
            write_json(run_dir / "run.json", record)
            raise
        model.save_pretrained(str(run_dir / "adapter"))
        result["adapter_sha256"] = sha256_file(run_dir / "adapter" / "adapter_model.safetensors")
        record |= {"status": "done", "finished_at": now(), "result": result}
        write_json(run_dir / "run.json", record)
        self.log(f"保存しました: {run_dir}")

        # 評価では推論用に読み込み直すので、学習用のモデルはここで解放する
        del model, client
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # --- 準備 ---
    def _source(self, dataset: str) -> dict:
        cfg = self.config
        if dataset == "own":
            return {"path": str(cfg.own_dataset_dir), "split_group": cfg.own_split_group,
                    "test_ratio": cfg.own_test_ratio}
        info = cfg.pubtabnet_dir / "source.json"
        return {"path": str(cfg.pubtabnet_dir), "source": read_json(info)["params"] if info.exists() else None}

    def _hyperparams(self) -> dict:
        c = self.config
        keys = ["lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", "train_epochs", "learning_rate",
                "warmup_ratio", "grad_accum_steps", "max_grad_norm", "train_max_image_pixels", "train_max_length",
                "train_seed", "attn_implementation"]
        return {k: list(v) if isinstance(v, tuple) else v for k, v in ((k, getattr(c, k)) for k in keys)}

    def _load_model(self, exp: Experiments, init: dict):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model

        from ..vlm_converter import QwenVlmClient
        cfg = self.config
        client = QwenVlmClient(cfg, gpu_reserve_gib=cfg.train_gpu_reserve_gib)
        base = client.model
        base.config.use_cache = False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        if init["type"] == BASE:
            # 言語モデル側の線形層だけにLoRAを付ける(画像エンコーダは学習しない)
            targets = "|".join(re.escape(m) for m in cfg.lora_target_modules)
            lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                              target_modules=rf".*language_model\.layers\.\d+\..*\.({targets})",
                              bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(base, lora)
        else:
            path = exp.adapter_dir(init["run"])
            self.log(f"学習済みのLoRAから続けて学習します: {path}")
            model = PeftModel.from_pretrained(base, str(path), is_trainable=True)
        for p in model.parameters():  # LoRAの重みは fp32 で持つ(fp16 のままだと更新が小さすぎて消える)
            if p.requires_grad:
                p.data = p.data.float()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log(f"学習するパラメータ: {n_train / 1e6:.1f}M")
        client.model = model
        return model, client.processor, client

    def _encode_all(self, processor, samples: list[TableSample]):
        """推論時と同じ指示文の後ろに正解を付ける。長すぎるものは除外して記録する"""
        cfg = self.config
        tok = processor.tokenizer
        end_id = tok.convert_tokens_to_ids("<|im_end|>")
        encoded, skipped = [], []
        for s in samples:
            prompt = _prompt_inputs(processor, s.image, cfg.table_structure_prompt, cfg.train_max_image_pixels)
            target = tok.encode(s.target, add_special_tokens=False) + [end_id]
            length = prompt["input_ids"].shape[-1] + len(target)
            if length > cfg.train_max_length:
                skipped.append({"dataset": s.dataset, "id": s.id, "length": length})
                continue
            encoded.append((s, target))
        return encoded, skipped

    # --- 学習 ---
    def _train(self, model, processor, encoded, run_dir: Path) -> dict:
        import torch
        from transformers import get_cosine_schedule_with_warmup
        cfg = self.config

        dtype = _compute_dtype(torch)
        use_scaler = dtype == torch.float16
        params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
        updates_per_epoch = math.ceil(len(encoded) / cfg.grad_accum_steps)
        total = updates_per_epoch * cfg.train_epochs
        sched = get_cosine_schedule_with_warmup(optim, int(total * cfg.warmup_ratio), total)
        # fp16 では、線形注意の層(PyTorch版の実装)の勾配が損失のスケーリングで溢れやすい。
        # 倍率は fp16_loss_scale(既定 1)から増やさず、溢れたときだけその更新を飛ばす
        scaler = torch.amp.GradScaler("cuda", init_scale=cfg.fp16_loss_scale, growth_interval=10 ** 9,
                                      enabled=use_scaler)
        device = model.get_input_embeddings().weight.device
        log_path = run_dir / "train_log.jsonl"
        rng = random.Random(cfg.train_seed)

        model.train()
        self.log(f"{len(encoded)}件 × {cfg.train_epochs}エポック / 更新 {total}回(精度: {dtype})")
        start, step, window, skipped = time.time(), 0, [], 0
        last_loss = None
        for epoch in range(cfg.train_epochs):
            order = list(range(len(encoded)))
            rng.shuffle(order)
            for i, idx in enumerate(order):
                sample, target = encoded[idx]
                inputs = _prompt_inputs(processor, sample.image, cfg.table_structure_prompt,
                                        cfg.train_max_image_pixels)
                with torch.autocast("cuda", dtype=dtype):
                    loss = _target_loss(model, inputs, target, device)
                if not torch.isfinite(loss):
                    self.log(f"  [警告] 損失が数値になりませんでした({sample.dataset}/{sample.id})。この表は飛ばします")
                    optim.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss / cfg.grad_accum_steps).backward()
                window.append(loss.item())
                if (i + 1) % cfg.grad_accum_steps and i + 1 != len(order):
                    continue
                scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm).item()
                ok = math.isfinite(grad_norm)
                if not ok:
                    skipped += 1
                    self.log(f"  [警告] 勾配が溢れたため、この更新を飛ばしました(累計 {skipped}回)")
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                last_loss = sum(window) / len(window)
                entry = {"step": step, "epoch": epoch + (i + 1) / len(order), "loss": last_loss,
                         "lr": sched.get_last_lr()[0], "grad_norm": grad_norm if ok else None,
                         "skipped": not ok, "loss_scale": scaler.get_scale() if use_scaler else None,
                         "elapsed_sec": round(time.time() - start, 1)}
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
                window = []
                if step % cfg.train_log_steps == 0 or step == total:
                    eta = (time.time() - start) / step * (total - step)
                    self.log(f"[{step}/{total}] 損失 {last_loss:.4f} / 学習率 {entry['lr']:.2e} / "
                             f"残り約 {eta / 60:.0f}分")
        model.eval()
        return {"updates": step, "skipped_updates": skipped, "final_loss": last_loss, "elapsed_sec": round(time.time() - start, 1),
                "n_samples": len(encoded)}


def model_revision(model_id: str) -> str | None:
    """Hugging Face のキャッシュにあるモデルの版(コミットのハッシュ)"""
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(model_id, "config.json")  # .../snapshots/{コミット}/config.json
        return Path(path).parent.name if isinstance(path, str) else None
    except Exception:
        return None


def _seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _compute_dtype(torch):
    majors = [torch.cuda.get_device_capability(i)[0] for i in range(torch.cuda.device_count())]
    return torch.bfloat16 if min(majors) >= 8 else torch.float16


def load_image(path: Path, max_pixels: int):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _prompt_inputs(processor, image: Path, prompt: str, max_pixels: int):
    """推論時(QwenVlmClient.generate)と同じ形の入力"""
    content = [{"type": "image", "image": load_image(image, max_pixels)}, {"type": "text", "text": prompt}]
    return processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", enable_thinking=False)


def _target_loss(model, inputs, target: list[int], device, chunk: int = 512):
    """正解部分だけの交差エントロピー

    語彙が約25万あるため、全位置のロジットを一度に作るとメモリが足りない。
    言語モデルの最後の隠れ状態を取り、正解の位置だけを chunk ずつ出力層に通す(勾配チェックポイントで再計算)。
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    prompt_ids = inputs["input_ids"]
    tgt = torch.tensor([target], dtype=prompt_ids.dtype)
    batch = {k: v.to(device) for k, v in inputs.items()}
    batch["input_ids"] = torch.cat([prompt_ids, tgt], dim=1).to(device)
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    if "mm_token_type_ids" in batch:  # 正解の部分は文字(0)
        batch["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(tgt)], dim=1).to(device)
    inner = model.get_base_model().model           # Qwen3_5ForConditionalGeneration.model
    lm_head = model.get_base_model().lm_head
    hidden = inner(**batch, use_cache=False).last_hidden_state[0]
    n_prompt = prompt_ids.shape[-1]
    h = hidden[n_prompt - 1: n_prompt - 1 + len(target)]  # 位置 t の隠れ状態が t+1 のトークンを予測する
    labels = tgt[0]

    def part(h_part, y_part):
        logits = lm_head(h_part.to(lm_head.weight.dtype)).float()
        return F.cross_entropy(logits, y_part.to(logits.device), reduction="sum")

    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, len(target), chunk):
        total = total + checkpoint(part, h[i:i + chunk], labels[i:i + chunk], use_reentrant=False)
    return total / len(target)
