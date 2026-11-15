"""Train a PEFT LoRA adapter with TRL's SFTTrainer.

The build environment may not have enough CPU/RAM to train even a 0.5B model.
This module records the real outcome in ``artifacts/train/train_log.json`` and
never pretends a failed run succeeded. When it fails, the fallback reranker in
``fallback_reranker.py`` is the trained component used by the ``rag-tuned``
system.
"""

from __future__ import annotations

import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clinicalnoterag.config import Settings
from clinicalnoterag.seeds import set_global_seed


def _plot(history: list[dict[str, Any]], png_path: Path, title: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train = [(row["step"], row["loss"]) for row in history if "loss" in row and "step" in row]
        evaluation = [
            (row["step"], row["eval_loss"])
            for row in history
            if "eval_loss" in row and "step" in row
        ]
        plt.figure(figsize=(6, 4))
        if train:
            plt.plot([row[0] for row in train], [row[1] for row in train], label="train loss")
        if evaluation:
            plt.plot(
                [row[0] for row in evaluation],
                [row[1] for row in evaluation],
                label="eval loss",
                marker="o",
            )
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(png_path, dpi=150)
        plt.close()
        return str(png_path)
    except Exception:  # noqa: BLE001 - plotting is best-effort
        return None


def train_lora(
    settings: Settings,
    data_dir: str | Path = "data/sft",
    output_dir: str | Path = "artifacts/adapters/clinical-lora",
    log_path: str | Path = "artifacts/train/train_log.json",
    max_steps: int = 12,
    max_length: int = 1024,
) -> dict[str, Any]:
    """Train and save a LoRA adapter, recording the real outcome.

    Returns:
        The training log dict (also written to ``log_path``). ``status`` is
        ``"trained"`` or ``"failed"``; a failed run includes the exception and
        traceback and explicitly points to the fallback reranker.
    """
    set_global_seed(settings.seed)
    log_target = Path(log_path)
    log_target.parent.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {
        "component": "peft-lora",
        "model": settings.llm_primary_model,
        "started_at": datetime.now(UTC).isoformat(),
        "max_steps": max_steps,
        "max_length": max_length,
        "learning_rate": settings.learning_rate,
        "lora_r": settings.lora_r,
        "lora_alpha": settings.lora_alpha,
        "lora_dropout": settings.lora_dropout,
        "target_modules": settings.lora_target_modules,
        "status": "failed",
    }
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        train_path = Path(data_dir) / "train.jsonl"
        val_path = Path(data_dir) / "val.jsonl"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError("SFT train/val JSONL not found; build the dataset first")
        train_rows = [
            json.loads(line)
            for line in train_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        val_rows = [
            json.loads(line)
            for line in val_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        log["n_train"] = len(train_rows)
        log["n_val"] = len(val_rows)

        tokenizer = AutoTokenizer.from_pretrained(settings.llm_primary_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        lengths = [
            len(tokenizer(row["text"], truncation=True, max_length=max_length)["input_ids"])
            for row in train_rows[:20]
        ]
        log["sample_token_lengths"] = lengths

        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            settings.llm_primary_model,
            torch_dtype=dtype,
        )
        model.config.use_cache = False
        log["device"] = "cuda" if use_cuda else "cpu"
        log["dtype"] = str(dtype)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=settings.lora_r,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            target_modules=[
                module.strip()
                for module in settings.lora_target_modules.split(",")
                if module.strip()
            ],
            bias="none",
        )
        sft_config = SFTConfig(
            output_dir=str(output_dir),
            max_steps=max_steps,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=settings.learning_rate,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=max(1, max_steps // 4),
            save_strategy="no",
            report_to="none",
            use_cpu=not use_cuda,
            bf16=use_cuda and torch.cuda.is_bf16_supported(),
            fp16=False,
            max_length=max_length,
            dataset_text_field="text",
            completion_only_loss=False,
            packing=False,
            seed=settings.seed,
            data_seed=settings.seed,
            warmup_steps=1,
            gradient_checkpointing=False,
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=Dataset.from_list([{"text": row["text"]} for row in train_rows]),
            eval_dataset=Dataset.from_list([{"text": row["text"]} for row in val_rows]),
            peft_config=lora_config,
            processing_class=tokenizer,
        )
        trainer.train()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        history = trainer.state.log_history
        log.update(
            {
                "status": "trained",
                "finished_at": datetime.now(UTC).isoformat(),
                "adapter_dir": str(output_dir),
                "history": history,
                "log_history_points": len(history),
                "train_runtime_sec": round(
                    float(trainer.state.log_history[-1].get("train_runtime", 0.0)), 3
                )
                if history
                else 0.0,
            }
        )
        png = _plot(history, Path("artifacts/train/lora_curve.png"), "LoRA SFT training curve")
        if png:
            log["curve_png"] = png
    except Exception as exc:  # noqa: BLE001 - record the real failure
        log.update(
            {
                "status": "failed",
                "failed_at": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "fallback": "fallback-mlp-reranker (see artifacts/train/reranker_log.json)",
            }
        )
    log_target.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def load_adapter_generator(settings: Settings, adapter_path: str | Path):
    """Load a local HF generator with a trained PEFT adapter, if available."""
    from clinicalnoterag.rag.generators import HFLocalGenerator

    return HFLocalGenerator(settings, adapter_path=adapter_path)


__all__ = ["load_adapter_generator", "train_lora"]
