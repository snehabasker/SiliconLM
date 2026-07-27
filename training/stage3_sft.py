"""Stage 3: SFT on {"prompt","response"} pairs from data/make_sft_pairs.py, on top of the DAPT adapters if given.

python stage3_sft.py --pairs sft.jsonl --dapt you/siliconlm-dapt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen2.5-Coder-1.5B"


def load_pairs(path: Path) -> Dataset:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return Dataset.from_list([
        {"messages": [{"role": "user", "content": r["prompt"]},
                      {"role": "assistant", "content": r["response"]}]}
        for r in rows
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--dapt", type=str, default=None, help="Stage-2 adapter repo/path")
    ap.add_argument("--out", type=Path, default=Path("sft_out"))
    ap.add_argument("--hub", type=str, default=None)
    ap.add_argument("--epochs", type=float, default=2.0)
    args = ap.parse_args()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto")
    if args.dapt:  # stack SFT on top of domain-adapted weights
        model = PeftModel.from_pretrained(model, args.dapt, is_trainable=False)
        model = model.merge_and_unload()

    trainer = SFTTrainer(
        model=model,
        train_dataset=load_pairs(args.pairs),
        peft_config=LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
        args=SFTConfig(
            output_dir=str(args.out),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=20,
            save_steps=200,
            save_total_limit=2,
            fp16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            max_length=1536,
            report_to="none",
            push_to_hub=bool(args.hub),
            hub_model_id=args.hub,
        ),
    )
    trainer.train(resume_from_checkpoint=any(args.out.glob("checkpoint-*")))
    trainer.save_model(str(args.out / "final"))
    if args.hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
