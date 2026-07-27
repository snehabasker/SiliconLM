"""Stage 4: DPO on {"prompt","chosen","rejected"} pairs from make_dpo_pairs.py, starting from the SFT model.

python stage4_dpo.py --pairs dpo_pairs.jsonl --sft you/siliconlm-sft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

BASE = "Qwen/Qwen2.5-Coder-1.5B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--sft", type=str, required=True, help="Stage-3 adapter repo/path")
    ap.add_argument("--out", type=Path, default=Path("dpo_out"))
    ap.add_argument("--hub", type=str, default=None)
    ap.add_argument("--beta", type=float, default=0.1)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    ds = Dataset.from_list(rows)

    tok = AutoTokenizer.from_pretrained(BASE)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(model, args.sft, is_trainable=False)
    model = model.merge_and_unload()

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL recreates the frozen reference from `model`
        train_dataset=ds,
        processing_class=tok,
        peft_config=LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
        args=DPOConfig(
            output_dir=str(args.out),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=2,
            learning_rate=5e-6,
            beta=args.beta,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            fp16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
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
