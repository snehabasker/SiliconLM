"""Stage 2: continue pretraining Qwen2.5-Coder-1.5B on our corpus with QLoRA (4-bit + LoRA, fits one T4).

python stage2_dapt.py --corpus corpus.txt --hub you/siliconlm-dapt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

BASE = "Qwen/Qwen2.5-Coder-1.5B"


def build_dataset(corpus: Path, tok, block: int = 1024) -> Dataset:
    text = corpus.read_text(encoding="utf-8", errors="ignore")
    ids = tok(text, return_tensors=None)["input_ids"]
    chunks = [ids[i:i + block] for i in range(0, len(ids) - block, block)]
    return Dataset.from_dict({"input_ids": chunks, "labels": chunks})


@torch.no_grad()
def perplexity(model, tok, texts: list[str], device) -> float:
    model.eval()
    losses = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=1024).to(device)
        out = model(**ids, labels=ids["input_ids"])
        losses.append(out.loss.item())
    return math.exp(sum(losses) / len(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("dapt_out"))
    ap.add_argument("--hub", type=str, default=None, help="HF repo for adapters")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    ds = build_dataset(args.corpus, tok)
    held_out = [tok.decode(ds[i]["input_ids"]) for i in range(min(20, len(ds)))]
    print(f"baseline domain ppl: {perplexity(model, tok, held_out, model.device):.2f}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=20,
            save_steps=200,
            save_total_limit=2,
            fp16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            report_to="none",
            push_to_hub=bool(args.hub),
            hub_model_id=args.hub,
            hub_strategy="checkpoint",
        ),
        train_dataset=ds,
    )
    trainer.train(resume_from_checkpoint=any(args.out.glob("checkpoint-*")))
    print(f"post-DAPT domain ppl: {perplexity(model, tok, held_out, model.device):.2f}")
    trainer.save_model(str(args.out / "final"))
    if args.hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
