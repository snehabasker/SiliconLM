# SiliconLM

A small LLM adapted for Verilog/chip-design work, trained end to end (from-scratch pretraining through DPO) on free-tier compute only.

## Why

General-purpose LLMs tend to do badly on semiconductor-specific text — you can check this yourself with `data/tokenizer_analysis.py`, which measures how many tokens a stock tokenizer needs for chip-design terms vs plain English. This follows the same idea as ChipNeMo (domain-adapt a small base model rather than prompt a big general one), but the DPO preference signal here doesn't need human labeling: sample Verilog from the model, run it through Icarus Verilog, and use "did it compile and pass the testbench" as chosen/rejected. No annotators, and it's cheap to scale.

Every number reported anywhere in this repo (README, ablation table, HF Space) comes from that same simulator check, not from an LLM judge. `silicon_eval/run_eval.py --selftest` proves the harness works before trusting anything it says: known-good code must score 1.0, a known-broken baseline must score 0.0.

## Pipeline

1. **Stage 0** (`training/stage0_scratch/`) — a small GPT (RoPE, SwiGLU, RMSNorm) trained from a random init in plain PyTorch, no `transformers.AutoModel`. Byte-level tokenizer, runs on CPU for a smoke test or a free T4 for a real run.
2. **Stage 2 DAPT** (`training/stage2_dapt.py`) — continue pretraining Qwen2.5-Coder-1.5B on the domain corpus, QLoRA 4-bit so it fits on one T4.
3. **Stage 3 SFT** (`training/stage3_sft.py`) — instruction tuning on prompt/response pairs, filtered through the simulator so broken code never enters the training set (`data/make_sft_pairs.py`).
4. **Stage 4 DPO** (`training/stage4_dpo.py`) — preference pairs built by sampling the SFT model and simulating every completion (`training/make_dpo_pairs.py`): passing completions are "chosen," failing ones are "rejected."
5. **Eval** (`silicon_eval/`) — every checkpoint gets scored the same way: compile + simulate against hidden testbenches, pass@k.
6. **Deploy** (`deploy/`) — a Gradio app (generate, verify, ablation table, RAG over datasheets), quantized to GGUF so it runs on a free CPU Space.

## Repo map

| Path | What's there |
|---|---|
| `silicon_eval/` | the pass@k harness + 5 benchmark problems with hidden self-checking testbenches |
| `training/stage0_scratch/` | the from-scratch GPT (`model.py`) and its trainer (`train.py`) |
| `training/stage2_dapt.py` | QLoRA domain-adaptive pretraining |
| `training/stage3_sft.py` | instruction tuning |
| `training/make_dpo_pairs.py` | generates the DPO preference pairs via simulation |
| `training/stage4_dpo.py` | DPO |
| `data/` | corpus builder, tokenizer fragmentation check, SFT pair generation |
| `rag/` | FAISS + bge-small RAG over datasheets, refuses to answer below a retrieval-score floor |
| `deploy/` | the Gradio app |
| `kaggle/` | notes on fitting each stage into Kaggle's free 30 GPU-h/week |

## Running it

```bash
# harness (needs iverilog: apt-get install iverilog)
python silicon_eval/make_problems.py
python silicon_eval/run_eval.py --selftest      # reference=1.0, buggy=0.0

# stage 0 (CPU smoke test: --max_iters 30, or a few hours on a T4 for real)
python training/stage0_scratch/train.py --corpus corpus.txt --out ckpt/

# stages 2-4 need a GPU (bitsandbytes 4-bit) — see kaggle/README.md
python training/stage2_dapt.py --corpus corpus.txt --hub you/siliconlm-dapt
python training/stage3_sft.py  --pairs sft.jsonl --dapt you/siliconlm-dapt --hub you/siliconlm-sft
python training/make_dpo_pairs.py --model you/siliconlm-sft --problems silicon_eval/problems --out dpo.jsonl
python training/stage4_dpo.py  --pairs dpo.jsonl --sft you/siliconlm-sft --hub you/siliconlm-dpo

# score any checkpoint
python silicon_eval/run_eval.py --hf Qwen/Qwen2.5-Coder-1.5B --name base
python silicon_eval/run_eval.py --hf you/siliconlm-dpo       --name dpo

# demo app
python deploy/app.py
```

## Cost

Everything here runs on free tiers: Kaggle's T4 quota for the training stages, Mistral/Groq free tiers for drafting SFT prompts, a local Icarus Verilog install (no cost) for every eval number, and an HF CPU Space for hosting. The point of keeping it free isn't frugality for its own sake — it's a check that a small, deeply-adapted model can beat a bigger model you only prompted, without needing a training budget to prove it.

## License

Apache-2.0. Base model is Qwen2.5-Coder-1.5B (Apache-2.0). The benchmark problems and testbenches in `silicon_eval/` are original to this repo.
