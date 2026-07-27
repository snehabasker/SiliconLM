# Running every stage inside Kaggle's free quota (30 GPU-h/week)

Weekly plan (T4 x2 where it helps):
  Week 1 — corpus build (CPU), tokenizer study (CPU), Stage 0 (~3 h)
  Week 2 — Stage 2 DAPT: 2 sessions x ~9 h, resumable (~18 h)
  Week 3 — SFT pairs (free-tier API, CPU) + Stage 3 SFT (~8 h)
  Week 4 — DPO pair sampling (~4 h GPU) + Stage 4 DPO (~6 h)
  Week 5 — full eval sweep (~4 h) + GGUF quantization (CPU) + Space

Session rules that make this painless:
  1. Add HF_TOKEN as a Kaggle secret; every script pushes checkpoints
     to the Hub (hub_strategy="checkpoint") so a dead session costs at
     most the interval since the last save.
  2. Re-running any train script auto-resumes from checkpoint-*.
  3. Keep eval --samples modest (5): simulation is cheap, generation
     is what burns GPU time.
  4. `apt-get install iverilog` works inside Kaggle notebooks for the
     DPO-pair and eval steps.
