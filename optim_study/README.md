# optim_study: AdamW vs SGD+momentum vs Lion vs Adafactor on the from-scratch GPT

**Status: work in progress. No training results yet.** Only the code and the corpus preparation are in place. No
optimiser comparison has been run, so there are no loss numbers in this folder.

## What exists

- `train_opt.py`: one training run of `VerilogGPT` (imported unchanged from `training/stage0_scratch/model.py`) with a
  chosen optimiser. Model init and batch order depend on the seed only, not on the optimiser. Token budget, warmup +
  cosine schedule, gradient clipping (1.0), bf16 autocast and weight-decay placement (matrices only) are identical across
  optimisers; only the learning rate is meant to be tuned per optimiser. It logs a JSONL file per run, including
  divergence step, fraction of clipped steps and loss spikes. Lion is implemented in the file; Adafactor is
  `torch.optim.Adafactor`.
- `make_corpus.py`: takes the first 80,000 rows of the `shailja/Verilog_GitHub` dataset, writes them as `.v` files
  and runs the repo's `data/build_corpus.py` unchanged.
- `results/corpus_stats.json`: 80,000 files scanned, 51,480 kept, 2,806 oversize, 25,714 duplicates, 439.49 MB,
  sha256 `52e551e2...`. The corpus itself is not committed (per-file licences are unknown).

Only a 20-step CPU smoke test of `train_opt.py` has been run, for all four optimisers.

## Planned, not done

Learning-rate sweep per optimiser, then 3 seeds each at equal token budgets, final validation loss with intervals, and
any instability seen.

## Limitations to keep in mind when results exist

- The model is small (about 11M parameters) and byte-level, so conclusions may not carry over to larger models.
- Weight decay is fixed per optimiser, not tuned; only the learning rate is.
