# Inference study: speculative decoding for Verilog generation

Every number below is read from [`results/specdec/specdec.json`](results/specdec/specdec.json), produced by [`silicon_eval/specdec_bench.py`](silicon_eval/specdec_bench.py) (commit `2cf3dea`). Nothing here is rounded up or taken from another run.

## Setup

| | |
|---|---|
| Target model | `Qwen/Qwen2.5-Coder-1.5B` (base, not the fine-tuned SiliconLM checkpoint) |
| Draft model | `Qwen/Qwen2.5-Coder-0.5B` |
| Method | Hugging Face assisted generation, constant draft length K |
| Hardware | NVIDIA GeForce RTX 5050 Laptop GPU, fp16, **on battery power** |
| Problems | 5 Verilog problems in `silicon_eval/problems/` (`adder4`, `counter3`, `dff_arst`, `mux2`, `prienc4`) |
| Generation | up to 300 new tokens, stopped at the first `endmodule` in both arms |
| Correctness | each sample compiled and simulated with Icarus Verilog against a testbench |

## 1. Greedy decoding: acceptance, exactness, speed

10 greedy generations per K (5 problems, repeated), baseline vs. speculative.

| K | Acceptance rate | Mean accepted / round | Tokens / verify round | Baseline tok/s | Speculative tok/s | Speed-up | Output identical to baseline |
|---|---|---|---|---|---|---|---|
| 3 | 85.0% | 2.43 | 3.36 | 6.69 | 6.33 | **0.95x** | 10/10 |
| 5 | 79.9% | 3.70 | 4.59 | 6.75 | 6.25 | **0.93x** | 10/10 |
| 8 | 72.5% | 4.98 | 5.82 | 6.77 | 5.72 | **0.85x** | 10/10 |

What this supports:

- **The draft is a good predictor of the target.** 73–85% of proposed tokens are accepted, so each target forward pass emits 3.4–5.8 tokens instead of 1.
- **Speculative decoding is exact under greedy decoding.** Output was token-identical to the baseline in 30/30 runs across the three K values.
- **Speculative decoding was slower than plain decoding on this machine.** Speed-up is 0.85–0.95x, not above 1x.

### Why it was slower

The benchmark records `notes: "timings taken on battery power, CPU launch-bound"`. At ~6.7 tok/s, the GPU spends its time waiting on kernel launches rather than computing, so a 0.5B draft pass costs almost as much wall-clock time as a 1.5B target pass.

A back-of-envelope check (derived here, not measured directly) agrees. Speed-up ≈ tokens-per-round / (1 + K·c), where c is draft-pass cost relative to a target pass. Solving for c from the table gives roughly 0.85 (K=3), 0.79 (K=5) and 0.73 (K=8). A draft that costs 73–85% of the target cannot pay off, however high the acceptance. For speculative decoding to win you want c well below ~0.3.

So the **acceptance and exactness results are hardware-independent; the speed-up is not**. This study does not show a speed-up from speculative decoding. It shows the conditions under which it didn't help. A speed claim needs a re-run on a GPU-bound setup (e.g. the Kaggle T4 recipe in [`kaggle/README.md`](kaggle/README.md)), plugged in and with the fine-tuned target.

## 2. Does speculative decoding change correctness? (sampled pass@k)

Sampled at temperature 0.7, top-p 0.95, 10 samples per problem × 5 problems = 50 samples per arm. K=5 for the speculative arm.

| | Baseline | Speculative |
|---|---|---|
| Compile rate | 0.60 | 0.70 |
| pass@1 | 0.40 | 0.44 |
| pass@5 | 0.599 | 0.795 |
| pass@10 | 0.60 | 1.00 |
| Correct samples | 20 / 50 (Wilson 95% CI 0.276–0.538) | 22 / 50 (0.312–0.577) |

Two-proportion test on per-sample pass rate: **p = 0.685**. The arms are not distinguishable.

Per problem, correct samples out of 10 (baseline → speculative): `adder4` 0 → 1, `counter3` 7 → 4, `dff_arst` 8 → 8, `mux2` 5 → 8, `prienc4` 0 → 1.

How to read this:

- The pass@5 and pass@10 gaps look large, but pass@10 here only asks "did any of 10 samples pass" for 5 problems, which moves by whole problems (3/5 vs 5/5). The per-sample comparison above is the one to trust, and it shows no difference.
- Theory says speculative decoding leaves the output distribution unchanged, and the greedy identity result in section 1 is consistent with that. The sampled arms differ only by sampling noise.
- With 5 problems and 50 samples per arm, this test is too small to conclude "no effect on quality". It is consistent with none, and it is not evidence of an improvement. The greedy identity check is the stronger evidence of exactness.

## Limitations

- Single hardware configuration, on battery power, launch-bound. Speed numbers should not be quoted as general.
- Target was the base Qwen2.5-Coder-1.5B, not the fine-tuned SiliconLM model. Acceptance may differ for the tuned target (the Kaggle notes flag this).
- Only 5 problems; confidence intervals on pass rates are wide (about ±13 points).
- The greedy "10 runs" per K are repeats over the same 5 prompts, not 10 independent problems.
- Timings use the corrected rerun that trims both arms at the first `endmodule`; earlier runs without that trim are not included.

## Reproduce

```bash
pip install -r requirements.txt && apt-get install -y iverilog   # or install Icarus Verilog locally
python silicon_eval/specdec_bench.py --target Qwen/Qwen2.5-Coder-1.5B \
    --draft Qwen/Qwen2.5-Coder-0.5B --samples 10
```

This writes `results/specdec/specdec.json`. Acceptance rate and greedy identity should reproduce on any hardware. Tokens/s will not.
