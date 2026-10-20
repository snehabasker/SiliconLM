"""Speculative decoding benchmark: HF assisted generation, small draft + larger target.

Reports, for the same target model with and without a draft model:
  * acceptance rate and mean accepted tokens per verification round
  * decode throughput (tokens/s) and speed-up
  * greedy output identity (speculative decoding is exact under greedy)
  * pass@k from the simulator harness, sampled, with confidence intervals

Writes results/specdec/<name>.json. Needs a CUDA GPU for meaningful timings and iverilog on PATH.

    python silicon_eval/specdec_bench.py --target Qwen/Qwen2.5-Coder-1.5B \
        --draft Qwen/Qwen2.5-Coder-0.5B
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silicon_eval.harness import evaluate, load_problems  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results" / "specdec"  # subfolder: deploy/app.py globs results/*.json


class CallCounter:
    """Counts forward passes of a model via a pre-hook."""

    def __init__(self, model):
        self.n = 0
        model.register_forward_pre_hook(lambda *_a, **_k: self._bump())

    def _bump(self):
        self.n += 1

    def reset(self):
        self.n = 0


class RoundStats:
    """Exact per-round draft statistics, read from HF's candidate-generator callback."""

    def __init__(self):
        from transformers.generation.candidate_generator import AssistedCandidateGenerator

        self.rounds = self.proposed = self.accepted = 0
        orig, stats = AssistedCandidateGenerator.update_candidate_strategy, self

        def patched(gen, input_ids, scores, num_matches):
            stats.rounds += 1
            stats.proposed += len(scores[0]) - 1  # last score row is the bonus position
            stats.accepted += int(num_matches)
            return orig(gen, input_ids, scores, num_matches)

        AssistedCandidateGenerator.update_candidate_strategy = patched

    def reset(self):
        self.rounds = self.proposed = self.accepted = 0


def wilson(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = c / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def two_prop_p(c1: int, n1: int, c2: int, n2: int) -> float:
    """Two-sided two-proportion z-test p-value."""
    p = (c1 + c2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (c1 / n1 - c2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-Coder-0.5B")
    ap.add_argument("--name", default="specdec")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5, 8],
                    help="draft tokens per round to sweep (greedy speed test)")
    ap.add_argument("--k_eval", type=int, default=5, help="K used for the pass@k run")
    ap.add_argument("--speed_reps", type=int, default=3)
    ap.add_argument("--samples", type=int, default=20, help="samples per problem for pass@k")
    ap.add_argument("--skip_passk", action="store_true", help="only run the greedy phase")
    ap.add_argument("--smoke", action="store_true", help="1 problem, tiny budget")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.target)
    dtok = AutoTokenizer.from_pretrained(args.draft)
    if tok.get_vocab() != dtok.get_vocab():
        print("WARNING: draft/target vocabularies differ; assisted generation needs "
              "the same tokenizer (or universal assisted decoding).")
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=dtype).to(dev).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=dtype).to(dev).eval()
    t_calls, stats = CallCounter(target), RoundStats()

    problems = load_problems(HERE / "problems")
    if args.smoke:
        problems, args.samples, args.speed_reps, args.k = problems[:1], 2, 1, [5]
        args.max_new_tokens = 200

    def encode(prompt: str):
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # base models without a chat template
            text = prompt + "\n"
        return tok(text, return_tensors="pt").to(dev)

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    def run(ids, k=None, sample=False):
        """One generation. Returns (new_token_ids, seconds, target_calls, draft_calls)."""
        # stop at the first endmodule: extract_verilog keeps only that module anyway,
        # and both arms stop identically
        # (a custom criterion, not stop_strings: the draft's inner generate() would
        # inherit stop_strings without a tokenizer and raise)
        n_prompt = ids["input_ids"].shape[1]
        kw = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.eos_token_id,
                  stopping_criteria=StoppingCriteriaList([
                      lambda inp, scores, **_: torch.tensor(
                          ["endmodule" in tok.decode(row[n_prompt:]) for row in inp],
                          device=inp.device)]))
        if sample:
            kw.update(do_sample=True, temperature=0.7, top_p=0.95)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if k is not None:
            draft.generation_config.num_assistant_tokens = k
            draft.generation_config.num_assistant_tokens_schedule = "constant"
            kw.update(assistant_model=draft)
        t_calls.reset(), stats.reset()
        sync()
        t0 = time.perf_counter()
        out = target.generate(**ids, **kw)
        sync()
        dt = time.perf_counter() - t0
        new = out[0, n_prompt:]
        # speculative rounds can overshoot the stop token by a few tokens: cut both
        # arms at the first endmodule so identity checks and tok/s are comparable
        if "endmodule" in tok.decode(new):
            lo, hi = 1, len(new)
            while lo < hi:  # smallest prefix containing the stop string
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if "endmodule" in tok.decode(new[:mid]) else (mid + 1, hi)
            new = new[:lo]
        return new, dt, stats.rounds, stats.proposed, stats.accepted

    report: dict = {
        "target": args.target, "draft": args.draft, "device": dev,
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
        "max_new_tokens": args.max_new_tokens,
    }

    # ---- warm-up (kernel autotune / allocator) so the first timed run isn't penalised
    w = encode(problems[0].prompt)
    run(w), run(w, k=args.k[0])

    # ---- 1. greedy speed, acceptance, exactness -------------------------------------
    print("\n[1] greedy speed / acceptance / exactness")
    greedy = {}
    for k in args.k:
        base_t = base_n = spec_t = spec_n = rounds = proposed = accepted = 0
        identical = total = 0
        for prob in problems:
            ids = encode(prob.prompt)
            for _ in range(args.speed_reps):
                bt, bdt, *_ = run(ids)
                st, sdt, rc, pc, ac = run(ids, k=k)
                base_t, base_n = base_t + bdt, base_n + len(bt)
                spec_t, spec_n = spec_t + sdt, spec_n + len(st)
                rounds, proposed, accepted = rounds + rc, proposed + pc, accepted + ac
                total += 1
                identical += int(torch.equal(bt, st))
        greedy[f"K={k}"] = {
            "baseline_tok_s": round(base_n / base_t, 2),
            "speculative_tok_s": round(spec_n / spec_t, 2),
            "speedup": round((spec_n / spec_t) / (base_n / base_t), 3),
            "acceptance_rate": round(accepted / max(proposed, 1), 3),
            "mean_accepted_per_round": round(accepted / max(rounds, 1), 3),
            "tokens_per_target_pass": round(spec_n / max(rounds, 1), 3),
            "verify_rounds": rounds, "draft_tokens_proposed": proposed,
            "greedy_identical": f"{identical}/{total}",
        }
        print(f"  K={k}: {json.dumps(greedy[f'K={k}'])}")
    report["greedy"] = greedy
    if args.skip_passk:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{args.name}.json").write_text(json.dumps(report, indent=2))
        return 0

    # ---- 2. pass@k, sampled, baseline vs speculative ---------------------------------
    print("\n[2] pass@k with the simulator harness")
    ks = tuple(k for k in (1, 5, 10) if k <= args.samples)

    def make_gen(k):
        def gen(prompt: str, n: int) -> list[str]:
            ids = encode(prompt)
            outs = []
            for _ in range(n):  # assisted generation is batch-size-1 only
                new, *_ = run(ids, k=k, sample=True)
                outs.append(tok.decode(new, skip_special_tokens=True))
            return outs
        return gen

    reports = {}
    for label, k in (("baseline", None), ("speculative", args.k_eval)):
        t0 = time.perf_counter()
        rep = evaluate(label, make_gen(k), problems, n_samples=args.samples, ks=ks)
        reports[label] = rep
        print(f"  {label:12s} {rep.pass_at} compile_rate={rep.compile_rate} "
              f"({time.perf_counter() - t0:.0f}s)")

    def counts(rep):
        c = sum(p["correct"] for p in rep.per_problem.values())
        n = sum(p["n"] for p in rep.per_problem.values())
        return c, n

    (cb, nb), (cs, ns) = counts(reports["baseline"]), counts(reports["speculative"])
    report["pass_at_k"] = {
        "samples_per_problem": args.samples,
        "n_problems": len(problems),
        "baseline": reports["baseline"].__dict__,
        "speculative": reports["speculative"].__dict__,
        "sample_pass_rate": {
            "baseline": {"correct": cb, "n": nb, "wilson95": [round(x, 3) for x in wilson(cb, nb)]},
            "speculative": {"correct": cs, "n": ns, "wilson95": [round(x, 3) for x in wilson(cs, ns)]},
            "two_proportion_p": round(two_prop_p(cb, nb, cs, ns), 4),
        },
    }
    print(f"  sample pass rate: baseline {cb}/{nb}  speculative {cs}/{ns}  "
          f"p={report['pass_at_k']['sample_pass_rate']['two_proportion_p']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.name}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
