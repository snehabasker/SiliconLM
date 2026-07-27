"""Sample completions from the SFT model, sim each one, pair passing vs failing as chosen/rejected for DPO.

python make_dpo_pairs.py --model you/siliconlm-sft --problems ../silicon_eval/problems --out dpo_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silicon_eval.harness import check_candidate, extract_verilog, load_problems  # noqa: E402
from silicon_eval.run_eval import hf_generator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("dpo_pairs.jsonl"))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max_pairs_per_prompt", type=int, default=4)
    args = ap.parse_args()

    gen = hf_generator(args.model)
    n_pairs = 0
    with args.out.open("w") as f:
        for prob in load_problems(args.problems):
            outputs = gen(prob.prompt, args.samples)
            chosen, rejected = [], []
            for raw in outputs:
                code = extract_verilog(raw)
                res = check_candidate(code, prob.testbench)
                (chosen if res.passed else rejected).append(code)
            # de-duplicate to avoid degenerate identical pairs
            chosen, rejected = list(dict.fromkeys(chosen)), list(dict.fromkeys(rejected))
            for c, r in list(product(chosen, rejected))[: args.max_pairs_per_prompt]:
                f.write(json.dumps({
                    "prompt": prob.prompt, "chosen": c, "rejected": r,
                }) + "\n")
                n_pairs += 1
            print(f"{prob.pid}: {len(chosen)} pass / {len(rejected)} fail")
    print(f"wrote {n_pairs} preference pairs to {args.out}")


if __name__ == "__main__":
    main()
