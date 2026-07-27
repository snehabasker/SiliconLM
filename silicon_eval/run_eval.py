"""Run the eval harness.

--selftest checks the harness itself: reference solutions must score
pass@1=1.0, a deliberately-broken generator must score 0.0.
--hf scores any HF causal LM. Both write results/<name>.json, which
deploy/app.py reads for the ablation tab.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silicon_eval.harness import evaluate, load_problems  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"


def reference_generator(problems):
    """Emit each problem's reference solution — harness sanity ceiling."""
    ref = {p.prompt: p.reference for p in problems}
    return lambda prompt, n: [ref[prompt]] * n


def buggy_generator(problems):
    """Emit compilable-but-wrong code — harness must score this 0."""
    hdr = {p.prompt: p.module_header for p in problems}

    def gen(prompt, n):
        h = hdr[prompt]
        # ties every output to zero: compiles, fails every testbench
        body = h + "\n" + "\n".join(
            f"  assign {name} = 0;" for name in _output_names(h)
        ) + "\nendmodule"
        return [body] * n

    return gen


def _output_names(header: str) -> list[str]:
    import re
    return re.findall(r"output\s+(?:reg\s+)?(?:\[[^\]]+\]\s*)?(\w+)", header)


def hf_generator(model_id: str, max_new_tokens: int = 400):
    """Wrap any HF causal LM as a generate_fn (GPU strongly recommended)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    def gen(prompt: str, n: int) -> list[str]:
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # base models without a chat template
            text = prompt + "\n"
        ids = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids, do_sample=True, temperature=0.7, top_p=0.95,
            num_return_sequences=n, max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
        return [tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True)
                for o in out]

    return gen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--hf", type=str, help="HuggingFace model id or local path")
    ap.add_argument("--name", type=str, default=None, help="label for results file")
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()

    problems = load_problems(HERE / "problems")
    if not problems:
        print("No problems found — run make_problems.py first.")
        return 1
    RESULTS.mkdir(exist_ok=True)

    if args.selftest:
        ok = True
        for label, gen, want in [
            ("reference", reference_generator(problems), 1.0),
            ("buggy-baseline", buggy_generator(problems), 0.0),
        ]:
            rep = evaluate(label, gen, problems, n_samples=2, ks=(1,))
            got = rep.pass_at["pass@1"]
            status = "OK " if got == want else "FAIL"
            ok &= got == want
            print(f"[{status}] {label:15s} pass@1={got:.2f} (expected {want:.2f}) "
                  f"compile_rate={rep.compile_rate:.2f}")
            (RESULTS / f"{label}.json").write_text(rep.to_json())
        print("HARNESS SELF-TEST PASSED" if ok else "HARNESS SELF-TEST FAILED")
        return 0 if ok else 1

    if args.hf:
        name = args.name or args.hf.split("/")[-1]
        rep = evaluate(name, hf_generator(args.hf), problems, n_samples=args.samples)
        (RESULTS / f"{name}.json").write_text(rep.to_json())
        print(rep.to_json())
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
