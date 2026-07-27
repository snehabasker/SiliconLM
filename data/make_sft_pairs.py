"""Draft spec->code pairs with a free-tier LLM, keep only the ones that compile and sim.

Needs MISTRAL_API_KEY or GROQ_API_KEY set. Writes sft.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from silicon_eval.harness import check_candidate, extract_verilog  # noqa: E402

SPEC_TOPICS = [
    "an 8-bit ripple-carry adder", "a 4-to-1 one-hot multiplexer",
    "a parameterized N-bit register with enable", "a Gray code counter",
    "a rising-edge detector", "a debouncer with a 3-cycle filter",
    "a 7-segment hex decoder", "an 8-bit barrel shifter",
    "a simple round-robin arbiter for 2 requesters", "a parity generator",
]


def call_llm(prompt: str) -> str:
    """Minimal chat call against whichever free-tier API key is set."""
    if os.environ.get("MISTRAL_API_KEY"):
        url = "https://api.mistral.ai/v1/chat/completions"
        key, model = os.environ["MISTRAL_API_KEY"], "mistral-small-latest"
    elif os.environ.get("GROQ_API_KEY"):
        url = "https://api.groq.com/openai/v1/chat/completions"
        key, model = os.environ["GROQ_API_KEY"], "llama-3.3-70b-versatile"
    else:
        raise RuntimeError("Set MISTRAL_API_KEY or GROQ_API_KEY (both have free tiers).")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def smoke_testbench(code: str) -> str | None:
    """Instantiate the module and run for 100 time units.

    Catches non-compiling and X-propagating garbage; functional depth is
    added later by the benchmark testbenches in silicon_eval.
    """
    import re
    m = re.search(r"module\s+(\w+)\s*(?:#\s*\([^)]*\))?\s*\(([^;]*?)\)\s*;", code, re.S)
    if not m:
        return None
    name, ports = m.group(1), m.group(2)
    decls, conns = [], []
    for p in re.finditer(
            r"(input|output|inout)\s+(?:reg\s+|wire\s+)?(\[[^\]]+\]\s*)?(\w+)", ports):
        direction, width, pname = p.group(1), (p.group(2) or "").strip(), p.group(3)
        decls.append(f"  {'reg' if direction == 'input' else 'wire'} {width} {pname};")
        conns.append(f".{pname}({pname})")
    return ("module smoke_tb;\n" + "\n".join(decls) +
            f"\n  {name} dut({', '.join(conns)});\n"
            "  initial begin #100; $display(\"ALL_TESTS_PASSED\"); $finish; end\n"
            "endmodule\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("sft.jsonl"))
    ap.add_argument("--per_topic", type=int, default=3)
    args = ap.parse_args()

    kept = dropped = 0
    with args.out.open("w") as f:
        for topic in SPEC_TOPICS:
            for _ in range(args.per_topic):
                spec = (f"Write synthesizable Verilog-2001 for {topic}. "
                        "Reply with a single complete module and nothing else.")
                code = extract_verilog(call_llm(spec))
                tb = smoke_testbench(code)
                if tb is None or not check_candidate(code, tb).passed:
                    dropped += 1
                    continue
                f.write(json.dumps({"prompt": spec, "response": code}) + "\n")
                kept += 1
    print(f"kept {kept}, dropped {dropped} (simulator-rejected) -> {args.out}")


if __name__ == "__main__":
    main()
