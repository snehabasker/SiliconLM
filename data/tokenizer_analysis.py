"""Check how many tokens the base tokenizer needs for chip-design terms vs plain English.

python tokenizer_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-Coder-1.5B"

DOMAIN_TERMS = [
    "MOSFET", "FinFET", "photolithography", "electromigration", "metastability",
    "setup/hold slack", "clock domain crossing", "wafer bin map", "ionic contamination",
    "chemical mechanical planarization", "shallow trench isolation", "gate oxide",
    "threshold voltage", "subthreshold leakage", "silicidation", "vias", "netlist",
    "floorplanning", "design rule check", "parasitic extraction", "testbench",
    "synthesizable", "nonblocking assignment", "sensitivity list", "tapeout",
    "FDC traces", "SECS/GEM", "overlay misregistration", "epitaxy", "dopant",
]

EVERYDAY_TERMS = [
    "table", "morning", "believe", "running", "beautiful", "computer", "house",
    "yesterday", "important", "children", "water", "understand", "different",
    "question", "together", "development", "information", "possible", "example",
    "government", "education", "business", "history", "national", "problem",
    "company", "money", "music", "market", "family",
]


def fertility(tok, terms: list[str]) -> tuple[float, list[tuple[str, int]]]:
    counts = [(t, len(tok(t, add_special_tokens=False)["input_ids"])) for t in terms]
    words = sum(len(t.split()) for t in terms)
    return sum(c for _, c in counts) / words, sorted(counts, key=lambda x: -x[1])


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    dom_f, dom = fertility(tok, DOMAIN_TERMS)
    eng_f, _ = fertility(tok, EVERYDAY_TERMS)

    print(f"tokenizer: {MODEL}")
    print(f"fertility (tokens/word) — everyday English : {eng_f:.2f}")
    print(f"fertility (tokens/word) — semiconductor     : {dom_f:.2f}")
    print(f"domain overhead: {dom_f / eng_f:.2f}x\n")
    print("worst-fragmented domain terms:")
    for term, n in dom[:8]:
        pieces = tok.convert_ids_to_tokens(tok(term, add_special_tokens=False)["input_ids"])
        print(f"  {term:35s} -> {n:2d} tokens  {pieces}")

    Path("tokenizer_report.json").write_text(json.dumps({
        "tokenizer": MODEL,
        "fertility_english": round(eng_f, 3),
        "fertility_domain": round(dom_f, 3),
        "overhead_x": round(dom_f / eng_f, 3),
        "worst": [{"term": t, "tokens": n} for t, n in dom[:10]],
    }, indent=2))
    print("\nwrote tokenizer_report.json")


if __name__ == "__main__":
    main()
