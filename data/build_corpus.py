"""Build corpus.txt from a folder of .v/.sv/.txt/.md files.

Drops anything >100KB (usually a generated netlist), dedupes exact
matches by hash, then dedupes near-matches with 5-gram shingles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

MAX_FILE_BYTES = 100_000
SHINGLE = 5
NEAR_DUP_JACCARD = 0.85


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def shingles(text: str, k: int = SHINGLE) -> set[int]:
    toks = text.split()
    return {hash(" ".join(toks[i:i + k])) for i in range(max(len(toks) - k, 1))}


def near_dup(a: set[int], b: set[int]) -> bool:
    if not a or not b:
        return False
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter) >= NEAR_DUP_JACCARD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, required=True,
                    help="directory of raw .v/.sv/.txt/.md files")
    ap.add_argument("--out", type=Path, default=Path("corpus.txt"))
    args = ap.parse_args()

    seen_exact: set[str] = set()
    kept_shingles: list[set[int]] = []
    docs: list[str] = []
    n_raw = n_big = n_dup = 0

    for path in sorted(args.sources.rglob("*")):
        if path.suffix.lower() not in {".v", ".sv", ".txt", ".md", ".vh"}:
            continue
        n_raw += 1
        if path.stat().st_size > MAX_FILE_BYTES:
            n_big += 1
            continue
        text = clean(path.read_text(encoding="utf-8", errors="ignore"))
        if len(text) < 50:
            continue
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen_exact:
            n_dup += 1
            continue
        sh = shingles(text)
        if any(near_dup(sh, prev) for prev in kept_shingles[-500:]):
            n_dup += 1
            continue
        seen_exact.add(h)
        kept_shingles.append(sh)
        docs.append(text)

    args.out.write_text("\n\n".join(docs), encoding="utf-8")
    stats = {
        "files_scanned": n_raw, "kept": len(docs),
        "dropped_oversize": n_big, "dropped_duplicate": n_dup,
        "corpus_mb": round(args.out.stat().st_size / 1e6, 2),
    }
    Path("corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
