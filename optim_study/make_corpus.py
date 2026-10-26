"""Build the Verilog pretraining corpus for the optimiser study from the public shailja/Verilog_GitHub dump.

python optim_study/make_corpus.py --csv ~/data/verilog_raw/Verilog_bigquery_GitHub.csv --max_files 80000 \
       --work ~/data/verilog_files --out ~/data/corpus_optim.txt

Step 1 writes the first `max_files` rows of the CSV (a single `text` column) to one .v file each.
Step 2 runs the repo's own data/build_corpus.py unchanged (drops files > 100 KB, exact + 5-gram near-duplicate removal).
The dump carries no per-file licence, so the corpus is for this experiment only and is never redistributed.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--max_files", type=int, default=80000)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stats_out", type=Path, default=Path(__file__).parent / "results" / "corpus_stats.json")
    args = ap.parse_args()
    csv.field_size_limit(1 << 30)
    args.work.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.csv.open(newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            (args.work / f"{n:06d}.v").write_text(row["text"], encoding="utf-8")
            n += 1
            if n >= args.max_files:
                break
    print(f"wrote {n} files to {args.work}")
    out = args.out.resolve()
    subprocess.run([sys.executable, str(ROOT / "data" / "build_corpus.py"), "--sources", str(args.work), "--out", str(out)],
                   check=True, cwd=args.out.parent)   # build_corpus.py writes corpus_stats.json into the cwd
    stats = json.loads((args.out.parent / "corpus_stats.json").read_text())
    stats.update({"rows_taken": n, "source": "huggingface.co/datasets/shailja/Verilog_GitHub (Verilog_bigquery_GitHub.csv)",
                  "corpus_bytes": out.stat().st_size, "corpus_sha256": hashlib.sha256(out.read_bytes()).hexdigest()})
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
