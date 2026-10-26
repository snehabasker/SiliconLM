"""FID / class agreement of distilled students (Euler, K steps, label only) next to the guided teacher sampler.

python eval_students.py --base runs/fm_base/ckpt.pt --students runs/distill --classifier runs/classifier.pt \
    --scale 3 --k0 16 --out results/distill_eval.jsonl

Rows: the stage-0 teacher (guided Heun, K0 steps), then every student_K*.pt, and for context the BASE model with the same
guidance sampled with the same number of NFE as each student (Euler and Heun), which is the fair "no distillation" baseline.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_samplers import generate, load_ema
from flowgen.data import load_cifar10, to_float
from flowgen.metrics import InceptionFeatures, SmallResNet, fid, reference_stats, stats
from flowgen.model import UNet, UNetConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--students", type=Path, required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--k0", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()

    inc = InceptionFeatures()
    clf = SmallResNet().cuda().eval()
    clf.load_state_dict(torch.load(a.classifier))
    xtr, _ = load_cifar10(a.data, True, "cpu")
    ref = reference_stats(inc, xtr, Path(a.data) / "cifar10_train_inception_stats.npz")
    base = load_ema(a.base)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def record(name, net, method, nfe, scale, seed, K=None):
        imgs, ys, passes, real_nfe, secs = generate(net, a.n, method, nfe, scale, 0.0, seed)
        with torch.no_grad():
            pred = torch.cat([clf(to_float(imgs[i:i + 1000]).cuda()).argmax(1).cpu() for i in range(0, len(imgs), 1000)])
        rec = {"model": name, "method": method, "nfe": real_nfe, "net_passes": passes, "scale": scale, "seed": seed,
               "n": a.n, "fid": fid(inc, imgs, ref), "class_agreement": float((pred == ys).float().mean()),
               "gen_seconds": round(secs, 1)}
        with a.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)

    for seed in a.seeds:
        record("teacher_guided_heun", base, "heun", 2 * a.k0, a.scale, seed)
        for p in sorted(a.students.glob("student_K*.pt"), key=lambda q: -int(q.stem.split("K")[1])):
            s = torch.load(p, map_location="cuda")
            net = UNet(UNetConfig(base=s["ema"]["inp.weight"].shape[0])).cuda().eval().requires_grad_(False)
            net.load_state_dict(s["ema"])
            K = s["K"]
            record(f"student_K{K}", net, "euler", K, 1.0, seed)          # scale 1.0 = label only; guidance is baked in
            record(f"base_guided_euler_nfe{K}", base, "euler", K, a.scale, seed)
            if K >= 2:
                record(f"base_guided_heun_nfe{K}", base, "heun", K, a.scale, seed)


if __name__ == "__main__":
    main()
