"""Classifier-free guidance sweep at a fixed sampler: FID, class agreement and feature diversity per guidance scale.

python eval_cfg.py --ckpt runs/fm_base/ckpt.pt --classifier runs/classifier.pt --method heun --nfe 32 \
    --scales 1 1.5 2 3 4 6 --out results/cfg.jsonl

class_agreement = fraction of generated images that the separate CIFAR-10 classifier assigns to the class the sampler
was asked for. diversity_ratio = trace(Cov of Inception features of generated) / trace(Cov of real train features);
< 1 means less spread than real data (the usual price of guidance).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_samplers import generate, load_ema
from flowgen.data import load_cifar10, to_float
from flowgen.metrics import InceptionFeatures, SmallResNet, fid, reference_stats, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--method", default="heun")
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--scales", nargs="+", type=float, default=[1, 1.5, 2, 3, 4, 6])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()

    net = load_ema(args.ckpt)
    clf = SmallResNet().cuda().eval()
    clf.load_state_dict(torch.load(args.classifier))
    inc = InceptionFeatures()
    xtr, _ = load_cifar10(args.data, True, "cpu")
    cache = Path(args.data) / "cifar10_train_inception_stats.npz"
    ref = reference_stats(inc, xtr, cache)
    real_trace = float(np.trace(ref[1]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for s in args.scales:
            imgs, ys, passes, nfe, secs = generate(net, args.n, args.method, args.nfe, s, 0.0, seed)
            with torch.no_grad():
                pred = torch.cat([clf(to_float(imgs[i:i + 1000]).cuda()).argmax(1).cpu() for i in range(0, len(imgs), 1000)])
            _, sig = stats(inc(imgs))
            rec = {"method": args.method, "nfe": args.nfe, "scale": s, "seed": seed, "n": args.n, "fid": fid(inc, imgs, ref),
                   "class_agreement": float((pred == ys).float().mean()), "diversity_ratio": float(np.trace(sig)) / real_trace,
                   "net_passes": passes, "gen_seconds": round(secs, 1), "ckpt": args.ckpt}
            with args.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


if __name__ == "__main__":
    main()
