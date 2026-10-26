"""Sampler x step-count sweep: FID (10k samples vs CIFAR-10 train) for a trained flow-matching checkpoint.

python eval_samplers.py --ckpt runs/fm_base/ckpt.pt --out results/samplers.jsonl --seeds 0 \
    --methods euler heun midpoint rk4 sde --nfes 8 16 32 64 --scale 1.0

One JSON line per (method, nfe, scale, seed): FID, NFE, network forward passes, wall-clock seconds. Resumable: lines
already in --out are skipped. Sampling uses bf16 autocast for the network; the ODE state stays fp32.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from flowgen.data import load_cifar10, to_uint8
from flowgen.metrics import InceptionFeatures, fid, reference_stats
from flowgen.model import UNet, UNetConfig
from flowgen.samplers import NFE_PER_STEP, sample


def load_ema(ckpt: str, dev: str = "cuda") -> UNet:
    s = torch.load(ckpt, map_location=dev)
    net = UNet(UNetConfig(base=s["ema"]["inp.weight"].shape[0])).to(dev).eval().requires_grad_(False)
    net.load_state_dict(s["ema"])
    return net


@torch.no_grad()
def generate(net, n: int, method: str, nfe: int, scale: float, eta: float, seed: int, bs: int = 1000, dev="cuda"):
    """n images, classes balanced. -> (uint8 [n,3,32,32] on cpu, labels, passes_per_batch, seconds)."""
    ys = torch.arange(n, device=dev) % 10
    outs, t0 = [], time.time()
    for i in range(0, n, bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x, c = sample(net, ys[i:i + bs], method, nfe, scale, eta, seed * 100003 + i)
        outs.append(to_uint8(x.float()).cpu())
    torch.cuda.synchronize()
    return torch.cat(outs), ys.cpu(), c.passes, c.nfe, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--methods", nargs="+", default=["euler", "heun", "midpoint", "rk4", "sde"])
    ap.add_argument("--nfes", nargs="+", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=1.0, help="noise level of the sde sampler")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()

    net = load_ema(args.ckpt)
    inc = InceptionFeatures()
    xtr, _ = load_cifar10(args.data, True, "cpu")
    ref = reference_stats(inc, xtr, Path(args.data) / "cifar10_train_inception_stats.npz")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for l in args.out.read_text().splitlines():
            r = json.loads(l); done.add((r["method"], r["nfe"], r["scale"], r["seed"]))
    for seed in args.seeds:
        for method in args.methods:
            for nfe in args.nfes:
                if nfe < NFE_PER_STEP[method] or (method, nfe, args.scale, seed) in done:
                    continue
                imgs, _, passes, real_nfe, secs = generate(net, args.n, method, nfe, args.scale, args.eta, seed)
                rec = {"method": method, "nfe": nfe, "nfe_used": real_nfe, "scale": args.scale, "seed": seed, "n": args.n,
                       "eta": args.eta if method == "sde" else None, "fid": fid(inc, imgs, ref),
                       "net_passes": passes, "gen_seconds": round(secs, 1), "ckpt": args.ckpt}
                with args.out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(rec, flush=True)


if __name__ == "__main__":
    main()
