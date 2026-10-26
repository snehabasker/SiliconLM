"""Train a class-conditional flow-matching U-Net on CIFAR-10 from scratch.

python train.py --out runs/fm_base --steps 60000 --seed 0

Loss: E || v_theta(x_t, t, y) - (x1 - x0) ||^2 with x_t = (1-t) x0 + t x1, x0 ~ N(0, I), x1 a training image, t ~ U(0, 1).
Labels are replaced by the null label with prob. `p_uncond` (needed for classifier-free guidance). An EMA of the
weights is kept and is what all evaluations use. Logs one JSON line per `log_every` steps and checkpoints periodically.
"""

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import torch

from flowgen.data import load_cifar10, to_float
from flowgen.model import UNet, UNetConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--p_uncond", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--base", type=int, default=96)
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    x_all, y_all = load_cifar10(args.data, True, dev)
    net = UNet(UNetConfig(base=args.base)).to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    start = 0
    ck = args.out / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        net.load_state_dict(s["net"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"]); start = s["step"]
        print("resumed at", start)
    null = net.cfg.num_classes
    meta = {"args": {k: str(v) for k, v in vars(args).items()}, "params": net.num_params(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(), "cfg": net.cfg.__dict__.copy()}
    meta["cfg"]["mults"] = list(meta["cfg"]["mults"]); meta["cfg"]["attn_levels"] = list(meta["cfg"]["attn_levels"])
    (args.out / "config.json").write_text(json.dumps(meta, indent=1))
    log = open(args.out / "train_log.jsonl", "a")
    g = torch.Generator(device=dev).manual_seed(args.seed + 1000 * start)
    t0, run = time.time(), 0.0
    for step in range(start + 1, args.steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = args.lr * min(1.0, step / args.warmup)
        idx = torch.randint(len(x_all), (args.batch,), device=dev, generator=g)
        x1 = to_float(x_all[idx])
        flip = torch.rand(args.batch, device=dev, generator=g) < 0.5
        x1 = torch.where(flip[:, None, None, None], x1.flip(-1), x1)
        y = torch.where(torch.rand(args.batch, device=dev, generator=g) < args.p_uncond,
                        torch.full_like(y_all[idx], null), y_all[idx])
        x0 = torch.randn(x1.shape, device=dev, generator=g)
        t = torch.rand(args.batch, device=dev, generator=g)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = net(xt.contiguous(memory_format=torch.channels_last), t, y)
        loss = (v.float() - (x1 - x0)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0).item()
        opt.step()
        with torch.no_grad():
            d = min(args.ema, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - d)
        run += loss.item()
        if step % args.log_every == 0:
            log.write(json.dumps({"step": step, "loss": run / args.log_every, "grad_norm": gn,
                                  "elapsed": round(time.time() - t0, 1)}) + "\n"); log.flush()
            run = 0.0
        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step}, ck)
    print("done", args.out)


if __name__ == "__main__":
    main()
