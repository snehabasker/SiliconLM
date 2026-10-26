"""One training run of the from-scratch VerilogGPT with a chosen optimiser, for the optimiser comparison.

python optim_study/train_opt.py --corpus corpus.txt --opt adamw --lr 1e-3 --seed 0 --steps 3000 --out runs/x.jsonl

Fixed across optimisers (so only the optimiser and its learning rate differ): model + init (from `seed`), the batch
sequence (from `seed`, independent of the optimiser), token budget, warmup + cosine schedule shape, gradient clipping,
bf16 autocast, which parameters get weight decay (matrices only). Imports the model from training/stage0_scratch
unchanged. Writes one JSON line per logged step plus a final `summary` line; nothing is estimated or smoothed here.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "stage0_scratch"))
from model import GPTConfig, VerilogGPT  # noqa: E402

VOCAB = 256  # byte-level, as in stage0_scratch/train.py


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023): update = sign(b1*m + (1-b1)*g); m = b2*m + (1-b2)*g; decoupled weight decay."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_((m * b1 + p.grad * (1 - b1)).sign_(), alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


# weight decay per optimiser (decoupled): AdamW 0.1 as in stage0_scratch/train.py; Lion 0.3 (Lion's lr is ~3x smaller
# than AdamW's, so 3x the decay keeps lr*wd comparable); SGD and Adafactor 0.0. Only the lr is tuned.
WD = {"adamw": 0.1, "lion": 0.3, "sgdm": 0.0, "adafactor": 0.0}


def build_opt(name: str, model: torch.nn.Module, lr: float):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    wd = WD[name]
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    if name == "sgdm":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9)
    if name == "lion":
        return Lion(groups, lr=lr, betas=(0.9, 0.99))
    if name == "adafactor":
        return torch.optim.Adafactor(groups, lr=lr)   # torch's implementation, library defaults otherwise
    raise ValueError(name)


def lr_at(step: int, total: int, peak: float, warmup: int, floor: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--opt", choices=list(WD), required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_batches", type=int, default=64)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.frombuffer(bytearray(args.corpus.read_bytes()), dtype=torch.uint8)
    n_val = max(args.block_size * args.eval_batches * 2, int(args.val_frac * len(data)))
    train_d, val_d = data[:-n_val].to(dev), data[-n_val:].to(dev)

    torch.manual_seed(args.seed)
    model = VerilogGPT(GPTConfig(vocab_size=VOCAB, block_size=args.block_size)).to(dev)
    opt = build_opt(args.opt, model, args.lr)
    gen = torch.Generator(device="cpu").manual_seed(10_000 + args.seed)       # batch order: seed only, not optimiser
    val_gen = torch.Generator(device="cpu").manual_seed(777)                   # identical val batches for every run
    val_ix = [torch.randint(len(val_d) - args.block_size - 1, (args.batch_size,), generator=val_gen)
              for _ in range(args.eval_batches)]
    ar = torch.arange(args.block_size)

    def batch(d, ix):
        ix = ix.to(dev)
        return d[ix[:, None] + ar.to(dev)].long(), d[ix[:, None] + ar.to(dev) + 1].long()   # corpus stays uint8 on the GPU

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        tot = 0.0
        for ix in val_ix:
            x, y = batch(val_d, ix)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                _, loss = model(x, y)
            tot += loss.item()
        model.train()
        return tot / len(val_ix)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    f = args.out.open("w")
    meta = {"type": "meta", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "wd": WD[args.opt], "params": model.num_params(), "device": dev,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None, "torch": torch.__version__,
            "python": platform.python_version(), "train_bytes": len(train_d), "val_bytes": len(val_d),
            "tokens_per_step": args.batch_size * args.block_size,
            "total_tokens": args.batch_size * args.block_size * args.steps}
    f.write(json.dumps(meta) + "\n")

    t0, diverged, clipped, losses, spikes, run_min = time.time(), None, 0, [], 0, float("inf")
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        ix = torch.randint(len(train_d) - args.block_size - 1, (args.batch_size,), generator=gen)
        x, y = batch(train_d, ix)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        l = loss.item()
        if not math.isfinite(l) or not math.isfinite(gn):
            diverged = step
            f.write(json.dumps({"type": "step", "step": step, "loss": None if not math.isfinite(l) else l,
                                "grad_norm": None, "nonfinite": True}) + "\n")
            break
        clipped += gn > args.clip
        losses.append(l)
        # a "spike": loss more than 0.5 nats above the minimum of the running 50-step mean seen so far
        if len(losses) >= 50:
            m50 = sum(losses[-50:]) / 50
            run_min = min(run_min, m50)
            spikes += (m50 > run_min + 0.5)
        opt.step()
        if step % args.log_every == 0:
            f.write(json.dumps({"type": "step", "step": step, "loss": l, "grad_norm": gn,
                                "lr": opt.param_groups[0]["lr"]}) + "\n")
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            f.write(json.dumps({"type": "eval", "step": step + 1, "val_loss": val_loss(),
                                "seconds": round(time.time() - t0, 1)}) + "\n")
            f.flush()
    final = None if diverged is not None else val_loss()
    f.write(json.dumps({"type": "summary", "diverged_at_step": diverged, "final_val_loss": final,
                        "final_train_loss_last100": (sum(losses[-100:]) / len(losses[-100:])) if losses else None,
                        "steps_run": len(losses), "frac_steps_clipped": clipped / max(1, len(losses)),
                        "spike_steps": spikes, "seconds": round(time.time() - t0, 1),
                        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None}) + "\n")
    f.close()
    print(f"{args.opt} lr={args.lr} seed={args.seed}: final val {final} diverged_at={diverged}")


if __name__ == "__main__":
    main()
