"""Progressive distillation of a guided flow-matching sampler into few-step students (Salimans & Ho, 2022, adapted).

python distill.py --ckpt runs/fm_base/ckpt.pt --out runs/distill --scale 3 --k0 16 --stages 4 --steps 10000

Stage 0 teacher : the base EMA model with classifier-free guidance `scale`, integrated with Heun in K0 steps.
Stage j student : integrates with K0 / 2^j EULER steps and has the guidance baked in (one network pass per step; it is
                  called with the real label only, no null label, no guidance-scale input).
Each student step of size h = 1/K from time t must reproduce TWO consecutive steps (size h/2) of the previous stage's
sampler. Target velocity = (x_after_two_teacher_steps - x_t) / h; loss = MSE to the student velocity. Training inputs
x_t are on the flow-matching marginal path, x_t = (1-t) x0 + t x1 with x1 a training image, t on the student's grid.
Student j starts from the weights of student j-1 (student 1 from the base EMA model). Per stage a student checkpoint
and a train log are written; evaluate them with eval_students.py.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from eval_samplers import load_ema
from flowgen.data import load_cifar10, to_float
from flowgen.samplers import guided_velocity


@torch.no_grad()
def teacher_two_steps(prev_is_base: bool, net, x, t0: float, h2: float, y, scale: float, null: int):
    """Advance x from t0 by two sub-steps of size h2 with the previous stage's sampler."""
    if prev_is_base:                                   # guided base model, Heun
        f = guided_velocity(net, scale, null)
        for k in range(2):
            t = t0 + k * h2
            k1 = f(x, t, y)
            k2 = f(x + h2 * k1, t + h2, y)
            x = x + h2 * 0.5 * (k1 + k2)
        return x
    for k in range(2):                                 # previous student, Euler, label only
        t = t0 + k * h2
        x = x + h2 * net(x, torch.full((x.shape[0],), t, device=x.device), y)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--scale", type=float, required=True, help="guidance scale baked into the students")
    ap.add_argument("--k0", type=int, default=16, help="step count of the stage-0 (Heun) teacher")
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10000, help="optimiser steps per stage")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    base = load_ema(args.ckpt)
    null = base.cfg.num_classes
    xtr = load_cifar10(args.data, True, dev)[0]
    g = torch.Generator(device=dev).manual_seed(args.seed)
    log = open(args.out / "train_log.jsonl", "a")
    (args.out / "config.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}}, indent=1))

    prev = base
    for stage in range(1, args.stages + 1):
        K = args.k0 // 2 ** stage
        assert K >= 1, "too many stages for k0"
        student = copy.deepcopy(prev).train().requires_grad_(True)
        for m in student.modules():                      # dropout off: the regression target is deterministic
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
        ema = copy.deepcopy(student).eval().requires_grad_(False)
        h, t0 = 1.0 / K, time.time()
        for step in range(1, args.steps + 1):
            t = torch.randint(K, (args.batch,), device=dev, generator=g).float() / K
            y = torch.randint(10, (args.batch,), device=dev, generator=g)
            x0 = torch.randn(args.batch, 3, 32, 32, device=dev, generator=g)
            x1 = to_float(xtr[torch.randint(len(xtr), (args.batch,), device=dev, generator=g)])
            xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
            target = torch.empty_like(xt)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for ti in t.unique():                    # the teacher sub-steps need one python-float start time at a time
                    sel = t == ti
                    xe = teacher_two_steps(stage == 1, prev, xt[sel], float(ti), h / 2, y[sel], args.scale, null)
                    target[sel] = (xe.float() - xt[sel]) / h
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = student(xt, t, y)
            loss = (v.float() - target).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                d = min(0.999, (1 + step) / (10 + step))
                for pe, pn in zip(ema.parameters(), student.parameters()):
                    pe.lerp_(pn, 1 - d)
            if step % 100 == 0:
                log.write(json.dumps({"stage": stage, "K": K, "step": step, "loss": loss.item(),
                                      "elapsed": round(time.time() - t0, 1)}) + "\n"); log.flush()
        torch.save({"ema": ema.state_dict(), "K": K, "scale": args.scale, "stage": stage, "base": args.ckpt},
                   args.out / f"student_K{K}.pt")
        prev = ema
        print(f"stage {stage} (K={K}) done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
