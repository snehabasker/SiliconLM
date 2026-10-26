"""Train the small CIFAR-10 classifier used to measure class agreement of conditional samples.

python train_classifier.py --out runs/classifier.pt      (logs test accuracy; this is a measuring instrument, not a result)
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from flowgen.data import load_cifar10, to_float
from flowgen.metrics import SmallResNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    xtr, ytr = load_cifar10(a.data, True)
    xte, yte = load_cifar10(a.data, False)
    net = SmallResNet().cuda()
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = a.epochs * (len(xtr) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, total_steps=steps)
    log, t0 = [], time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(xtr), device="cuda")
        net.train()
        for i in range(0, len(xtr) - 255, 256):
            idx = perm[i:i + 256]
            x = to_float(xtr[idx])
            x = torch.where(torch.rand(256, 1, 1, 1, device="cuda") < 0.5, x.flip(-1), x)
            pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
            dx, dy = torch.randint(0, 9, (2,)).tolist()
            x = pad[:, :, dy:dy + 32, dx:dx + 32]
            loss = F.cross_entropy(net(x), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            acc = sum((net(to_float(xte[i:i + 1000])).argmax(1) == yte[i:i + 1000]).sum().item() for i in range(0, 10000, 1000)) / 10000
        log.append({"epoch": ep + 1, "test_acc": acc, "seconds": round(time.time() - t0, 1)})
        print(log[-1], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), a.out)
    a.out.with_suffix(".json").write_text(json.dumps({"final_test_acc": log[-1]["test_acc"], "log": log, "seed": a.seed}, indent=1))


if __name__ == "__main__":
    main()
