"""Train VerilogGPT from a random init on a text corpus. Byte-level tokenizer, no external deps.

Resumes from ckpt/verilog_gpt.pt if it exists. CPU is fine for a smoke test (--max_iters 20-30).

python train.py --corpus path/to/verilog.txt --out ckpt/ [--max_iters N]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from model import GPTConfig, VerilogGPT

VOCAB = 256  # byte-level


def load_corpus(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    return torch.tensor(list(data), dtype=torch.long)


def get_batch(data: torch.Tensor, block: int, bsz: int, device: str):
    ix = torch.randint(len(data) - block - 1, (bsz,))
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + block + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def est_loss(model, data, block, bsz, device, iters=20) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, block, bsz, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("ckpt"))
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=250)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_corpus(args.corpus)
    n_val = max(1, int(0.05 * len(data)))
    train_d, val_d = data[:-n_val], data[-n_val:]

    cfg = GPTConfig(vocab_size=VOCAB, block_size=args.block_size)
    model = VerilogGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_iters)

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "verilog_gpt.pt"
    start = 0
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start = state["iter"] + 1
        print(f"resumed from iter {start}")

    print(f"device={device} params={model.num_params()/1e6:.1f}M "
          f"corpus={len(data)/1e6:.2f}M bytes")

    block = min(args.block_size, len(train_d) - 2)
    t0 = time.time()
    for it in range(start, args.max_iters):
        x, y = get_batch(train_d, block, args.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % args.eval_every == 0 or it == args.max_iters - 1:
            vl = est_loss(model, val_d, block, args.batch_size, device)
            print(f"iter {it:5d} | train {loss.item():.3f} | val {vl:.3f} "
                  f"| {time.time()-t0:.0f}s")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "iter": it,
                        "cfg": cfg.__dict__}, ckpt_path)

    seed = torch.tensor([[ord(c) for c in "module "]], device=device)
    out = model.generate(seed, max_new_tokens=200)
    print("--- sample ---")
    print(bytes(out[0].tolist()).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
