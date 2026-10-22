"""Calibrate a trained model's confidences on the val split and store it in the checkpoint.

Class-weighted loss + label smoothing make raw softmax scores under-confident
(especially for the down-weighted organic class). Vector scaling fits one
temperature and a per-class bias on val logits:  p = softmax(logits / T + b).

    python src/calibrate.py models/convnext_tiny.pt
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import load_checkpoint
from train import WasteDataset, transforms


def ece(probs, labels, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return total


@torch.no_grad()
def logits_for(model, loader, device):
    out, ys = [], []
    for x, y in loader:
        out.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(out), torch.cat(ys)


def summary(name, logits, labels, classes):
    probs = logits.softmax(1).numpy()
    y = labels.numpy()
    pred = probs.argmax(1)
    mean_conf = {c: round(float(probs[(y == i) & (pred == i), i].mean()), 3) for i, c in enumerate(classes)}
    print(f"  {name:6s} acc {(pred == y).mean():.4f}  ECE {ece(probs, y):.4f}  mean conf when correct {mean_conf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    classes = ckpt["classes"]
    _, tf_eval = transforms(ckpt["img_size"])
    data = {}
    for split in ("val", "test"):
        ds = WasteDataset(os.path.join(args.splits, f"{split}.csv"), args.root, tf_eval)
        data[split] = logits_for(model, DataLoader(ds, batch_size=64, num_workers=4), device)

    log_t = torch.zeros(1, requires_grad=True)
    bias = torch.zeros(len(classes), requires_grad=True)
    opt = torch.optim.LBFGS([log_t, bias], lr=0.1, max_iter=200)
    val_logits, val_y = data["val"]

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(val_logits / log_t.exp() + bias, val_y)
        loss.backward()
        return loss

    opt.step(closure)
    T, b = log_t.exp().item(), bias.detach()
    print(f"{ckpt['arch']}: T={T:.3f} bias={[round(v, 3) for v in b.tolist()]}")
    for split, (lg, y) in data.items():
        print(split)
        summary("before", lg, y, classes)
        summary("after", lg / T + b, y, classes)

    ckpt["calibration"] = {"temperature": T, "bias": b.tolist()}
    torch.save(ckpt, args.checkpoint)
    print("saved calibration to", args.checkpoint)


if __name__ == "__main__":
    main()
