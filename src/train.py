"""Train a waste classifier and evaluate it on the held-out test split.

    python src/train.py --arch convnext_tiny
"""
import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from timm.utils import ModelEmaV3
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from models import build_model
from prepare_data import CLASSES

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class WasteDataset(Dataset):
    def __init__(self, csv_path, root, tf):
        with open(csv_path) as f:
            self.rows = list(csv.DictReader(f))
        self.root, self.tf = root, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(os.path.join(self.root, r["path"])) as im:
            x = self.tf(im.convert("RGB"))
        return x, CLASSES.index(r["label"])


def transforms(size):
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(0.2),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(0.25),
    ])
    evaluate = T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    return train, evaluate


@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x).float()
            if tta:
                logits = logits + model(torch.flip(x, dims=[3])).float()
        probs.append(logits.softmax(1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def keep_awake():
    """Stop Windows from sleeping while this process runs (sleep kills the CUDA context)."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def plot_history(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(hist) + 1)
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in hist], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend()
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=20)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="convnext_tiny", choices=["convnext_tiny", "efficientnet_b2"])
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="continue from models/<arch>_last.pth")
    args = ap.parse_args()

    keep_awake()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    out_dir = os.path.join("results", args.arch)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    tf_train, tf_eval = transforms(args.size)
    sets = {s: WasteDataset(os.path.join(args.splits, f"{s}.csv"), args.root, tf_train if s == "train" else tf_eval)
            for s in ("train", "val", "test")}
    loaders = {s: DataLoader(d, batch_size=args.batch if s == "train" else args.batch * 2, shuffle=s == "train",
                             num_workers=args.workers, pin_memory=True, persistent_workers=True, drop_last=s == "train")
               for s, d in sets.items()}

    # inverse-frequency class weights to counter the organic-heavy imbalance
    counts = Counter(r["label"] for r in sets["train"].rows)
    weights = torch.tensor([len(sets["train"]) / (len(CLASSES) * counts[c]) for c in CLASSES], device=device)
    print("train counts:", dict(counts), "class weights:", weights.tolist())

    model = build_model(args.arch, len(CLASSES)).to(device).to(memory_format=torch.channels_last)
    ema = ModelEmaV3(model, decay=0.999, use_warmup=True)
    print(f"{args.arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params on {device}")

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(loaders["train"])
    warmup = len(loaders["train"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    ckpt_path = os.path.join("models", f"{args.arch}.pt")
    last_path = os.path.join("models", f"{args.arch}_last.pth")
    best_f1, hist, start = -1, [], 1
    if args.resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.module.load_state_dict(state["ema"])
        if "opt" in state:  # absent when rebuilt from a best-only checkpoint
            opt.load_state_dict(state["opt"])
            scaler.load_state_dict(state["scaler"])
        sched.last_epoch = state["epoch"] * len(loaders["train"]) - 1
        sched.step()
        best_f1, hist, start = state["best_f1"], state["hist"], state["epoch"] + 1
        print(f"resumed from epoch {state['epoch']} (best val f1 {best_f1:.4f})")

    for epoch in range(start, args.epochs + 1):
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for i, (x, y) in enumerate(loaders["train"]):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model, step=(epoch - 1) * len(loaders["train"]) + i)
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        probs, labels = predict(ema.module, loaders["val"], device)
        val_loss = nn.functional.nll_loss(torch.log(torch.from_numpy(probs) + 1e-8), torch.from_numpy(labels)).item()
        val_acc = (probs.argmax(1) == labels).mean()
        val_f1 = f1_score(labels, probs.argmax(1), average="macro")
        hist.append({"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen,
                     "val_loss": val_loss, "val_acc": float(val_acc), "val_f1": float(val_f1)})
        flag = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"arch": args.arch, "classes": CLASSES, "img_size": args.size, "mean": MEAN, "std": STD,
                        "epoch": epoch, "val_acc": float(val_acc), "val_f1": float(val_f1),
                        "state_dict": {k: v.half() if v.is_floating_point() else v
                                       for k, v in ema.module.state_dict().items()}}, ckpt_path)
            flag = "  * saved"
        print(f"epoch {epoch:2d}/{args.epochs}  train loss {loss_sum / seen:.4f} acc {correct / seen:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f} | {time.time() - t0:.0f}s{flag}", flush=True)
        torch.save({"epoch": epoch, "model": model.state_dict(), "ema": ema.module.state_dict(),
                    "opt": opt.state_dict(), "scaler": scaler.state_dict(), "best_f1": best_f1, "hist": hist}, last_path)

    plot_history(hist, os.path.join(out_dir, "history.png"))

    # final evaluation of the best checkpoint
    from models import load_checkpoint
    best, ckpt = load_checkpoint(ckpt_path, device)
    best = best.to(memory_format=torch.channels_last)
    report = {"arch": args.arch, "best_epoch": ckpt["epoch"], "params_M": sum(p.numel() for p in best.parameters()) / 1e6,
              "args": vars(args), "history": hist}
    for split in ("val", "test"):
        probs, labels = predict(best, loaders[split], device, tta=True)
        pred = probs.argmax(1)
        cm = confusion_matrix(labels, pred)
        plot_confusion(cm, os.path.join(out_dir, f"confusion_{split}.png"), f"{args.arch} - {split}")
        per_sub = defaultdict(lambda: [0, 0])
        for r, p in zip(sets[split].rows, pred):
            per_sub[r["subclass"]][0] += int(CLASSES[p] == r["label"])
            per_sub[r["subclass"]][1] += 1
        report[split] = {
            "accuracy": float((pred == labels).mean()),
            "macro_f1": float(f1_score(labels, pred, average="macro")),
            "confusion_matrix": cm.tolist(),
            "per_class": classification_report(labels, pred, target_names=CLASSES, output_dict=True, digits=4),
            "per_subclass_accuracy": {k: round(c / n, 4) for k, (c, n) in sorted(per_sub.items())},
        }
        print(f"\n== {split.upper()} (TTA) acc {report[split]['accuracy']:.4f}  macro-F1 {report[split]['macro_f1']:.4f}")
        print(classification_report(labels, pred, target_names=CLASSES, digits=4))
        print("per-subclass accuracy:", report[split]["per_subclass_accuracy"])
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
