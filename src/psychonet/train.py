"""Train MetricNet on the generated splits."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from psychonet.config import load_config
from psychonet.data import MetricDataset, Split, Standardizer
from psychonet.model import MetricNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            loss = loss_fn(model(x), y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total += loss.item() * len(y)
            count += len(y)
    return total / count


def build_model(cfg: dict) -> MetricNet:
    m = cfg["model"]
    return MetricNet(
        n_mels=cfg["features"]["n_mels"],
        channels=tuple(m["channels"]),
        hidden=m["hidden"],
        dropout=m["dropout"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/metricnet.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    data_dir = Path(cfg["data_dir"])
    train_split = Split.load(data_dir / "train.npz")
    val_split = Split.load(data_dir / "val.npz")
    standardizer = Standardizer.fit(train_split)

    t = cfg["training"]
    train_loader = DataLoader(
        MetricDataset(train_split, standardizer), batch_size=t["batch_size"], shuffle=True
    )
    val_loader = DataLoader(MetricDataset(val_split, standardizer), batch_size=256)

    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=t["lr"], total_steps=t["epochs"] * len(train_loader)
    )
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"train={len(train_split)} val={len(val_split)} params={n_params:,}")

    history, best, stale = [], float("inf"), 0
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, t["epochs"] + 1):
        start = time.perf_counter()
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, scheduler)
        val_loss = run_epoch(model, val_loader, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"({time.perf_counter() - start:.0f} s)"
        )
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "standardizer": standardizer.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                },
                ckpt_path,
            )
        else:
            stale += 1
            if stale >= t["patience"]:
                print(f"early stop at epoch {epoch}")
                break

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
