"""Evaluate MetricNet against MOSQITO references and two linear baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from psychonet.config import load_config
from psychonet.data import TARGETS, Split
from psychonet.inference import MetricEstimator
from psychonet.references import reference_metrics
from psychonet.signals import generate

UNITS = {"loudness_level_phon": "phon", "sharpness_acum": "acum"}


def band_levels(split: Split) -> np.ndarray:
    return split.features.astype(np.float64).mean(axis=2)


def overall_level(split: Split) -> np.ndarray:
    power = (10.0 ** (split.features.astype(np.float64) / 10.0)).sum(axis=1).mean(axis=1)
    return 10.0 * np.log10(power)


def fit_linear(x: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    xb = np.hstack([x, np.ones((len(x), 1))])
    reg = ridge * np.eye(xb.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def apply_linear(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((len(x), 1))]) @ w


def scores(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    err = pred - ref
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "p95_abs_error": float(np.percentile(np.abs(err), 95)),
        "pearson_r": float(np.corrcoef(pred, ref)[0, 1]),
    }


def predict_split(est: MetricEstimator, split: Split) -> np.ndarray:
    feats = torch.from_numpy(split.features.astype(np.float32))
    return np.concatenate(
        [est.predict_features(feats[i : i + 512]) for i in range(0, len(feats), 512)]
    )


def runtime_benchmark(est: MetricEstimator, cfg: dict, n: int = 64) -> dict[str, float]:
    rng = np.random.default_rng(0)
    fs, dur = cfg["audio"]["fs"], cfg["audio"]["duration_s"]
    signals = np.stack([generate("colored_noise", 70.0, rng, fs, dur) for _ in range(n)])
    start = time.perf_counter()
    for x in signals:
        reference_metrics(x, fs)
    t_ref = (time.perf_counter() - start) / n
    est.predict(signals[:2])
    start = time.perf_counter()
    est.predict(signals)
    t_net = (time.perf_counter() - start) / n
    return {"mosqito_ms_per_clip": 1e3 * t_ref, "metricnet_ms_per_clip": 1e3 * t_net}


def plot_scatter(results: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (key, label) in zip(
        axes,
        [("loudness_level_phon", "Loudness level [phon]"), ("sharpness_acum", "Sharpness [acum]")],
        strict=True,
    ):
        for split_name, color in (("test", "tab:blue"), ("ood", "tab:orange")):
            ref = results[split_name]["reference"][key]
            pred = results[split_name]["prediction"][key]
            ax.scatter(ref, pred, s=4, alpha=0.4, color=color, label=split_name)
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"MOSQITO reference: {label}")
        ax.set_ylabel(f"MetricNet: {label}")
        ax.legend(markerscale=4)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_history(history: list[dict], out: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Smooth L1 loss (standardised targets)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def markdown_report(report: dict) -> str:
    lines = [
        "| Split | Model | Loudness MAE [phon] | Loudness r | Sharpness MAE [acum] | Sharpness r |",
        "|---|---|---|---|---|---|",
    ]
    for split_name in ("test", "ood"):
        for model_name, s in report["overall"][split_name].items():
            lo, sh = s["loudness_level_phon"], s["sharpness_acum"]
            lines.append(
                f"| {split_name} | {model_name} | {lo['mae']:.2f} | {lo['pearson_r']:.3f} "
                f"| {sh['mae']:.3f} | {sh['pearson_r']:.3f} |"
            )
    lines += [
        "",
        "| Family | n | Loudness MAE [phon] | Sharpness MAE [acum] |",
        "|---|---|---|---|",
    ]
    for family, s in report["per_family"].items():
        lines.append(
            f"| {family} | {s['n']} | {s['loudness_level_phon']['mae']:.2f} "
            f"| {s['sharpness_acum']['mae']:.3f} |"
        )
    rt = report["runtime"]
    lines += [
        "",
        f"Runtime on CPU: MOSQITO {rt['mosqito_ms_per_clip']:.1f} ms/clip, "
        f"MetricNet {rt['metricnet_ms_per_clip']:.2f} ms/clip (batched, including front end).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/metricnet.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_dir, results_dir = Path(cfg["data_dir"]), Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    est = MetricEstimator(args.checkpoint)
    train = Split.load(data_dir / "train.npz")
    y_train = train.targets()
    w_level = fit_linear(overall_level(train)[:, None], y_train)
    w_bands = fit_linear(band_levels(train), y_train)

    report: dict = {"overall": {}, "per_family": {}}
    curves: dict = {}
    for split_name in ("test", "ood"):
        split = Split.load(data_dir / f"{split_name}.npz")
        ref = split.targets()
        preds = {
            "MetricNet": predict_split(est, split),
            "Linear on band levels": apply_linear(w_bands, band_levels(split)),
            "Linear on overall SPL": apply_linear(w_level, overall_level(split)[:, None]),
        }
        report["overall"][split_name] = {
            name: {t: scores(p[:, i], ref[:, i]) for i, t in enumerate(TARGETS)}
            for name, p in preds.items()
        }
        curves[split_name] = {
            "reference": {t: ref[:, i] for i, t in enumerate(TARGETS)},
            "prediction": {t: preds["MetricNet"][:, i] for i, t in enumerate(TARGETS)},
        }
        for family in sorted(set(split.family.tolist())):
            mask = split.family == family
            p = preds["MetricNet"][mask]
            report["per_family"][family] = {"n": int(mask.sum())} | {
                t: scores(p[:, i], ref[mask, i]) for i, t in enumerate(TARGETS)
            }

    report["runtime"] = runtime_benchmark(est, cfg)

    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    (results_dir / "metrics.md").write_text(markdown_report(report), encoding="utf-8")
    plot_scatter(curves, results_dir / "scatter.png")
    history_path = results_dir / "history.json"
    if history_path.exists():
        plot_history(
            json.loads(history_path.read_text(encoding="utf-8")), results_dir / "history.png"
        )
    print(markdown_report(report))


if __name__ == "__main__":
    main()
