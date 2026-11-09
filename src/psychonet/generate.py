"""Build the train/val/test/ood splits.

Each example stores its log-mel features, the MOSQITO reference loudness and
sharpness, the signal family and the presentation level.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from psychonet.config import load_config
from psychonet.features import LogMel
from psychonet.references import reference_metrics
from psychonet.signals import generate

_FRONTEND: LogMel | None = None
_CFG: dict | None = None


def _init_worker(cfg: dict) -> None:
    global _FRONTEND, _CFG
    torch.set_num_threads(1)
    _CFG = cfg
    _FRONTEND = LogMel(fs=cfg["audio"]["fs"], **cfg["features"])


def _make_example(args: tuple[int, str]) -> tuple[np.ndarray, float, float, str, float]:
    seed, family = args
    assert _CFG is not None and _FRONTEND is not None
    rng = np.random.default_rng(seed)
    fs = _CFG["audio"]["fs"]
    level = float(rng.uniform(*_CFG["audio"]["level_range_db"]))
    x = generate(family, level, rng, fs, _CFG["audio"]["duration_s"])
    loudness, sharpness = reference_metrics(x, fs)
    with torch.no_grad():
        feats = _FRONTEND(torch.from_numpy(x).float()[None])[0].numpy()
    return feats.astype(np.float16), loudness, sharpness, family, level


def build_split(
    name: str, n: int, families: list[str], cfg: dict, base_seed: int, workers: int
) -> Path:
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 2**31 - 1, size=n)
    chosen = rng.choice(families, size=n)
    jobs = list(zip(seeds.tolist(), chosen.tolist(), strict=True))

    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        rows = list(pool.map(_make_example, jobs, chunksize=32))

    out = Path(cfg["data_dir"]) / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=np.stack([r[0] for r in rows]),
        loudness=np.array([r[1] for r in rows], dtype=np.float32),
        sharpness=np.array([r[2] for r in rows], dtype=np.float32),
        family=np.array([r[3] for r in rows]),
        level=np.array([r[4] for r in rows], dtype=np.float32),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)

    in_dist = cfg["families"]["in_distribution"]
    plan = [
        ("train", cfg["splits"]["train"], in_dist),
        ("val", cfg["splits"]["val"], in_dist),
        ("test", cfg["splits"]["test"], in_dist),
        ("ood", cfg["splits"]["ood"], cfg["families"]["out_of_distribution"]),
    ]
    for offset, (name, n, families) in enumerate(plan):
        start = time.perf_counter()
        path = build_split(name, n, families, cfg, cfg["seed"] + offset, args.workers)
        print(f"{name:>5}: {n} examples -> {path} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
