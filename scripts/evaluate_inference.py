#!/usr/bin/env python3
"""Compute configured ADDSE metrics from saved inference archives."""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import polars as pl
from hydra.utils import instantiate
from tqdm import tqdm

from addse.app.eval import compute_metrics, update_db
from addse.metrics import BaseMetric
from addse.utils import load_hydra_config


def main() -> None:
    """Score every inference archive and write per-example and aggregate results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML file whose eval.metrics field is used")
    parser.add_argument("inference_dir", type=Path)
    parser.add_argument("output_db", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg, _ = load_hydra_config(args.config)
    metrics: dict[str, BaseMetric] = instantiate(cfg.eval.metrics)
    paths = sorted(args.inference_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No inference archives found in {args.inference_dir}")

    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.output_db)
    db.execute("""
        CREATE TABLE IF NOT EXISTS results (
            dset TEXT NOT NULL,
            idx INT NOT NULL,
            metric TEXT NOT NULL,
            name TEXT NOT NULL,
            value REAL,
            PRIMARY KEY (dset, idx, metric, name)
        )
    """)

    seen_dsets: set[str] = set()
    seen_names: set[str] = set()
    for path in tqdm(paths, desc="Metrics"):
        with np.load(path, allow_pickle=False) as archive:
            estimate = archive["estimate"]
            target = archive["target"]
            dset = str(archive["dset"].item())
            idx = int(archive["idx"].item())
            name = str(archive["name"].item())
        seen_dsets.add(dset)
        seen_names.add(name)
        if not args.overwrite:
            count = db.execute(
                "SELECT COUNT(*) FROM results WHERE dset = ? AND idx = ? AND name = ?",
                (dset, idx, name),
            ).fetchone()[0]
            if count == len(metrics):
                continue
        update_db(compute_metrics(estimate, target, metrics), dset, idx, name, db)
        db.commit()

    query = """
        SELECT dset, metric, name, ROUND(AVG(value), 2) AS mean_value
        FROM results
        GROUP BY dset, metric, name
        ORDER BY dset, metric, name
    """
    print(pl.read_database(query, db))
    db.close()


if __name__ == "__main__":
    main()
