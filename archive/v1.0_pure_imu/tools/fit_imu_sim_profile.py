#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit a practical synthetic-IMU profile from a real static interval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.sync import load_imu

DATA = ROOT / "datasets"


def _block_walk(values: np.ndarray, rate: float, seconds: float = 1.0) -> np.ndarray:
    n = max(2, int(round(rate * seconds)))
    count = len(values) // n
    if count < 3:
        return np.zeros(values.shape[1])
    means = values[:count * n].reshape(count, n, -1).mean(axis=1)
    return np.std(np.diff(means, axis=0), axis=0, ddof=1) / np.sqrt(2.0)


def _covariance_floor(cov: np.ndarray, floor: np.ndarray) -> np.ndarray:
    cov = (cov + cov.T) * 0.5
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, float(np.min(floor) ** 2))
    cov = (vectors * values) @ vectors.T
    diagonal = np.maximum(np.diag(cov), floor ** 2)
    cov[np.diag_indices(3)] = diagonal
    return cov


def fit(session: str, static_seconds: float) -> dict:
    path = DATA / session / "imu_stream.csv"
    t, y = load_imu(path)
    if len(t) < 200:
        raise RuntimeError(f"too few IMU samples in {path}")
    end = t[0] + static_seconds
    m = t <= end
    t = t[m]
    y = y[m]
    dt = np.diff(t)
    rate = (len(t) - 1) / max(float(t[-1] - t[0]), 1e-9)
    acc = y[:, 0:3]
    gyro = y[:, 3:6]
    acc_quant = np.full(3, 1.0 / 2048.0)
    gyro_quant = np.full(3, 2000.0 / 32768.0)
    acc_white = np.maximum(
        np.std(np.diff(acc, axis=0), axis=0, ddof=1) / np.sqrt(2.0),
        acc_quant / np.sqrt(12.0),
    )
    gyro_white = np.maximum(
        np.std(np.diff(gyro, axis=0), axis=0, ddof=1) / np.sqrt(2.0),
        gyro_quant / np.sqrt(12.0),
    )
    acc_walk = np.maximum(_block_walk(acc, rate), acc_quant / 100.0)
    gyro_walk = np.maximum(_block_walk(gyro, rate), gyro_quant / 100.0)
    acc_white_cov = _covariance_floor(
        np.cov(np.diff(acc, axis=0), rowvar=False) / 2.0,
        acc_quant / np.sqrt(12.0),
    )
    gyro_white_cov = _covariance_floor(
        np.cov(np.diff(gyro, axis=0), rowvar=False) / 2.0,
        gyro_quant / np.sqrt(12.0),
    )
    block_n = max(2, int(round(rate)))
    block_count = len(acc) // block_n
    acc_blocks = acc[:block_count * block_n].reshape(block_count, block_n, 3).mean(1)
    gyro_blocks = gyro[:block_count * block_n].reshape(block_count, block_n, 3).mean(1)
    acc_walk_cov = _covariance_floor(
        np.cov(np.diff(acc_blocks, axis=0), rowvar=False) / 2.0,
        acc_quant / 100.0,
    )
    gyro_walk_cov = _covariance_floor(
        np.cov(np.diff(gyro_blocks, axis=0), rowvar=False) / 2.0,
        gyro_quant / 100.0,
    )
    profile = {
        "source_session": session,
        "static_seconds": static_seconds,
        "samples": int(len(t)),
        "sample_rate_hz": rate,
        "sample_dt_std_ms": float(np.std(dt, ddof=1) * 1000.0),
        "sample_dt_p95_ms": float(np.percentile(dt, 95) * 1000.0),
        "accelerometer": {
            "unit": "g",
            "static_mean": acc.mean(axis=0).tolist(),
            "white_noise_std_per_sample": acc_white.tolist(),
            "white_noise_covariance": acc_white_cov.tolist(),
            "bias_random_walk_std_per_1s": acc_walk.tolist(),
            "bias_random_walk_covariance_per_1s": acc_walk_cov.tolist(),
            "quantization": acc_quant.tolist(),
        },
        "gyroscope": {
            "unit": "deg/s",
            "static_bias": gyro.mean(axis=0).tolist(),
            "white_noise_std_per_sample": gyro_white.tolist(),
            "white_noise_covariance": gyro_white_cov.tolist(),
            "bias_random_walk_std_per_1s": gyro_walk.tolist(),
            "bias_random_walk_covariance_per_1s": gyro_walk_cov.tolist(),
            "quantization": gyro_quant.tolist(),
        },
        "temperature_c": {
            "mean": float(np.mean(y[:, 14])),
            "std": float(np.std(y[:, 14])),
        },
        "notes": [
            "White noise is estimated from first differences divided by sqrt(2).",
            "Random walk is the standard deviation of adjacent 1 s block means divided by sqrt(2).",
            "Accelerometer static mean is orientation-dependent and is not treated as a fixed bias.",
        ],
    }
    return profile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="rigid_20260924_142902")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA / "sim_assets" / "imu_profile_usb.json",
    )
    args = ap.parse_args()
    profile = fit(args.session, args.seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
