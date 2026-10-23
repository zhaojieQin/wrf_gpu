#!/usr/bin/env python
"""Scan wrfout_d0* files for non-finite or huge values (diagnostic only).

For each file prints the max |value| of U, V, W, T, QNICE and lists every 3-D+ float
variable that has NaN/Inf or |value| > --limit, with the index of the worst point.

Usage:
  python scripts/check_wrfout_finite.py --dir ./runs/swift100m_coupled_mpi
"""
import argparse
import glob
import os

import numpy as np
import netCDF4 as nc

WATCH = ("U", "V", "W", "T", "QNICE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./runs/swift100m_coupled_mpi")
    ap.add_argument("--limit", type=float, default=1e10)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "wrfout_d0*")))
    if not files:
        raise SystemExit(f"no wrfout_d0* in {args.dir}")
    for f in files:
        with nc.Dataset(f) as ds:
            watch, bad = [], []
            for name, var in ds.variables.items():
                if var.ndim < 3 or var.dtype.kind != "f":
                    continue
                a = np.ma.filled(var[:], np.nan).astype(np.float64)
                finite = np.isfinite(a)
                m = float(np.max(np.abs(a[finite]))) if finite.any() else np.nan
                if name in WATCH:
                    watch.append(f"{name}={m:.3g}")
                if not finite.all() or m > args.limit:
                    worst = np.where(finite, np.abs(a), np.inf)
                    idx = np.unravel_index(int(np.argmax(worst)), a.shape)
                    bad.append(f"{name}(max={m:.3g}, nonfinite={int((~finite).sum())}, at {idx})")
            print(os.path.basename(f), " ".join(watch))
            print("    BAD:", "; ".join(bad) if bad else "none")


if __name__ == "__main__":
    main()
