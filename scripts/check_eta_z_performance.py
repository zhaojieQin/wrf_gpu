#!/usr/bin/env python3
"""Performance test: eta→z interpolation with 10000 random LBM nodes.

Tests the interpolation performance on a large set of random target points
to verify it can handle realistic LBM inlet boundary node counts.

Exits with error if total time exceeds 100ms.

Usage:
    python scripts/check_eta_z_performance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpuwrf.wrf_lbm_coupling.vertical_interp import (
    WRFColumnData,
    interp_wrf_to_lbm_nodes,
)


def destagger_u(u_stag: np.ndarray) -> np.ndarray:
    """Average U from x-staggered to mass points."""
    return 0.5 * (u_stag[:, :, :-1] + u_stag[:, :, 1:])


def destagger_v(v_stag: np.ndarray) -> np.ndarray:
    """Average V from y-staggered to mass points."""
    return 0.5 * (v_stag[:, :-1, :] + v_stag[:, 1:, :])


def main():
    wrfinput_path = PROJECT_ROOT / "examples" / "switzerland_d01" / "wrfinput_d01"
    if not wrfinput_path.exists():
        print(f"ERROR: {wrfinput_path} not found")
        sys.exit(1)

    print("Loading WRF data...")
    with Dataset(wrfinput_path, "r") as nc:
        ph = np.array(nc.variables["PH"][0])
        phb = np.array(nc.variables["PHB"][0])
        u_stag = np.array(nc.variables["U"][0])
        v_stag = np.array(nc.variables["V"][0])
        w = np.array(nc.variables["W"][0])
        t_pert = np.array(nc.variables["T"][0])
        hgt = np.array(nc.variables["HGT"][0])
        dx = float(nc.DX)
        dy = float(nc.DY)

    nz, ny, nx = t_pert.shape
    print(f"Grid: nz={nz}, ny={ny}, nx={nx}")

    u_mass = destagger_u(u_stag)
    v_mass = destagger_v(v_stag)
    ph_total = phb + ph

    # Build WRFColumnData
    wrf_data = WRFColumnData(
        ph_total=ph_total,
        u=u_mass,
        v=v_mass,
        w=w,
        theta=t_pert,
        terrain_height=hgt,
        dx_wrf=dx,
        dy_wrf=dy,
    )

    # Generate 10000 random LBM nodes within the WRF domain
    n_nodes = 10000
    rng = np.random.default_rng(42)

    # Horizontal: within grid bounds with small margin
    x_min, x_max = dx, (nx - 1) * dx - dx
    y_min, y_max = dy, (ny - 1) * dy - dy
    target_x = rng.uniform(x_min, x_max, n_nodes)
    target_y = rng.uniform(y_min, y_max, n_nodes)

    # Vertical: 10m to 1000m AGL (reasonable LBM domain height)
    # Add terrain height at approximate positions to get MSL heights
    # For simplicity, sample terrain at random grid points
    i_samples = rng.integers(0, nx, n_nodes)
    j_samples = rng.integers(0, ny, n_nodes)
    terrain_samples = hgt[j_samples, i_samples]
    height_agl = rng.uniform(10.0, 1000.0, n_nodes)
    target_z = terrain_samples + height_agl

    print(f"\nGenerated {n_nodes} random LBM nodes:")
    print(f"  x range: [{target_x.min():.1f}, {target_x.max():.1f}] m")
    print(f"  y range: [{target_y.min():.1f}, {target_y.max():.1f}] m")
    print(f"  z range: [{target_z.min():.1f}, {target_z.max():.1f}] m MSL")

    # Warm-up run (compile/cache initialization)
    print("\nWarm-up run...")
    _ = interp_wrf_to_lbm_nodes(
        wrf_data,
        target_z[:100],
        target_x[:100],
        target_y[:100],
    )

    # Timed run
    print(f"\nTiming interpolation for {n_nodes} nodes...")
    t_start = time.perf_counter()
    result = interp_wrf_to_lbm_nodes(wrf_data, target_z, target_x, target_y)
    t_elapsed = time.perf_counter() - t_start

    elapsed_ms = t_elapsed * 1000.0
    print(f"Elapsed time: {elapsed_ms:.2f} ms")
    print(f"Throughput: {n_nodes / t_elapsed:.0f} nodes/s")

    # Sanity checks
    print("\nOutput validation:")
    print(f"  u shape: {result.u.shape}, range: [{result.u.min():.2f}, {result.u.max():.2f}]")
    print(f"  v shape: {result.v.shape}, range: [{result.v.min():.2f}, {result.v.max():.2f}]")
    print(f"  w shape: {result.w.shape}, range: [{result.w.min():.2f}, {result.w.max():.2f}]")
    print(
        f"  theta shape: {result.theta.shape}, range: [{result.theta.min():.2f}, {result.theta.max():.2f}]"
    )

    # Performance gate
    threshold_ms = 100.0
    if elapsed_ms > threshold_ms:
        print(f"\n❌ PERFORMANCE GATE FAILED: {elapsed_ms:.2f}ms > {threshold_ms}ms")
        sys.exit(1)
    else:
        print(f"\n✓ Performance gate passed: {elapsed_ms:.2f}ms ≤ {threshold_ms}ms")


if __name__ == "__main__":
    main()
