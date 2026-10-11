#!/usr/bin/env python3
"""Verification script: eta→z conversion with real WRF data.

Reads switzerland_d01 wrfinput_d01 and tests vertical interpolation at domain
center for target heights [10, 50, 100, 200, 500]m above local terrain.

Outputs:
  - Console diagnostics: η indices, weights, interpolated vs native values
  - PNG figure: u-profile comparison (WRF native vs interpolated)

Usage:
    python scripts/check_eta_z_conversion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpuwrf.wrf_lbm_coupling.vertical_interp import (
    WRFColumnData,
    compute_z_mass,
    compute_z_w,
    interp_wrf_to_lbm_nodes,
)

G = 9.81


def destagger_u(u_stag: np.ndarray) -> np.ndarray:
    """Average U from x-staggered to mass points: (nz, ny, nx+1) → (nz, ny, nx)."""
    return 0.5 * (u_stag[:, :, :-1] + u_stag[:, :, 1:])


def destagger_v(v_stag: np.ndarray) -> np.ndarray:
    """Average V from y-staggered to mass points: (nz, ny+1, nx) → (nz, ny, nx)."""
    return 0.5 * (v_stag[:, :-1, :] + v_stag[:, 1:, :])


def main():
    wrfinput_path = PROJECT_ROOT / "examples" / "switzerland_d01" / "wrfinput_d01"
    if not wrfinput_path.exists():
        print(f"ERROR: {wrfinput_path} not found")
        sys.exit(1)

    print(f"Reading {wrfinput_path}")
    with Dataset(wrfinput_path, "r") as nc:
        ph = np.array(nc.variables["PH"][0])  # (45, 42, 42)
        phb = np.array(nc.variables["PHB"][0])  # (45, 42, 42)
        u_stag = np.array(nc.variables["U"][0])  # (44, 42, 43)
        v_stag = np.array(nc.variables["V"][0])  # (44, 43, 42)
        w = np.array(nc.variables["W"][0])  # (45, 42, 42)
        t_pert = np.array(nc.variables["T"][0])  # (44, 42, 42) - perturbation
        hgt = np.array(nc.variables["HGT"][0])  # (42, 42) - terrain MSL
        dx = float(nc.DX)
        dy = float(nc.DY)

    nz, ny, nx = t_pert.shape
    print(f"Grid: nz={nz}, ny={ny}, nx={nx}, dx={dx:.1f}m, dy={dy:.1f}m")

    # Destagger U/V to mass points
    u_mass = destagger_u(u_stag)
    v_mass = destagger_v(v_stag)

    # Compute total geopotential and heights
    ph_total = phb + ph
    z_w = compute_z_w(ph_total)
    z_mass = compute_z_mass(z_w)

    # Domain center (j, i)
    jc = ny // 2
    ic = nx // 2
    terrain_height_center = hgt[jc, ic]
    print(f"\nDomain center: j={jc}, i={ic}, terrain={terrain_height_center:.1f}m MSL")

    # Target heights above local terrain
    target_heights_agl = np.array([10.0, 50.0, 100.0, 200.0, 500.0])
    target_z_msl = target_heights_agl + terrain_height_center

    print("\nTarget heights:")
    for h_agl, h_msl in zip(target_heights_agl, target_z_msl):
        print(f"  {h_agl:6.1f}m AGL → {h_msl:7.1f}m MSL")

    # Extract column at domain center
    u_col = u_mass[:, jc, ic]
    v_col = v_mass[:, jc, ic]
    w_col = w[:, jc, ic]
    theta_col = t_pert[:, jc, ic]
    z_mass_col = z_mass[:, jc, ic]

    print("\nWRF native η-levels at domain center:")
    print("  k |    z_mass (m) |      u (m/s) |      v (m/s) |    theta (K)")
    print("  " + "-" * 68)
    for k in range(min(10, nz)):  # Print first 10 levels
        print(
            f" {k:2d} | {z_mass_col[k]:13.2f} | {u_col[k]:12.4f} | "
            f"{v_col[k]:12.4f} | {theta_col[k]:11.4f}"
        )
    if nz > 10:
        print(f"  ... ({nz - 10} more levels)")

    # Build WRFColumnData (single column, but keep 2D horizontal dims for compatibility)
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

    # Target coordinates: (x, y) at domain center in WRF grid frame (meters from SW corner)
    x_center = ic * dx
    y_center = jc * dy
    target_x = np.full_like(target_z_msl, x_center)
    target_y = np.full_like(target_z_msl, y_center)

    # Run interpolation
    print("\nRunning eta→z interpolation...")
    result = interp_wrf_to_lbm_nodes(wrf_data, target_z_msl, target_x, target_y)

    # Diagnostics: find bracketing levels and weights
    print("\nInterpolation diagnostics:")
    print("  Target z (m) | k_lower | k_upper | weight_upper |   u_interp (m/s)")
    print("  " + "-" * 75)
    for i, z_target in enumerate(target_z_msl):
        # Find bracketing levels manually for diagnostics
        k_upper = np.searchsorted(z_mass_col, z_target, side="right")
        k_lower = max(0, k_upper - 1)
        k_upper = min(nz - 1, k_upper)

        if k_lower == k_upper:
            weight_upper = 0.0  # clamped to boundary
        else:
            z_lower = z_mass_col[k_lower]
            z_upper = z_mass_col[k_upper]
            weight_upper = (z_target - z_lower) / (z_upper - z_lower)

        print(
            f"  {z_target:12.2f} | {k_lower:7d} | {k_upper:7d} | "
            f"{weight_upper:12.6f} | {result.u[i]:16.8f}"
        )

    # Sanity checks
    print("\nSanity checks:")
    u_max = np.max(np.abs(result.u))
    v_max = np.max(np.abs(result.v))
    w_max = np.max(np.abs(result.w))
    theta_max = np.max(np.abs(result.theta))
    print(f"  max |u|     = {u_max:.4f} m/s")
    print(f"  max |v|     = {v_max:.4f} m/s")
    print(f"  max |w|     = {w_max:.4f} m/s")
    print(f"  max |theta| = {theta_max:.4f} K")

    anomalies = []
    if u_max > 100.0:
        anomalies.append(f"ANOMALY: |u| > 100 m/s")
    if v_max > 100.0:
        anomalies.append(f"ANOMALY: |v| > 100 m/s")
    if w_max > 50.0:
        anomalies.append(f"ANOMALY: |w| > 50 m/s")
    if theta_max > 50.0:
        anomalies.append(f"ANOMALY: |theta| > 50 K")

    if anomalies:
        print("  " + "\n  ".join(anomalies))
    else:
        print("  All interpolated values within expected ranges.")

    # Generate figure: u-profile comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left panel: WRF native u-profile
    ax1.plot(u_col, z_mass_col, "b-", linewidth=1.5, label="WRF native η-levels")
    ax1.set_xlabel("u (m/s)", fontsize=12)
    ax1.set_ylabel("Height MSL (m)", fontsize=12)
    ax1.set_title(f"WRF u-profile at domain center\nj={jc}, i={ic}", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Right panel: Interpolated u-values
    ax2.plot(u_col, z_mass_col, "b-", linewidth=1.5, alpha=0.3, label="WRF native")
    ax2.scatter(
        result.u,
        target_z_msl,
        c="red",
        s=80,
        zorder=5,
        edgecolors="black",
        linewidth=1.2,
        label="Interpolated",
    )
    ax2.set_xlabel("u (m/s)", fontsize=12)
    ax2.set_ylabel("Height MSL (m)", fontsize=12)
    ax2.set_title("Interpolated u at target heights", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Match y-axis limits
    ymax = max(target_z_msl[-1] * 1.1, z_mass_col[-1] * 0.5)
    ax1.set_ylim(terrain_height_center - 50, ymax)
    ax2.set_ylim(terrain_height_center - 50, ymax)

    plt.tight_layout()
    output_path = PROJECT_ROOT / "proofs" / "eta_z_conversion_check.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nFigure saved: {output_path}")
    print("Verification complete. Visual inspection: check if red scatter points align with blue curve.")


if __name__ == "__main__":
    main()
