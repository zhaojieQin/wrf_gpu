#!/usr/bin/env python
"""Plot the VirtualFluids x–z plane probe (xzPlaneMidY_bin_ID_0_t_<n>.vtk.pvtu series) to PNG.

Probe arrays (VF Probe, Statistic.Instantaneous): vx, vy, vz [m/s], rho, phi [K, θ − θ_ref]
(phi only when the temperature field is on). One PNG per file and variable; optional GIF.

Examples
--------
  python plot.py                                         # all files in the default run dir, vx vz phi
  python plot.py --dir runs/swift100m_coupled_mpi/lbm --vars vx phi --every 5
  python plot.py --vars vx --vmin -8 --vmax 2 --gif
  python plot.py --t0 3600 --dt 0.130594 --probe-every 15  # titles with model time

Reader: uses the `vtk` package if installed (pip install vtk), else `meshio` (pip install meshio).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DIR = "/home/zhaqi746/code/wrf_gpu/runs/swift100m_coupled_mpi/lbm"
UNITS = {"vx": "m/s", "vy": "m/s", "vz": "m/s", "rho": "-", "phi": "K (θ−θ_ref)"}
CMAPS = {"vx": "RdYlBu_r", "vy": "RdYlBu_r", "vz": "RdBu_r", "rho": "viridis", "phi": "RdYlBu_r"}


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _read_with_vtk(path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    out = reader.GetOutput()
    pts = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
    pd = out.GetPointData()
    data = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)).astype(np.float64)
            for i in range(pd.GetNumberOfArrays())}
    return pts, data


def _read_with_meshio(path):
    import meshio
    folder = os.path.dirname(path)
    pieces = [p.get("Source") for p in ET.parse(path).getroot().iter("Piece")]
    pts_all, data_all = [], {}
    for src in pieces:
        m = meshio.read(os.path.join(folder, src))
        pts_all.append(np.asarray(m.points, dtype=np.float64))
        for k, v in m.point_data.items():
            data_all.setdefault(k, []).append(np.asarray(v, dtype=np.float64).ravel())
    return np.concatenate(pts_all), {k: np.concatenate(v) for k, v in data_all.items()}


def read_pvtu(path):
    try:
        return _read_with_vtk(path)
    except ImportError:
        return _read_with_meshio(path)


def to_grid(pts, values, axis_h=0, axis_v=2):
    """Scattered plane nodes → regular 2-D array (nz, nx); missing nodes stay NaN."""
    h = np.round(pts[:, axis_h], 3)
    v = np.round(pts[:, axis_v], 3)
    hu, vu = np.unique(h), np.unique(v)
    grid = np.full((vu.size, hu.size), np.nan)
    grid[np.searchsorted(vu, v), np.searchsorted(hu, h)] = values
    return hu, vu, grid


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def file_index(path):
    m = re.search(r"_t_(\d+)\.vtk\.pvtu$", path)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR, help="folder with the probe files")
    ap.add_argument("--probe", default="xzPlaneMidY", help="probe name (file prefix)")
    ap.add_argument("--vars", nargs="+", default=["vx", "vz", "phi"], help="variables to plot")
    ap.add_argument("--out", default=None, help="output folder (default: <dir>/plots)")
    ap.add_argument("--every", type=int, default=1, help="plot every N-th file")
    ap.add_argument("--first", type=int, default=None, help="first file index to plot")
    ap.add_argument("--last", type=int, default=None, help="last file index to plot")
    ap.add_argument("--vmin", type=float, default=None, help="colour range min (all vars); default: per-var percentile")
    ap.add_argument("--vmax", type=float, default=None, help="colour range max (all vars)")
    ap.add_argument("--t0", type=float, default=None, help="model time of file index 0 (s), for titles")
    ap.add_argument("--dt", type=float, default=0.130594, help="LBM dt (s), for titles")
    ap.add_argument("--probe-every", type=int, default=15, help="LBM steps between probe files, for titles")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--gif", action="store_true", help="also write one animated GIF per variable (needs imageio)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, f"{args.probe}_bin_ID_*_t_*.vtk.pvtu")), key=file_index)
    if args.first is not None:
        files = [f for f in files if file_index(f) >= args.first]
    if args.last is not None:
        files = [f for f in files if file_index(f) <= args.last]
    files = files[:: max(1, args.every)]
    if not files:
        raise SystemExit(f"no {args.probe}_bin_ID_*_t_*.vtk.pvtu files in {args.dir}")
    out = args.out or os.path.join(args.dir, "plots")
    os.makedirs(out, exist_ok=True)
    print(f"{len(files)} files ({os.path.basename(files[0])} … {os.path.basename(files[-1])}) → {out}")

    # fixed colour range per variable from the first and last file (robust percentiles)
    ranges = {}
    for var in args.vars:
        if args.vmin is not None and args.vmax is not None:
            ranges[var] = (args.vmin, args.vmax)
            continue
        vals = []
        for f in {files[0], files[-1]}:
            _, data = read_pvtu(f)
            if var in data:
                vals.append(data[var])
        if not vals:
            print(f"  variable {var!r} not in files, skipped")
            continue
        lo, hi = np.nanpercentile(np.concatenate(vals), [1, 99])
        if var in ("vz",):                       # symmetric for vertical velocity
            m = max(abs(lo), abs(hi)); lo, hi = -m, m
        ranges[var] = (args.vmin if args.vmin is not None else lo, args.vmax if args.vmax is not None else hi)
    if not ranges:
        raise SystemExit("none of the requested variables found")

    pngs = {v: [] for v in ranges}
    for f in files:
        idx = file_index(f)
        pts, data = read_pvtu(f)
        y_plane = float(np.median(pts[:, 1]))
        for var, (lo, hi) in ranges.items():
            if var not in data:
                continue
            x, z, g = to_grid(pts, data[var])
            fig, ax = plt.subplots(figsize=(12, 3.2))
            im = ax.pcolormesh(x / 1000.0, z, g, cmap=CMAPS.get(var, "viridis"), vmin=lo, vmax=hi, shading="nearest")
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label(f"{var} [{UNITS.get(var, '')}]")
            ax.set_xlabel("x [km]")
            ax.set_ylabel("z [m]")
            title = f"{args.probe}  {var}  file t_{idx}  (y = {y_plane:.0f} m)"
            if args.t0 is not None:
                t = args.t0 + idx * args.probe_every * args.dt
                title += f"  t = {t:.1f} s (+{idx * args.probe_every * args.dt:.1f} s, step {idx * args.probe_every})"
            ax.set_title(title, fontsize=10)
            fig.tight_layout()
            png = os.path.join(out, f"{args.probe}_{var}_t_{idx:05d}.png")
            fig.savefig(png, dpi=args.dpi)
            plt.close(fig)
            pngs[var].append(png)
        print(f"  t_{idx}: " + ", ".join(f"{v} [{np.nanmin(data[v]):.3g}, {np.nanmax(data[v]):.3g}]"
                                       for v in ranges if v in data))

    if args.gif:
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("imageio not installed (pip install imageio); GIF skipped")
            return
        for var, frames in pngs.items():
            if frames:
                gif = os.path.join(out, f"{args.probe}_{var}.gif")
                imageio.mimsave(gif, [imageio.imread(p) for p in frames], duration=0.2)
                print(f"GIF: {gif}")


if __name__ == "__main__":
    main()
