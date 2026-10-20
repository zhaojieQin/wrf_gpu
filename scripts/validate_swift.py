#!/usr/bin/env python
"""Diurnal-cycle validation of the WRF→LBM coupled SWiFT run.

Reproduces the analyses of 大气稳定度与风力机尾流日周期演变研究, §3.5 (2013-11-08 00:00 LST →
2013-11-09 00:00 LST, LST = UTC−6) from the samples written by lbm_adapter/validation.py:

  fig3_07_profiles.png       wind speed / direction / θ / TKE profiles at 12:00, 16:00, 24:00 LST
  fig3_13_theta_zt.png       θ height–time section (LBM planar mean, WRF footprint mean)
  fig3_14_ws70.png           70 m wind speed time series (1 Hz, 10-min mean; WRF)
  fig3_15_16_wd70.png        70 m wind rose (whole day) + direction histograms 12–14 / 21–23 LST
  fig3_18_errors70.png       70 m θ / speed / direction, 10…60 min averages (vs obs if given)
  fig3_20_ws_hourly.png      hourly wind profiles 20–120 m (mean, 25–75 % band; WRF mean)
  fig3_21_alpha.png          power-law shear exponent α(20 m → 120 m), hourly
  fig3_22_ti_hourly.png      hourly TI profiles, TI = σ(horizontal speed) / U(70 m)
  fig3_23_ti_rotor.png       rotor-mean (20–120 m) TI, hourly
  summary.csv                hourly 70 m values (LBM / WRF)

Inputs (RUN_DIR = --run-dir):
  RUN_DIR/validation/meta.json, RUN_DIR/validation/wrf_tower.nc
  RUN_DIR/lbm/valTower_timeseries*.txt, RUN_DIR/lbm/valPlanarZ_bin_ID_0_t_*.vtk.pvtu
  RUN_DIR/wrfout_d03_* (optional, WRF TKE from QKE)
  --obs CSV (optional): columns time_utc (ISO), height_m, and any of wspd, wdir, theta.

Usage:
  python scripts/validate_swift.py --run-dir ./runs/swift100m_coupled_mpi [--obs ttu_tower.csv]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HUB_Z = 70.0
ROTOR = (20.0, 120.0)
TYPICAL = (("12:00 LST (unstable)", 12.0), ("16:00 LST (near-neutral)", 16.0), ("24:00 LST (stable)", 24.0))


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def read_tower_timeseries(path: str):
    """VF Probe time series: text header + binary records [t, q1(p1..pN), q2(p1..pN), ...]."""
    with open(path, "rb") as f:
        raw = f.read()
    lines, pos = [], 0
    while len(lines) < 4:
        end = raw.index(b"\n", pos)
        lines.append(raw[pos:end].decode())
        pos = end + 1
    names = [q.strip() for q in lines[1].split(":", 1)[1].split(",") if q.strip()]
    npts = int(lines[3])
    end = raw.index(b"\n", pos)          # "Positions: x, y, z"
    pos = end + 1
    coords = []
    for _ in range(npts):
        end = raw.index(b"\n", pos)
        coords.append([float(c) for c in raw[pos:end].decode().split(",")])
        pos = end + 1
    body = raw[pos:]
    rec = 1 + len(names) * npts
    for dtype in (np.float32, np.float64):
        size = rec * np.dtype(dtype).itemsize
        if len(body) % size == 0 and len(body) > 0:
            data = np.frombuffer(body, dtype=dtype).reshape(-1, rec).astype(np.float64)
            if np.all(np.diff(data[:, 0]) >= -1e-6):
                break
    else:
        raise ValueError(f"{path}: cannot determine record precision")
    t = data[:, 0]
    q = data[:, 1:].reshape(-1, len(names), npts)
    coords = np.asarray(coords)
    order = np.argsort(coords[:, 2])
    out = {name: q[:, k, order] for k, name in enumerate(names)}
    keep = np.r_[True, np.diff(t) > 0]      # drop duplicated time stamps at flush boundaries
    return t[keep], coords[order, 2], {k: v[keep] for k, v in out.items()}


def read_pvtu(path: str):
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        reader = vtk.vtkXMLPUnstructuredGridReader()   # .pvtu (parallel collection)
        reader.SetFileName(path)
        reader.Update()
        out = reader.GetOutput()
        pts = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
        pd = out.GetPointData()
        return pts, {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)).astype(np.float64)
                     for i in range(pd.GetNumberOfArrays())}
    except ImportError:
        import meshio
        folder = os.path.dirname(path)
        pts_all, data_all = [], {}
        for piece in ET.parse(path).getroot().iter("Piece"):
            m = meshio.read(os.path.join(folder, piece.get("Source")))
            pts_all.append(np.asarray(m.points, dtype=np.float64))
            for k, v in m.point_data.items():
                data_all.setdefault(k, []).append(np.asarray(v, dtype=np.float64).ravel())
        return np.concatenate(pts_all), {k: np.concatenate(v) for k, v in data_all.items()}


def read_planar(lbm_dir: str, dt: float):
    files = glob.glob(os.path.join(lbm_dir, "valPlanarZ_bin_ID_0_t_*.vtk.pvtu"))
    if not files:
        return None
    steps = sorted((int(re.search(r"_t_(\d+)\.vtk\.pvtu$", f).group(1)), f) for f in files)
    times, prof, z = [], {}, None
    for step, f in steps:
        pts, data = read_pvtu(f)
        order = np.argsort(pts[:, 2])
        z = pts[order, 2]
        for k, v in data.items():
            prof.setdefault(k, []).append(v[order])
        times.append(step * dt)
    return np.asarray(times), z, {k: np.asarray(v) for k, v in prof.items()}


def read_wrf_tower(path: str):
    import netCDF4 as nc
    with nc.Dataset(path) as ds:
        out = {k: np.asarray(ds[k][:], dtype=np.float64) for k in ds.variables}
    return out


def read_wrfout_tke(run_dir: str, x_wrf: float, y_wrf: float, dx: float, start: datetime):
    """WRF d03 TKE profile (QKE/2) at the tower from hourly wrfout files; None if unavailable."""
    import netCDF4 as nc
    files = sorted(glob.glob(os.path.join(run_dir, "wrfout_d03_*")))
    res = []
    for f in files:
        try:
            with nc.Dataset(f) as ds:
                key = "QKE" if "QKE" in ds.variables else ("TKE_PBL" if "TKE_PBL" in ds.variables else None)
                if key is None:
                    return None
                i, j = int(round(x_wrf / dx)), int(round(y_wrf / dx))
                tke = np.asarray(ds[key][0, :, j, i], dtype=np.float64) * (0.5 if key == "QKE" else 1.0)
                ph = np.asarray(ds["PH"][0, :, j, i] + ds["PHB"][0, :, j, i], dtype=np.float64) / 9.81
                zm = 0.5 * (ph[1:] + ph[:-1]) - float(ds["HGT"][0, j, i])
                tstr = "".join(c.decode() if isinstance(c, bytes) else str(c) for c in ds["Times"][0])
                t = datetime.strptime(tstr, "%Y-%m-%d_%H:%M:%S")
                res.append(((t - start).total_seconds(), zm, tke))
        except Exception as exc:  # noqa: BLE001
            print(f"[validate] skip {f}: {exc}")
    return res or None


def read_obs(path: str | None, start: datetime):
    if not path:
        return None
    import csv
    rows = list(csv.DictReader(open(path)))
    out = {"t": [], "z": []}
    for r in rows:
        out["t"].append((datetime.fromisoformat(r["time_utc"].replace("Z", "")) - start).total_seconds())
        out["z"].append(float(r["height_m"]))
        for k in ("wspd", "wdir", "theta"):
            if k in r and r[k] not in ("", None):
                out.setdefault(k, []).append(float(r[k]))
            elif k in r:
                out.setdefault(k, []).append(np.nan)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def wdir(u, v):
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def circ_mean(deg, axis=None):
    r = np.radians(deg)
    return np.degrees(np.arctan2(np.nanmean(np.sin(r), axis=axis), np.nanmean(np.cos(r), axis=axis))) % 360.0


def at_height(z, field, zt):
    """Linear interpolation of field[..., z] to height zt."""
    k = np.searchsorted(z, zt)
    k = min(max(k, 1), len(z) - 1)
    w = (zt - z[k - 1]) / (z[k] - z[k - 1])
    return (1 - w) * field[..., k - 1] + w * field[..., k]


def window(t, t0, t1):
    return (t >= t0) & (t < t1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="./runs/swift100m_coupled_mpi")
    ap.add_argument("--obs", default=None, help="observation CSV (time_utc, height_m, wspd, wdir, theta)")
    ap.add_argument("--out", default=None, help="figure folder (default RUN_DIR/validation/figures)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    val_dir = os.path.join(args.run_dir, "validation")
    meta = json.load(open(os.path.join(val_dir, "meta.json")))
    out_dir = args.out or os.path.join(val_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    start = datetime.strptime(meta["wrf_start_date"], "%Y-%m-%d_%H:%M:%S")
    lst_off = float(meta.get("lst_offset_hours", -6.0))
    t_c0 = float(meta["coupling_start_s"])
    theta_ref = float(meta["theta_ref"])
    dt = float(meta["dt_lbm"])
    # hours since 00:00 LST of the coupling start day
    lst_start = start + timedelta(seconds=t_c0) + timedelta(hours=lst_off)
    lst_midnight = datetime(lst_start.year, lst_start.month, lst_start.day)
    to_lst_h = lambda t_s: ((start + timedelta(hours=lst_off) - lst_midnight).total_seconds() + t_s) / 3600.0  # noqa: E731
    print(f"[validate] WRF start {start} UTC, coupling from {lst_start} LST, θ_ref={theta_ref:.2f} K")

    # ---- LBM tower (1 Hz) ----
    lbm_dir = meta.get("lbm_output_dir") or os.path.join(args.run_dir, "lbm")
    if not os.path.isdir(lbm_dir):
        lbm_dir = os.path.join(args.run_dir, "lbm")
    tfiles = sorted(glob.glob(os.path.join(lbm_dir, "valTower_timeseries*.txt")))
    if not tfiles:
        raise SystemExit(f"no valTower_timeseries*.txt in {lbm_dir}")
    t_l, z_l, q = read_tower_timeseries(tfiles[0])
    h_l = to_lst_h(t_c0 + t_l)
    u_l, v_l, w_l = q["vx"], q["vy"], q["vz"]
    th_l = q["phi"] + theta_ref if "phi" in q else None
    s_l = np.hypot(u_l, v_l)
    print(f"[validate] LBM tower: {t_l.size} samples, {z_l.size} levels {z_l[0]:.1f}..{z_l[-1]:.1f} m, "
          f"{h_l[0]:.2f}..{h_l[-1]:.2f} h LST")

    # ---- WRF tower ----
    wt = read_wrf_tower(os.path.join(val_dir, "wrf_tower.nc"))
    h_w = to_lst_h(wt["time"])
    z_w = wt["z"]
    s_w = np.hypot(wt["u"], wt["v"])
    d_w = wdir(wt["u"], wt["v"])

    # ---- planar averages ----
    planar = read_planar(lbm_dir, dt)
    obs = read_obs(args.obs, start)
    if obs is not None:
        obs["h"] = to_lst_h(obs["t"])

    # ======================= Fig 3.13: θ height–time =======================
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    if planar is not None and "phi" in planar[2]:
        tp, zp, pp = planar
        th_p = pp["phi"] + theta_ref
        sel = zp <= 1500
        vmin, vmax = np.nanpercentile(th_p[:, sel], [1, 99])
        m = axs[0].pcolormesh(to_lst_h(t_c0 + tp), zp[sel] / 1000, th_p[:, sel].T, shading="auto",
                              cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        fig.colorbar(m, ax=axs[0], label="θ (K)")
    axs[0].set_title("LBM planar-mean θ")
    hf = to_lst_h(wt["time_fp"])
    sel = z_w <= 1500
    m = axs[1].pcolormesh(hf, z_w[sel] / 1000, wt["theta_fp"][:, sel].T, shading="auto", cmap="RdYlBu_r")
    fig.colorbar(m, ax=axs[1], label="θ (K)")
    axs[1].set_title("WRF d03 θ, mean over LBM footprint")
    for ax in axs:
        ax.set_ylabel("height (km)")
    axs[1].set_xlabel("time (LST, h)")
    fig.savefig(os.path.join(out_dir, "fig3_13_theta_zt.png"), dpi=args.dpi)
    plt.close(fig)

    # ======================= Fig 3.14: 70 m wind speed =======================
    s70_l = at_height(z_l, s_l, HUB_Z)
    s70_w = at_height(z_w, s_w, HUB_Z)
    edges = np.arange(np.floor(h_l[0] * 6) / 6, h_l[-1] + 1 / 6, 1 / 6)
    idx = np.digitize(h_l, edges)
    s70_10 = np.array([np.nanmean(s70_l[idx == k]) if np.any(idx == k) else np.nan for k in range(1, len(edges))])
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(h_l, s70_l, lw=0.3, color="0.6", label="LBM 1 Hz")
    ax.plot(0.5 * (edges[1:] + edges[:-1]), s70_10, lw=1.5, color="C0", label="LBM 10-min mean")
    ax.plot(h_w, s70_w, lw=1.0, color="C3", label="WRF d03")
    if obs is not None and "wspd" in obs:
        sel = np.abs(obs["z"] - HUB_Z) < 10
        ax.plot(obs["h"][sel], obs["wspd"][sel], ".", ms=2, color="k", label="obs")
    ax.set_xlabel("time (LST, h)"); ax.set_ylabel("wind speed at 70 m (m/s)"); ax.set_xlim(0, 24)
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out_dir, "fig3_14_ws70.png"), dpi=args.dpi)
    plt.close(fig)

    # ======================= Fig 3.15/3.16: 70 m direction =======================
    d70_l = wdir(at_height(z_l, u_l, HUB_Z), at_height(z_l, v_l, HUB_Z))
    fig = plt.figure(figsize=(12, 4.5), constrained_layout=True)
    axr = fig.add_subplot(1, 3, 1, projection="polar")
    bins = np.radians(np.arange(0, 361, 22.5))
    hist, _ = np.histogram(np.radians(d70_l), bins=bins)
    axr.bar(bins[:-1] + np.radians(11.25), hist / hist.sum(), width=np.radians(22.5), color="C0", alpha=0.8)
    axr.set_theta_zero_location("N"); axr.set_theta_direction(-1)
    axr.set_title("LBM 70 m wind rose, 0–24 LST")
    for n, (h0, h1) in enumerate(((12, 14), (21, 23))):
        ax = fig.add_subplot(1, 3, n + 2)
        sel = window(h_l, h0, h1)
        ax.hist(d70_l[sel], bins=np.arange(170, 260, 2.5), density=True, color="C0", alpha=0.7, label="LBM")
        selw = window(h_w, h0, h1)
        ax.hist(at_height(z_w, d_w, HUB_Z)[selw], bins=np.arange(170, 260, 2.5), density=True,
                histtype="step", color="C3", label="WRF")
        ax.set_title(f"{h0:02d}:00–{h1:02d}:00 LST"); ax.set_xlabel("direction (°)"); ax.legend()
    fig.savefig(os.path.join(out_dir, "fig3_15_16_wd70.png"), dpi=args.dpi)
    plt.close(fig)

    # ======================= hourly statistics =======================
    hours = np.arange(24)
    rotor = (z_l >= ROTOR[0]) & (z_l <= ROTOR[1])
    hourly = []
    for hh in hours:
        sel = window(h_l, hh, hh + 1)
        selw = window(h_w, hh, hh + 1)
        if sel.sum() < 10:
            hourly.append(None)
            continue
        mean_s = np.nanmean(s_l[sel], axis=0)
        q25, q75 = np.nanpercentile(s_l[sel], [25, 75], axis=0)
        sig = np.nanstd(s_l[sel], axis=0)
        u70 = at_height(z_l, mean_s, HUB_Z)
        ti = sig / u70
        u20, u120 = at_height(z_l, mean_s, ROTOR[0]), at_height(z_l, mean_s, ROTOR[1])
        alpha = np.log(u120 / u20) / np.log(ROTOR[1] / ROTOR[0])
        ws_w = np.nanmean(s_w[selw], axis=0) if selw.any() else np.full(z_w.size, np.nan)
        hourly.append(dict(hour=hh, mean=mean_s, q25=q25, q75=q75, ti=ti, u70=u70, alpha=alpha,
                           ti_rotor=float(np.nanmean(ti[rotor])), ws_w=ws_w,
                           alpha_w=float(np.log(at_height(z_w, ws_w, ROTOR[1]) / at_height(z_w, ws_w, ROTOR[0]))
                                         / np.log(ROTOR[1] / ROTOR[0])),
                           u70_w=float(at_height(z_w, ws_w, HUB_Z)),
                           th70=float(at_height(z_l, np.nanmean(th_l[sel], axis=0), HUB_Z)) if th_l is not None else np.nan,
                           th70_w=float(at_height(z_w, np.nanmean(wt["theta"][selw], axis=0), HUB_Z)) if selw.any() else np.nan,
                           d70=float(circ_mean(d70_l[sel]))))

    # Fig 3.20: hourly wind profiles 20–120 m
    fig, axs = plt.subplots(4, 6, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
    zr = z_l[rotor]
    zwr = (z_w >= ROTOR[0]) & (z_w <= ROTOR[1])
    for hh, ax in zip(hours, axs.ravel()):
        r = hourly[hh]
        if r is not None:
            ax.fill_betweenx(zr, r["q25"][rotor], r["q75"][rotor], color="C0", alpha=0.25)
            ax.plot(r["mean"][rotor], zr, color="C0", label="LBM")
            ax.plot(r["ws_w"][zwr], z_w[zwr], color="C3", ls="--", label="WRF")
        ax.set_title(f"{hh:02d}:00–{hh + 1:02d}:00 LST", fontsize=9)
        ax.set_xlim(0, 18); ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    fig.supxlabel("wind speed (m/s)"); fig.supylabel("height (m)")
    fig.savefig(os.path.join(out_dir, "fig3_20_ws_hourly.png"), dpi=args.dpi)
    plt.close(fig)

    # Fig 3.22: hourly TI profiles
    fig, axs = plt.subplots(4, 6, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
    for hh, ax in zip(hours, axs.ravel()):
        r = hourly[hh]
        if r is not None:
            ax.plot(r["ti"][rotor], zr, color="C0")
        ax.set_title(f"{hh:02d}:00–{hh + 1:02d}:00 LST", fontsize=9)
        ax.set_xlim(0, 0.5); ax.grid(alpha=0.3)
    fig.supxlabel("TI = σ(horizontal speed) / U(70 m)"); fig.supylabel("height (m)")
    fig.savefig(os.path.join(out_dir, "fig3_22_ti_hourly.png"), dpi=args.dpi)
    plt.close(fig)

    # Fig 3.21 / 3.23: α and rotor TI
    hc = np.array([r["hour"] + 0.5 for r in hourly if r is not None])
    for name, key, keyw, ylabel, yl in (
            ("fig3_21_alpha.png", "alpha", "alpha_w", "shear exponent α (20→120 m)", (0, 0.5)),
            ("fig3_23_ti_rotor.png", "ti_rotor", None, "rotor-mean TI (20–120 m)", (0, 0.35))):
        fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
        ax.plot(hc, [r[key] for r in hourly if r is not None], "o-", color="C0", label="LBM")
        if keyw:
            ax.plot(hc, [r[keyw] for r in hourly if r is not None], "s--", color="C3", label="WRF")
        ax.set_xlim(0, 24); ax.set_ylim(*yl); ax.set_xlabel("time (LST, h)"); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3); ax.legend()
        fig.savefig(os.path.join(out_dir, name), dpi=args.dpi)
        plt.close(fig)

    # ======================= Fig 3.7: profiles at typical times =======================
    tke_w = read_wrfout_tke(args.run_dir,
                            meta["tower_lbm"][0] + meta["lbm_origin_wrf"][0],
                            meta["tower_lbm"][1] + meta["lbm_origin_wrf"][1],
                            float(meta.get("dx_wrf", 100.0)), start)
    fig, axs = plt.subplots(3, 4, figsize=(15, 13), sharey=True, constrained_layout=True)
    for row, (label, hc0) in enumerate(TYPICAL):
        sel = window(h_l, hc0 - 1.0, hc0 + 1e-6)          # preceding hour (24:00 LST is the run end)
        selw = window(h_w, hc0 - 1.0, hc0 + 1e-6)
        if sel.sum() < 10:
            continue
        ul, vl, wl = u_l[sel], v_l[sel], w_l[sel]
        tke_l = 0.5 * (np.nanvar(ul, axis=0) + np.nanvar(vl, axis=0) + np.nanvar(wl, axis=0))
        prof_l = (np.nanmean(np.hypot(ul, vl), axis=0), circ_mean(wdir(ul, vl), axis=0),
                  np.nanmean(th_l[sel], axis=0) if th_l is not None else None, tke_l)
        prof_w = (np.nanmean(s_w[selw], axis=0), circ_mean(d_w[selw], axis=0),
                  np.nanmean(wt["theta"][selw], axis=0), None)
        for col, (pl, pw, xl) in enumerate(zip(prof_l, prof_w, ("wind speed (m/s)", "direction (°)", "θ (K)",
                                                                "TKE (m²/s²)"))):
            ax = axs[row, col]
            if pl is not None:
                ax.plot(pl, z_l, color="C0", label="LBM tower")
            if pw is not None:
                ax.plot(pw, z_w, color="C3", ls="--", label="WRF tower")
            if col == 3 and tke_w is not None:
                k = int(np.argmin([abs(to_lst_h(t) - hc0) for t, _, _ in tke_w]))
                ax.plot(tke_w[k][2], tke_w[k][1], color="C3", ls="--", label="WRF QKE/2")
            if obs is not None:
                key = ("wspd", "wdir", "theta", None)[col]
                if key and key in obs:
                    so = window(obs["h"], hc0 - 1.0, hc0 + 1e-6)
                    if so.any():
                        zs = np.unique(obs["z"][so])
                        mo = [np.nanmean(obs[key][so & (obs["z"] == zz)]) if key != "wdir"
                              else circ_mean(obs[key][so & (obs["z"] == zz)]) for zz in zs]
                        ax.plot(mo, zs, "ko", ms=4, label="obs")
            ax.set_xlabel(xl); ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"{label}\nheight (m)")
        axs[row, 0].legend(fontsize=8)
    for ax in axs.ravel():
        ax.set_ylim(0, 1500)
    fig.savefig(os.path.join(out_dir, "fig3_07_profiles.png"), dpi=args.dpi)
    plt.close(fig)

    # ======================= Fig 3.18: 70 m averages =======================
    avgs = [10, 20, 30, 40, 50, 60]
    th70_l = at_height(z_l, th_l, HUB_Z) if th_l is not None else None
    th70_w = at_height(z_w, wt["theta"], HUB_Z)
    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    for n, (label, hc0) in enumerate(TYPICAL):
        for m_, a in enumerate(avgs):
            xpos = n + (m_ - 2.5) * 0.12
            sel = window(h_l, hc0 - a / 60.0, hc0 + 1e-6)
            selw = window(h_w, hc0 - a / 60.0, hc0 + 1e-6)
            vals_l = (np.nanmean(th70_l[sel]) if th70_l is not None else np.nan, np.nanmean(s70_l[sel]),
                      circ_mean(d70_l[sel]))
            vals_w = (np.nanmean(th70_w[selw]), np.nanmean(s70_w[selw]),
                      circ_mean(at_height(z_w, d_w, HUB_Z)[selw]))
            ref = (np.nan, np.nan, np.nan)
            if obs is not None:
                so = window(obs["h"], hc0 - a / 60.0, hc0 + 1e-6) & (np.abs(obs["z"] - HUB_Z) < 10)
                ref = tuple(np.nanmean(obs[k][so]) if k in obs and so.any() else np.nan
                            for k in ("theta", "wspd")) + (circ_mean(obs["wdir"][so]) if "wdir" in obs and so.any()
                                                          else np.nan,)
            for row in range(3):
                yl, yw = vals_l[row], vals_w[row]
                if obs is not None:
                    yl, yw = yl - ref[row], yw - ref[row]
                    if row == 2:
                        yl, yw = (yl + 180) % 360 - 180, (yw + 180) % 360 - 180
                axs[row].plot(xpos, yl, "o", color=f"C{m_}", label=f"{a} min" if n == 0 and row == 0 else None)
                axs[row].plot(xpos, yw, "x", color=f"C{m_}")
    names = ("θ (K)", "wind speed (m/s)", "direction (°)")
    for row in range(3):
        axs[row].set_ylabel(("error " if obs is not None else "") + names[row]); axs[row].grid(alpha=0.3)
    axs[0].legend(ncol=6, fontsize=8, title="o LBM   x WRF")
    axs[2].set_xticks(range(3)); axs[2].set_xticklabels([t[0] for t in TYPICAL])
    fig.suptitle("70 m, averages ending at the typical time" + ("" if obs is not None else " (no obs: model values)"))
    fig.savefig(os.path.join(out_dir, "fig3_18_errors70.png"), dpi=args.dpi)
    plt.close(fig)

    # ======================= summary =======================
    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.write("hour_lst,u70_lbm,u70_wrf,dir70_lbm,theta70_lbm,theta70_wrf,alpha_lbm,alpha_wrf,ti_rotor_lbm\n")
        for r in hourly:
            if r is None:
                continue
            f.write(f"{r['hour']},{r['u70']:.3f},{r['u70_w']:.3f},{r['d70']:.1f},{r['th70']:.2f},"
                    f"{r['th70_w']:.2f},{r['alpha']:.3f},{r['alpha_w']:.3f},{r['ti_rotor']:.3f}\n")
    print(f"[validate] figures and summary.csv -> {out_dir}")


if __name__ == "__main__":
    main()
