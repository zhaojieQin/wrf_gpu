"""Diurnal-cycle validation sampling (SWiFT, TTU 200 m tower; see scripts/validate_swift.py).

Everything here only samples; nothing changes the LBM or WRF solution.

LBM side (VirtualFluids probes, added through setup_vf_simulation(extra_samplers=...)):
  - ``valTower``   : point probe, one LBM node column at the tower position, every fluid node from
                     the ground up to ``tower_zmax_m``; instantaneous u, v, w, rho (+ θ−θ_ref) as a
                     time series every ``tower_every_s`` (binary file valTower_timeseries_*.txt).
  - ``valPlanarZ`` : horizontal planar averages (means + covariances, + θ) over the whole LBM
                     domain, instantaneous every ``planar_every_s`` (no running time average).
  - ``valHubPlane``: horizontal plane at the node closest to ``hub_plane_z_m``, instantaneous
                     every ``hub_plane_every_s`` (structure snapshots).

WRF side (rank 1, from the frames it already receives):
  - ``WRFTowerSampler``: WRF d03 u, v, w, θ at the tower column on the LBM tower heights, plus
                         hfx / ustar there, every frame; the mean profile over the LBM footprint
                         every ``wrf_footprint_every_s``. Written to <out>/wrf_tower.nc.

Run metadata (θ_ref, dt, dx, start date, tower position, ...) goes to <out>/meta.json.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lbm_adapter.vertical_interp import build_lbm_grid_table
from lbm_adapter.wrf_to_lbm import WRFStaticContext, extract_most_fields, make_wrf_column_data


@dataclass
class ValidationConfig:
    tower_latlon: tuple[float, float] | None = None   # (lat, lon); None → LBM domain centre
    tower_zmax_m: float = 1500.0                      # highest tower node (m AGL)
    tower_every_s: float = 1.0                        # tower sampling interval
    tower_write_every_s: float = 300.0                # tower file flush interval (GPU buffer size)
    planar_every_s: float = 60.0                      # planar-average output interval
    hub_plane_z_m: float = 70.0                       # hub-height plane
    hub_plane_every_s: float | None = 600.0           # None → no hub plane
    wrf_footprint_every_s: float = 60.0               # WRF footprint-mean profile interval


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def latlon_to_wrf_xy(lat: float, lon: float, xlat: np.ndarray, xlong: np.ndarray,
                     dx: float, dy: float) -> tuple[float, float]:
    """(lat, lon) → WRF d03 (x, y) in m, origin at mass point (j, i) = (0, 0).

    Nearest mass point, then one linear (Jacobian) correction with central differences.
    """
    ny, nx = xlat.shape
    d2 = (xlat - lat) ** 2 + ((xlong - lon) * math.cos(math.radians(lat))) ** 2
    j, i = np.unravel_index(int(np.argmin(d2)), d2.shape)
    j = min(max(int(j), 1), ny - 2)
    i = min(max(int(i), 1), nx - 2)
    jac = np.array([
        [(xlat[j, i + 1] - xlat[j, i - 1]) / 2.0, (xlat[j + 1, i] - xlat[j - 1, i]) / 2.0],
        [(xlong[j, i + 1] - xlong[j, i - 1]) / 2.0, (xlong[j + 1, i] - xlong[j - 1, i]) / 2.0],
    ], dtype=np.float64)
    di, dj = np.linalg.solve(jac, np.array([lat - xlat[j, i], lon - xlong[j, i]], dtype=np.float64))
    return (i + di) * dx, (j + dj) * dy


def tower_column_nodes(x_lbm: float, y_lbm: float, dx: float, lz: float,
                       zmax: float) -> tuple[float, float, np.ndarray]:
    """LBM node column closest to (x_lbm, y_lbm): node x, node y, fluid node heights ≤ zmax.

    VF nodes sit at -dx/2 + k·dx; k = 0 is the ground BC layer, so heights start at k = 1.
    The column stays 2 dx below the domain top.
    """
    i = int(round((x_lbm + 0.5 * dx) / dx))
    j = int(round((y_lbm + 0.5 * dx) / dx))
    z_top = min(zmax, lz - 2.0 * dx)
    n = int(math.floor((z_top + 0.5 * dx) / dx))
    z = -0.5 * dx + dx * np.arange(1, n + 1)
    return -0.5 * dx + i * dx, -0.5 * dx + j * dx, z


# ---------------------------------------------------------------------------
# LBM probes
# ---------------------------------------------------------------------------

def make_lbm_samplers(vcfg: ValidationConfig, cfg, tower_xy_lbm: tuple[float, float],
                      tower_z: np.ndarray, sample_scalar: bool):
    """Return extra_samplers(para, cuda_memory_manager) for setup_vf_simulation."""

    def steps(seconds: float) -> int:
        return max(1, int(round(seconds / cfg.dt)))

    def add(para, cuda_memory_manager):
        from pyfluids import basics, gpu

        out = str(cfg.output_path)
        dx = cfg.dx
        x_t, y_t = tower_xy_lbm

        # Tower column: a probe point in the middle of the cell [node, node + dx] selects exactly
        # that node (Probe::addLevelData tests "point inside the node's cell").
        n_s, n_w = steps(vcfg.tower_every_s), steps(vcfg.tower_write_every_s)
        n_w = max(n_s, (n_w // n_s) * n_s)
        tower = gpu.probes.Probe(para, cuda_memory_manager, out, "valTower",
                                 0, n_s,          # t_start_avg, t_avg (sampling interval)
                                 0, n_w,          # t_start_out, t_out (file flush interval)
                                 True, False,     # output_timeseries, average_every_timestep
                                 sample_scalar)
        zs = [float(z) + 0.5 * dx for z in tower_z]
        tower.add_probe_points_from_list([x_t + 0.5 * dx] * len(zs), [y_t + 0.5 * dx] * len(zs), zs)
        tower.add_statistic(gpu.probes.Probe.Statistic.Instantaneous)
        para.add_sampler(tower)
        print(f"[validation] tower probe at LBM (x={x_t:.1f}, y={y_t:.1f}) m, {len(zs)} nodes "
              f"z={tower_z[0]:.1f}..{tower_z[-1]:.1f} m, every {n_s} steps "
              f"(≈{n_s * cfg.dt:.2f} s), flushed every {n_w} steps", flush=True)

        # Horizontal planar averages, instantaneous (positional args: C++ order is outputPath, probeName)
        n_p = steps(vcfg.planar_every_s)
        planar = gpu.probes.PlanarAverageProbe(
            para, cuda_memory_manager, out, "valPlanarZ",
            0, 0, n_p,          # t_start_sampling, t_start_tmp_averaging, t_between_samples
            0, n_p,             # t_start_out, t_out
            basics.geometry3d.Axis.z,
            False,              # compute_time_averages (would be cumulative over the whole run)
            sample_scalar,
        )
        planar.add_statistic(gpu.probes.PlanarAverageProbe.Statistic.Means)
        planar.add_statistic(gpu.probes.PlanarAverageProbe.Statistic.Covariances)
        para.add_sampler(planar)
        print(f"[validation] planar-average probe every {n_p} steps (≈{n_p * cfg.dt:.1f} s)", flush=True)

        # Hub-height plane snapshots
        if vcfg.hub_plane_every_s is not None:
            n_h = steps(vcfg.hub_plane_every_s)
            k = max(1, int(round((vcfg.hub_plane_z_m + 0.5 * dx) / dx)))
            z_node = -0.5 * dx + k * dx
            hub = gpu.probes.Probe(para, cuda_memory_manager, out, "valHubPlane",
                                   0, 1, 0, n_h, False, False, sample_scalar)
            hub.set_probe_plane(-dx, -dx, z_node - 0.5 * dx,
                                cfg.domain_length_x + 2 * dx, cfg.domain_length_y + 2 * dx, dx)
            hub.add_statistic(gpu.probes.Probe.Statistic.Instantaneous)
            para.add_sampler(hub)
            print(f"[validation] hub plane at z={z_node:.1f} m every {n_h} steps "
                  f"(≈{n_h * cfg.dt:.0f} s)", flush=True)

    return add


# ---------------------------------------------------------------------------
# WRF tower sampling (rank 1, from the received frames)
# ---------------------------------------------------------------------------

def _window(received: dict, static_ctx: WRFStaticContext, j0: int, j1: int, i0: int, i1: int):
    """Cut a horizontal window [j0:j1, i0:i1] (mass points) out of a received frame.

    Staggered fields (u: nx+1, v: ny+1, w/ph: nz+1 vertically) keep their extra point.
    """
    ny, nx = static_ctx.terrain_height.shape
    sub = {}
    for key, val in received.items():
        if isinstance(val, np.ndarray) and val.ndim >= 2:
            ey, ex = val.shape[-2] - ny, val.shape[-1] - nx
            sub[key] = val[..., j0:j1 + ey, i0:i1 + ex]
        else:
            sub[key] = val
    ctx = WRFStaticContext(terrain_height=static_ctx.terrain_height[j0:j1, i0:i1],
                           dx_wrf=static_ctx.dx_wrf, dy_wrf=static_ctx.dy_wrf,
                           z0_mean=static_ctx.z0_mean)
    return sub, ctx


def _bilinear_2d(field: np.ndarray, x: float, y: float, dx: float, dy: float) -> float:
    ny, nx = field.shape
    fi = min(max(x / dx, 0.0), nx - 1.000001)
    fj = min(max(y / dy, 0.0), ny - 1.000001)
    i, j = int(fi), int(fj)
    a, b = fi - i, fj - j
    return float((1 - a) * (1 - b) * field[j, i] + a * (1 - b) * field[j, i + 1]
                 + (1 - a) * b * field[j + 1, i] + a * b * field[j + 1, i + 1])


class WRFTowerSampler:
    """WRF d03 profiles at the tower (every frame) and over the LBM footprint (every N s)."""

    def __init__(self, path: Path, static_ctx: WRFStaticContext, tower_xy_wrf: tuple[float, float],
                 z_agl: np.ndarray, footprint_wrf: tuple[float, float, float, float],
                 footprint_every_s: float, sync_every: int = 60):
        import netCDF4 as nc

        self.ctx = static_ctx
        self.x, self.y = tower_xy_wrf
        self.z = np.asarray(z_agl, dtype=np.float64)
        self.fp_every = float(footprint_every_s)
        self.next_fp = -math.inf
        self.sync_every = int(sync_every)
        self.n = self.n_fp = 0

        dx, dy = static_ctx.dx_wrf, static_ctx.dy_wrf
        ny, nx = static_ctx.terrain_height.shape
        # tower window: 4 × 4 mass points around the tower (enough for bilinear + destagger)
        it, jt = int(self.x // dx), int(self.y // dy)
        self.tw = (max(jt - 1, 0), min(jt + 3, ny), max(it - 1, 0), min(it + 3, nx))
        # footprint: WRF mass points inside the LBM domain
        x0, x1, y0, y1 = footprint_wrf
        self.fw = (max(int(math.ceil(y0 / dy)), 0), min(int(math.floor(y1 / dy)) + 1, ny),
                   max(int(math.ceil(x0 / dx)), 0), min(int(math.floor(x1 / dx)) + 1, nx))

        path.parent.mkdir(parents=True, exist_ok=True)
        self.ds = nc.Dataset(path, "w")
        ds = self.ds
        ds.createDimension("time", None)
        ds.createDimension("time_fp", None)
        ds.createDimension("z", self.z.size)
        ds.createVariable("z", "f8", ("z",))[:] = self.z
        ds["z"].units = "m AGL"
        ds.createVariable("time", "f8", ("time",)).units = "s since WRF start"
        ds.createVariable("time_fp", "f8", ("time_fp",)).units = "s since WRF start"
        for name, units in (("u", "m s-1"), ("v", "m s-1"), ("w", "m s-1"), ("theta", "K")):
            ds.createVariable(name, "f4", ("time", "z")).units = units
            ds.createVariable(name + "_fp", "f4", ("time_fp", "z")).units = units
        ds.createVariable("hfx", "f4", ("time",)).units = "W m-2"
        ds.createVariable("ustar", "f4", ("time",)).units = "m s-1"
        ds.tower_x_wrf, ds.tower_y_wrf = self.x, self.y
        ds.footprint_ji = list(self.fw)
        ds.note = ("u, v, w, theta: WRF d03 at the tower column (same interpolation as the LBM "
                   "boundary data); *_fp: mean over the WRF mass points inside the LBM footprint. "
                   "theta is the full (moist) potential temperature.")

    def _profiles(self, received: dict, win, x_pts: np.ndarray, y_pts: np.ndarray):
        j0, j1, i0, i1 = win
        sub, ctx = _window(received, self.ctx, j0, j1, i0, i1)
        data = make_wrf_column_data(sub, ctx)
        return build_lbm_grid_table(data, x_pts - i0 * ctx.dx_wrf, y_pts - j0 * ctx.dy_wrf, self.z,
                                    most_fields=extract_most_fields(sub),
                                    fields=("u", "v", "w", "theta"))

    def __call__(self, received: dict) -> None:
        t = float(received.get("__wrf_time_s", 0.0))
        ds, n = self.ds, self.n
        tabs = self._profiles(received, self.tw, np.array([self.x]), np.array([self.y]))
        ds["time"][n] = t
        for name, tab in zip(("u", "v", "w", "theta"), tabs):
            ds[name][n, :] = tab[0, 0, :]
        dx, dy = self.ctx.dx_wrf, self.ctx.dy_wrf
        for key in ("hfx", "ustar"):
            if key in received:
                ds[key][n] = _bilinear_2d(np.asarray(received[key], dtype=np.float64),
                                          self.x, self.y, dx, dy)
        self.n += 1

        if t >= self.next_fp:
            j0, j1, i0, i1 = self.fw
            tabs = self._profiles(received, self.fw, np.arange(i0, i1) * dx, np.arange(j0, j1) * dy)
            m = self.n_fp
            ds["time_fp"][m] = t
            for name, tab in zip(("u", "v", "w", "theta"), tabs):
                ds[name + "_fp"][m, :] = tab.mean(axis=(0, 1))
            self.n_fp += 1
            self.next_fp = t + self.fp_every - 1e-6

        if self.n % self.sync_every == 0:
            ds.sync()

    def close(self) -> None:
        if self.ds is not None:
            self.ds.close()
            self.ds = None


def write_meta(path: Path, **meta) -> None:
    """Run metadata for scripts/validate_swift.py."""
    def conv(v):
        if isinstance(v, ValidationConfig):
            return asdict(v)
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, Path):
            return str(v)
        return v

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: conv(v) for k, v in meta.items()}, indent=2))
