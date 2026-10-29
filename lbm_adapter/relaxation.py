"""Lateral relaxation zone (one-way nesting buffer) for the coupled LBM.

Every fluid node within ``width_nodes`` cells of a selected lateral face is nudged towards the
WRF state each LBM step (VF ``BoundaryRelaxationZone``, C++/CUDA):

    du/dt|relax = r(d) / tau · (u_WRF − u),     dθ/dt|relax = r(d) / tau · (θ_WRF − θ),
    r(d) = cos²(π d / (2 W))  (1 at the face, 0 at the inner edge of the zone),

velocity as a body force, θ as an increment of the advection-diffusion distributions. The two
inflow streams (e.g. west and south face) are thus brought to the WRF state before they meet in the
interior, instead of being forced only on the single boundary layer of nodes (Davies relaxation as
in the WRF relax zone; Muñoz-Esparza et al. 2014/2017 place the CPM cells just inside it).

Targets: every received WRF frame is interpolated to the zone nodes (same interpolation as the LBM
boundary data) and pushed to the GPU; VF interpolates linearly in time between the last two frames.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from lbm_adapter.vertical_interp import _ColumnSource, build_lbm_grid_table
from lbm_adapter.wrf_to_lbm import WRFStaticContext, extract_most_fields, make_wrf_column_data

FACE_BITS = {"MX": 1, "PX": 2, "MY": 4, "PY": 8, "PZ": 16}


@dataclass
class RelaxationConfig:
    width_nodes: int = 20                    # lateral zone width in grid cells
    tau_s: float = 20.0                      # relaxation time at the face (s)
    relax_velocity: bool = True
    relax_temperature: bool = True
    faces: tuple[str, ...] = ("MX", "PX", "MY", "PY")
    top_width_nodes: int = 0                 # zone below the roof (only with "PZ" in faces)
    diag_every_frames: int = 60              # print target statistics every N pushed frames


class RelaxationZone:
    """Python driver of VF's BoundaryRelaxationZone.

    Parameters
    ----------
    sim            : VF Simulation (set_relaxation_zone / get_relaxation_zone_nodes /
                     push_relaxation_targets / apply_relaxation)
    cfg            : RelaxationConfig
    dx, lx, ly, lz : LBM grid spacing and domain lengths (m); nodes sit at −dx/2 + i·dx
    dt_lbm         : LBM time step (s)
    static_ctx     : WRFStaticContext of the WRF target domain (d03)
    origin_xy      : LBM SW corner in WRF coordinates (m)
    theta_ref      : K (targets are θ − θ_ref, as the AD concentration)
    velocity_ratio : dx/dt (m/s), converts the WRF velocity to lattice units
    """

    def __init__(self, sim, cfg: RelaxationConfig, dx: float, lx: float, ly: float, lz: float, dt_lbm: float,
                 static_ctx: WRFStaticContext, origin_xy: tuple[float, float], theta_ref: float,
                 velocity_ratio: float, level: int = 0):
        self.sim, self.cfg, self.ctx = sim, cfg, static_ctx
        self.dx = float(dx)
        self.origin = (float(origin_xy[0]), float(origin_xy[1]))
        self.theta_ref = float(theta_ref)
        self.velocity_ratio = float(velocity_ratio)
        self.rate = float(dt_lbm) / float(cfg.tau_s)
        if not 0.0 < self.rate < 1.0:
            raise ValueError(f"relaxation rate dt/tau = {self.rate} must be in (0, 1)")
        unknown = set(cfg.faces) - set(FACE_BITS)
        if unknown:
            raise ValueError(f"unknown relaxation faces {sorted(unknown)} (use {sorted(FACE_BITS)})")
        mask = sum(FACE_BITS[f] for f in cfg.faces)
        width = cfg.width_nodes * self.dx
        top_width = cfg.top_width_nodes * self.dx
        n = int(sim.set_relaxation_zone(0.0, lx, 0.0, ly, lz, width, top_width, mask, level=level))
        coords, weights = sim.get_relaxation_zone_nodes()
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        self.n = n
        self.weights = np.asarray(weights, dtype=np.float64)
        self._build_strips(coords, lx, ly, lz, width, top_width)
        self.n_pushed = 0
        mem_mb = n * (4 + 4 + 2 * 4 * 4) / 1e6
        print(f"[relax] zone: {n} fluid nodes, width {cfg.width_nodes} cells ({width:.0f} m), faces {cfg.faces}"
              + (f", top {cfg.top_width_nodes} cells" if "PZ" in cfg.faces else "")
              + f"; tau={cfg.tau_s:g} s (dt/tau={self.rate:.4g}), velocity={cfg.relax_velocity}, "
              f"temperature={cfg.relax_temperature}; strips "
              + ", ".join(f"{s['name']}={s['idx'].size}" for s in self.strips)
              + f"; GPU ≈ {mem_mb:.0f} MB", flush=True)

    # ---- geometry: zone nodes -> tensor-product strips (one WRF table per strip and frame) ----------
    def _build_strips(self, coords, lx, ly, lz, width, top_width):
        dx = self.dx
        nx, ny, nz = (int(round(L / dx)) for L in (lx, ly, lz))     # fluid nodes 1..n per axis
        i = np.rint((coords[:, 0] + 0.5 * dx) / dx).astype(np.int64)
        j = np.rint((coords[:, 1] + 0.5 * dx) / dx).astype(np.int64)
        k = np.rint((coords[:, 2] + 0.5 * dx) / dx).astype(np.int64)
        wn = max(1, int(math.ceil(width / dx + 0.5)) - 1)             # last index with (i − ½)dx < width
        tn = max(0, int(math.ceil(top_width / dx + 0.5)) - 1)
        full = lambda n_: (1, n_)                                     # noqa: E731
        spec = []
        f = self.cfg.faces
        if "MX" in f:
            spec.append(("W", (1, wn), full(ny), full(nz)))
        if "PX" in f:
            spec.append(("E", (nx - wn + 1, nx), full(ny), full(nz)))
        if "MY" in f:
            spec.append(("S", full(nx), (1, wn), full(nz)))
        if "PY" in f:
            spec.append(("N", full(nx), (ny - wn + 1, ny), full(nz)))
        if "PZ" in f and tn > 0:
            spec.append(("T", full(nx), full(ny), (nz - tn + 1, nz)))

        owner = np.full(self.n, -1, dtype=np.int64)
        self.strips = []
        for name, (i0, i1), (j0, j1), (k0, k1) in spec:
            inside = (owner < 0) & (i >= i0) & (i <= i1) & (j >= j0) & (j <= j1) & (k >= k0) & (k <= k1)
            idx = np.nonzero(inside)[0]
            owner[idx] = len(self.strips)
            n_i, n_k = i1 - i0 + 1, k1 - k0 + 1
            flat = ((j[idx] - j0) * n_i + (i[idx] - i0)) * n_k + (k[idx] - k0)
            self.strips.append(dict(
                name=name, idx=idx, flat=flat,
                x=-0.5 * dx + dx * np.arange(i0, i1 + 1) + self.origin[0],
                y=-0.5 * dx + dx * np.arange(j0, j1 + 1) + self.origin[1],
                z=np.maximum(-0.5 * dx + dx * np.arange(k0, k1 + 1), 0.0),
            ))
        if np.any(owner < 0):
            raise RuntimeError(f"[relax] {int(np.sum(owner < 0))} zone nodes outside every strip "
                               "(zone geometry and strip layout disagree)")

    # ---- per frame: WRF -> zone targets ---------------------------------------------------------------
    def push_frame(self, received: dict) -> None:
        t_start = time.perf_counter()
        data = make_wrf_column_data(received, self.ctx)
        most = extract_most_fields(received)
        src = _ColumnSource(data, most)
        fields = ("u", "v", "w", "theta") if self.cfg.relax_temperature else ("u", "v", "w")
        out = np.zeros((len(fields), self.n), dtype=np.float32)
        for s in self.strips:
            tabs = build_lbm_grid_table(data, s["x"], s["y"], s["z"], most_fields=most, fields=fields, source=src)
            for c, tab in enumerate(tabs):
                out[c, s["idx"]] = tab.reshape(-1)[s["flat"]]
        inv = np.float32(1.0 / self.velocity_ratio)
        u, v, w = out[0] * inv, out[1] * inv, out[2] * inv
        theta = (out[3] - np.float32(self.theta_ref)) if self.cfg.relax_temperature else None
        self.sim.push_relaxation_targets(u, v, w, theta)
        self.n_pushed += 1
        if self.cfg.diag_every_frames and (self.n_pushed <= 2 or self.n_pushed % self.cfg.diag_every_frames == 0):
            spd = np.hypot(out[0], out[1])
            msg = (f"[relax] frame #{self.n_pushed} t_wrf={float(received.get('__wrf_time_s', 0.0)):.0f}s: "
                   f"target |U| {spd.min():.2f}…{spd.max():.2f} m/s, w {out[2].min():+.2f}…{out[2].max():+.2f} m/s")
            if theta is not None:
                msg += f", θ−θ_ref {theta.min():+.2f}…{theta.max():+.2f} K"
            print(msg + f"  ({time.perf_counter() - t_start:.2f} s)", flush=True)

    # ---- per LBM step ----------------------------------------------------------------------------------
    def apply(self, t_lbm: float, db) -> None:
        span = db.t1 - db.t0
        alpha = 1.0 if span <= 0.0 else float(np.clip((t_lbm - db.t0) / span, 0.0, 1.0))
        self.sim.apply_relaxation(alpha, self.rate, self.cfg.relax_velocity, self.cfg.relax_temperature)
