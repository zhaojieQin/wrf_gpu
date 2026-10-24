"""Cell perturbation method (CPM) for inflow turbulence in the WRF-driven LBM.

References (literature/):
  Muñoz-Esparza et al. 2015, Phys. Fluids 27, 035102      — cells, Ec, Γ
  Muñoz-Esparza et al. 2017, JAMES 9, 1572                 — real diurnal cycle: z_CP, z_i, U_g, skip levels
  Muñoz-Esparza & Kosović 2018, MWR 146, 1889              — stable scaling Ri_m

Layout (fixed, built once)
--------------------------
Horizontal n_c × n_c-node columns, starting ``offset_nodes`` inside the lateral boundary, kept
when within ``rows`` columns of a lateral face (corner columns belong to two faces). Vertically
every grid level is its own cell (2-D cells, 2015) from fluid level ``skip_levels + 1`` up to
``z_max_m``.

Each refresh (every t_p = Γ d_c / U_1, Γ = 1, d_c = cell diagonal)
-------------------------------------------------------------------
  regime   : mean surface HFX ≥ 0 → convective/neutral, else stable
  z_i      : inflow-weighted θ profile on the lateral faces; convective → height of max ∂θ/∂z,
             stable → lowest height where ∂θ/∂z < stable_dthdz (2017); not found → ∞
  z_CP     : min(zi_fraction · z_i, z_max)
  amplitude: Ec scheme  θ_pm = U_g² / (c_p Ec), U_g = inflow wind speed at min(z_i, top)
             Ri_m scheme (stable, optional) θ_pm(z) = Δθ − Ri_m θ (Δu² + Δv²) / (g Δz)
  w_in     : ½[1 − tanh(u_n / u_0)] per cell from the boundary wind of its face, max over faces
  θ'       : w_in · θ_pm · U(−1, 1), one value per cell, 0 above z_CP
  momentum (optional, off by default): u', v' = α · |U_cell| · w_in · U(−1, 1), mean removed
The values are seeded instantaneously into the GPU distributions (VF CellPerturbation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FACES = ("MX", "PX", "MY", "PY")
# outward normals of the lateral faces
_NORMALS = {"MX": (-1.0, 0.0), "PX": (1.0, 0.0), "MY": (0.0, -1.0), "PY": (0.0, 1.0)}
_GRAVITY = 9.81


@dataclass
class CPMConfig:
    """CPM parameters (units in the field names / comments)."""
    cell_nodes: int = 8              # cell side in grid nodes (2015/2017: 8)
    rows: int = 3                    # rows of cells along each lateral face (2015/2017: 3)
    offset_nodes: int = 3            # first row starts this many nodes inside the boundary
    skip_levels: int = 3             # lowest fluid levels left unperturbed (2017)
    z_max_m: float = 600.0           # highest perturbed level (m AGL); also capped below the domain top
    zi_fraction: float = 2.0 / 3.0   # z_CP = zi_fraction · z_i (2015/2017)
    stable_dthdz: float = 0.014      # K/m, stable z_i criterion (2017)
    conv_min_dthdz: float = 0.003    # K/m, convective: max ∂θ/∂z below this → no inversion in domain
    ec: float = 0.2                  # perturbation Eckert number (2017/2018)
    cp: float = 1004.5               # J kg⁻¹ K⁻¹
    gamma: float = 1.0               # Γ = t_p U_1 / d_c (2015/2017)
    tp_min_s: float = 5.0            # clamp for t_p (s)
    tp_max_s: float = 300.0
    u0: float = 0.5                  # m/s, inflow switch scale in w_in
    angle_max_deg: float | None = None   # optional: no perturbation if wind vs inward normal > this angle
    stable_scheme: str = "ec"        # "ec" or "rim" (2018)
    rim: float = -1.0                # Ri_m for stable_scheme="rim" (2018)
    theta_pm_max: float = 2.0        # K, safety cap on the amplitude
    momentum: bool = False           # also perturb u, v (no literature optimum; off by default)
    momentum_alpha: float = 0.05     # u', v' amplitude as fraction of the local wind speed
    seed: int = 12345


# ---------------------------------------------------------------------------
# pure helpers (unit-testable)
# ---------------------------------------------------------------------------

def inflow_weight(u_n, u0: float):
    """w_in = ½[1 − tanh(u_n / u0)], u_n = wind · outward normal (m/s)."""
    return 0.5 * (1.0 - np.tanh(np.asarray(u_n, dtype=np.float64) / u0))


def theta_pm_from_ec(ug: float, ec: float, cp: float) -> float:
    """θ_pm = U_g² / (c_p Ec)  (K)."""
    return float(ug) ** 2 / (cp * ec)


def theta_pm_from_rim(theta_abs, u, v, dz: float, rim: float):
    """θ_pm(z) from Ri_m = g (Δθ − θ_pm) Δz / (θ (Δu² + Δv²))  (2018, Eq. 1), per level, ≥ 0."""
    theta_abs = np.asarray(theta_abs, dtype=np.float64)
    dth = np.gradient(theta_abs) if theta_abs.size > 1 else np.zeros_like(theta_abs)
    du = np.gradient(np.asarray(u, dtype=np.float64)) if theta_abs.size > 1 else np.zeros_like(theta_abs)
    dv = np.gradient(np.asarray(v, dtype=np.float64)) if theta_abs.size > 1 else np.zeros_like(theta_abs)
    tpm = dth - rim * theta_abs * (du ** 2 + dv ** 2) / (_GRAVITY * dz)
    return np.maximum(tpm, 0.0)


def diagnose_zi(z, theta, regime: str, stable_dthdz: float, conv_min_dthdz: float,
                z_min: float | None = None) -> float:
    """Boundary-layer height from a θ profile (2017 criteria). Returns np.inf if not found.

    Stable: top of the surface inversion = first height, at or above ``z_min``, where
    dθ/dz drops below ``stable_dthdz`` after having been above it. The search starts at
    the inversion, not at the ground: the lowest LBM levels lie below the lowest WRF mass
    level, where the boundary θ is held constant (dθ/dz = 0), which previously gave
    z_i ≈ 5 m and switched every CPM cell off at night.
    """
    z = np.asarray(z, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    ok = np.isfinite(theta)
    z, theta = z[ok], theta[ok]
    if z.size < 3:
        return np.inf
    dthdz = np.gradient(theta, z)
    if regime == "stable":
        start = 0 if z_min is None else int(np.searchsorted(z, float(z_min)))
        g, zz = dthdz[start:], z[start:]
        inversion = np.nonzero(g >= stable_dthdz)[0]
        if not inversion.size:
            return np.inf
        k0 = int(inversion[0])
        top = np.nonzero(g[k0:] < stable_dthdz)[0]
        return float(zz[k0 + top[0]]) if top.size else np.inf
    k = int(np.argmax(dthdz))
    return float(z[k]) if dthdz[k] >= conv_min_dthdz else np.inf


# ---------------------------------------------------------------------------
# perturber
# ---------------------------------------------------------------------------

class CPMPerturber:
    """Builds the cell layout once and seeds perturbations at refresh times.

    Parameters
    ----------
    sim        : VF Simulation (needs set_cell_perturbation_cells / apply_cell_perturbation)
    cfg        : CPMConfig
    dx         : LBM grid spacing (m)
    lx, ly, lz : LBM domain lengths (m); nodes sit at −dx/2 + i·dx
    bc_coords  : (n_bc, 3) open-boundary velocity nodes, LBM coordinates (m), order of db.interp u/v
    ad_coords  : (n_ad, 3) temperature Dirichlet nodes, LBM coordinates, order of db.interp_theta_ad
    theta_ref  : K (θ = θ_ref + concentration)
    velocity_ratio : dx/dt (m/s), converts momentum perturbations to lattice units
    """

    def __init__(self, sim, cfg: CPMConfig, dx: float, lx: float, ly: float, lz: float,
                 bc_coords: np.ndarray, ad_coords: np.ndarray, theta_ref: float,
                 velocity_ratio: float, level: int = 0, verbose: bool = True):
        self.sim, self.cfg, self.dx, self.level = sim, cfg, float(dx), level
        self.theta_ref = float(theta_ref)
        self.velocity_ratio = float(velocity_ratio)
        self.verbose = verbose
        self.rng = np.random.default_rng(cfg.seed)
        self.lx, self.ly, self.lz = float(lx), float(ly), float(lz)
        self.next_refresh_t: float | None = None
        self.n_refresh = 0
        self._build_layout()
        self._build_face_groups(np.asarray(bc_coords, dtype=np.float64),
                                np.asarray(ad_coords, dtype=np.float64))
        n_nodes = sim.set_cell_perturbation_cells(
            self.cell_table, self.x0, self.y0, self.z0, self.cell_dx, self.cell_dy, self.dx,
            self.n_cells, level=level)
        self.n_nodes = int(n_nodes)
        if verbose:
            print(f"[CPM] layout: {self.nxc}x{self.nyc} columns of {cfg.cell_nodes}x{cfg.cell_nodes} nodes "
                  f"({self.cell_dx:.1f} m), {len(self.col_ix)} perimeter columns × levels "
                  f"{self.k_min}..{self.k_max} (z={self._zk(self.k_min):.1f}..{self._zk(self.k_max):.1f} m) "
                  f"= {self.n_cells} cells, {self.n_nodes} fluid nodes; momentum={cfg.momentum}", flush=True)

    # ---- geometry ------------------------------------------------------
    def _zk(self, k):
        return -0.5 * self.dx + np.asarray(k) * self.dx

    def _build_layout(self) -> None:
        c, dx = self.cfg, self.dx
        nint_x = int(round(self.lx / dx))      # interior node count (excl. the two boundary nodes)
        nint_y = int(round(self.ly / dx))
        nint_z = int(round(self.lz / dx))
        n_c = int(c.cell_nodes)
        self.nxc = (nint_x - 2 * c.offset_nodes) // n_c
        self.nyc = (nint_y - 2 * c.offset_nodes) // n_c
        if self.nxc < 2 * c.rows or self.nyc < 2 * c.rows:
            raise ValueError("CPM: domain too small for the requested rows/cell size")
        # centre the cell grid; node i (1..nint) sits at −dx/2 + i·dx, cell edges halfway between nodes
        i_start = 1 + c.offset_nodes + ((nint_x - 2 * c.offset_nodes) - self.nxc * n_c) // 2
        j_start = 1 + c.offset_nodes + ((nint_y - 2 * c.offset_nodes) - self.nyc * n_c) // 2
        self.x0 = (i_start - 1) * dx
        self.y0 = (j_start - 1) * dx
        self.z0 = -0.5 * dx                     # level k sits at z0 + k·dx (k = 0: ground BC node)
        self.cell_dx = self.cell_dy = n_c * dx

        self.k_min = 1 + int(c.skip_levels)
        z_cap = min(float(c.z_max_m), self.lz - 2.0 * dx)
        self.k_max = int(np.floor((z_cap - self.z0) / dx))
        self.k_max = min(self.k_max, nint_z - 1)
        if self.k_max < self.k_min:
            raise ValueError("CPM: z_max_m below the first perturbed level")

        ix, iy = np.meshgrid(np.arange(self.nxc), np.arange(self.nyc), indexing="xy")
        ix, iy = ix.ravel(), iy.ravel()
        faces = {
            "MX": ix < c.rows, "PX": ix >= self.nxc - c.rows,
            "MY": iy < c.rows, "PY": iy >= self.nyc - c.rows,
        }
        keep = faces["MX"] | faces["PX"] | faces["MY"] | faces["PY"]
        self.col_ix, self.col_iy = ix[keep], iy[keep]
        self.col_faces = {f: m[keep] for f, m in faces.items()}

        n_col = self.col_ix.size
        n_lev = self.k_max - self.k_min + 1
        self.n_cells = n_col * n_lev
        # cell id = level_index · n_col + column
        self.cell_k = np.repeat(np.arange(self.k_min, self.k_max + 1), n_col)
        self.cell_col = np.tile(np.arange(n_col), n_lev)
        self.cell_z = self._zk(self.cell_k)

        nzc = self.k_max + 1
        table = -np.ones((nzc, self.nyc, self.nxc), dtype=np.int32)
        for li, k in enumerate(range(self.k_min, self.k_max + 1)):
            table[k, self.col_iy, self.col_ix] = li * n_col + np.arange(n_col, dtype=np.int32)
        self.cell_table = table

    def _classify(self, coords):
        # VF reports the fluid node next to each face (x = dx/2, lx − dx/2, ...), not a node
        # outside the domain, so a face is "within one cell of the boundary". Edge entries may
        # fall into two faces; that only affects the face averages, which is harmless.
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        k = np.rint((z - self.z0) / self.dx).astype(np.int64)
        dx = self.dx
        lateral = z <= self.lz
        return {
            "MX": lateral & (x < dx), "PX": lateral & (x > self.lx - dx),
            "MY": lateral & (y < dx), "PY": lateral & (y > self.ly - dx),
        }, k

    def _build_face_groups(self, bc_coords, ad_coords) -> None:
        """Per face: velocity BC nodes grouped by (level, tangential cell bin); θ nodes by level."""
        masks, k = self._classify(bc_coords)
        # profile levels kept: up to the top of the lateral faces
        self.nk = max(int(k.max()) + 1 if k.size else 1, self.k_max + 1)
        self.vel_groups = {}
        for f in _FACES:
            idx = np.nonzero(masks[f])[0]
            if f in ("MX", "PX"):
                t, t0, nt = bc_coords[idx, 1], self.y0, self.nyc
            else:
                t, t0, nt = bc_coords[idx, 0], self.x0, self.nxc
            tb = np.floor((t - t0) / self.cell_dx).astype(np.int64)
            kk = k[idx]
            valid = (tb >= 0) & (tb < nt) & (kk >= 0) & (kk < self.nk)
            self.vel_groups[f] = dict(
                idx=idx, key=np.where(valid, kk * nt + tb, -1), k=np.clip(kk, 0, self.nk - 1), nt=nt)
        amasks, ak = self._classify(ad_coords)
        self.th_groups = {}
        for f in _FACES:
            idx = np.nonzero(amasks[f])[0]
            kk = ak[idx]
            valid = (kk >= 0) & (kk < self.nk)
            self.th_groups[f] = dict(idx=idx[valid], k=kk[valid])
        # tangential bin of every cell column on each face it belongs to
        self.col_tb = {"MX": self.col_iy, "PX": self.col_iy, "MY": self.col_ix, "PY": self.col_ix}

    # ---- per-refresh diagnostics -----------------------------------------
    @staticmethod
    def _bin_mean(key, values, n):
        ok = key >= 0
        s = np.bincount(key[ok], weights=values[ok], minlength=n)
        c = np.bincount(key[ok], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, s / np.maximum(c, 1), np.nan)

    def _face_winds(self, u, v):
        """Per face: (u, v) per (level, bin) and per level."""
        out = {}
        for f, g in self.vel_groups.items():
            n = self.nk * g["nt"]
            uf, vf = u[g["idx"]], v[g["idx"]]
            ub = self._bin_mean(g["key"], uf, n).reshape(self.nk, g["nt"])
            vb = self._bin_mean(g["key"], vf, n).reshape(self.nk, g["nt"])
            ul = self._bin_mean(g["k"], uf, self.nk)
            vl = self._bin_mean(g["k"], vf, self.nk)
            # empty bins fall back to the level mean of the face
            ub = np.where(np.isnan(ub), ul[:, None], ub)
            vb = np.where(np.isnan(vb), vl[:, None], vb)
            out[f] = (ub, vb, ul, vl)
        return out

    def _w_in(self, u, v, face):
        nx, ny = _NORMALS[face]
        w = inflow_weight(u * nx + v * ny, self.cfg.u0)
        if self.cfg.angle_max_deg is not None:
            speed = np.hypot(u, v)
            with np.errstate(invalid="ignore", divide="ignore"):
                cosang = (-(u * nx + v * ny)) / np.maximum(speed, 1e-12)
            w = np.where(cosang < np.cos(np.radians(self.cfg.angle_max_deg)), 0.0, w)
        return np.nan_to_num(w, nan=0.0)

    def _profiles(self, winds, theta_ad):
        """Inflow-weighted lateral-face profiles: θ (K, absolute), u, v per level."""
        wsum = np.zeros(self.nk)
        th = np.zeros(self.nk)
        uu = np.zeros(self.nk)
        vv = np.zeros(self.nk)
        face_w = {}
        for f in _FACES:
            _, _, ul, vl = winds[f]
            face_w[f] = float(np.nanmean(self._w_in(ul, vl, f))) if np.isfinite(ul).any() else 0.0
        if sum(face_w.values()) < 1e-6:
            face_w = {f: 1.0 for f in _FACES}
        for f in _FACES:
            wf = face_w[f]
            _, _, ul, vl = winds[f]
            g = self.th_groups[f]
            thl = self._bin_mean(g["k"], theta_ad[g["idx"]], self.nk) if theta_ad is not None else np.full(self.nk, np.nan)
            ok = np.isfinite(thl) & np.isfinite(ul) & np.isfinite(vl)
            wsum += np.where(ok, wf, 0.0)
            th += np.where(ok, wf * np.nan_to_num(thl), 0.0)
            uu += np.where(ok, wf * np.nan_to_num(ul), 0.0)
            vv += np.where(ok, wf * np.nan_to_num(vl), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            th = np.where(wsum > 0, th / wsum, np.nan) + self.theta_ref
            uu = np.where(wsum > 0, uu / wsum, np.nan)
            vv = np.where(wsum > 0, vv / wsum, np.nan)
        return th, uu, vv

    # ---- main entry --------------------------------------------------------
    def maybe_refresh(self, t_lbm: float, u, v, theta_ad=None, hfx=None) -> bool:
        """Seed a new set of perturbations if t_lbm reached the refresh time. Returns True if seeded."""
        if self.next_refresh_t is not None and t_lbm < self.next_refresh_t:
            return False
        c = self.cfg
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        winds = self._face_winds(u, v)
        th, up, vp = self._profiles(winds, None if theta_ad is None else np.asarray(theta_ad, np.float64))

        regime = "stable" if (hfx is not None and np.size(hfx) and float(np.mean(hfx)) < 0.0) else "convective"
        levels = np.arange(1, self.nk)
        zl = self._zk(levels)
        # stable z_i is searched from the lowest perturbed level up (skips the clamped
        # sub-WRF-level layer near the ground)
        zi = diagnose_zi(zl, th[levels], regime, c.stable_dthdz, c.conv_min_dthdz,
                         z_min=float(self._zk(self.k_min)))
        z_cp = min(c.zi_fraction * zi, c.z_max_m)

        speed = np.hypot(up, vp)
        ok = np.isfinite(speed[levels])
        z_ug = min(zi, float(zl[ok][-1])) if ok.any() else 0.0
        ug = float(np.interp(z_ug, zl[ok], speed[levels][ok])) if ok.any() else 0.0
        if regime == "stable" and c.stable_scheme == "rim":
            tpm_level = theta_pm_from_rim(np.nan_to_num(th, nan=self.theta_ref), np.nan_to_num(up),
                                          np.nan_to_num(vp), self.dx, c.rim)
            theta_pm = np.minimum(tpm_level[self.cell_k], c.theta_pm_max)
            tpm_report = float(np.max(theta_pm)) if theta_pm.size else 0.0
        else:
            tpm_report = min(theta_pm_from_ec(ug, c.ec, c.cp), c.theta_pm_max)
            theta_pm = np.full(self.n_cells, tpm_report)

        # per-cell inflow weight (max over the faces of its column)
        w_cell = np.zeros(self.n_cells)
        u_cell = np.zeros(self.n_cells)
        v_cell = np.zeros(self.n_cells)
        for f in _FACES:
            on = self.col_faces[f][self.cell_col]
            if not on.any():
                continue
            ub, vb, _, _ = winds[f]
            kk = np.clip(self.cell_k[on], 0, self.nk - 1)
            tb = self.col_tb[f][self.cell_col[on]]
            uc, vc = ub[kk, tb], vb[kk, tb]
            w = self._w_in(uc, vc, f)
            better = w > w_cell[on]
            idx = np.nonzero(on)[0]
            w_cell[idx[better]] = w[better]
            u_cell[idx[better]] = np.nan_to_num(uc[better])
            v_cell[idx[better]] = np.nan_to_num(vc[better])

        active = self.cell_z <= z_cp
        theta_prime = np.where(active, w_cell * theta_pm * self.rng.uniform(-1.0, 1.0, self.n_cells), 0.0)

        u_prime = v_prime = None
        if c.momentum:
            amp = c.momentum_alpha * np.hypot(u_cell, v_cell) * w_cell
            up_c = np.where(active, amp * self.rng.uniform(-1.0, 1.0, self.n_cells), 0.0)
            vp_c = np.where(active, amp * self.rng.uniform(-1.0, 1.0, self.n_cells), 0.0)
            if active.any():   # no net momentum injection
                up_c[active] -= up_c[active].mean()
                vp_c[active] -= vp_c[active].mean()
            u_prime = (up_c / self.velocity_ratio).astype(np.float32)
            v_prime = (vp_c / self.velocity_ratio).astype(np.float32)

        self.sim.apply_cell_perturbation(theta_prime.astype(np.float32), u_prime, v_prime)

        u1 = float(speed[1]) if np.isfinite(speed[1]) else 0.0
        d_c = np.sqrt(2.0) * self.cell_dx
        t_p = c.gamma * d_c / max(u1, 1e-6)
        t_p = float(np.clip(t_p, c.tp_min_s, c.tp_max_s))
        self.next_refresh_t = t_lbm + t_p
        self.n_refresh += 1
        if self.verbose:
            n_act = int(np.count_nonzero(theta_prime))
            print(f"[CPM] #{self.n_refresh} t={t_lbm:.1f}s regime={regime} zi={zi:.0f}m z_CP={z_cp:.0f}m "
                  f"Ug={ug:.2f}m/s θpm={tpm_report:.3f}K U1={u1:.2f}m/s tp={t_p:.1f}s "
                  f"active={n_act}/{self.n_cells} mean(w_in|active z)={float(w_cell[active].mean()) if active.any() else 0.0:.2f} "
                  f"max|θ'|={float(np.abs(theta_prime).max()):.3f}K"
                  + (f" max|u'|={float(np.abs(u_prime).max()) * self.velocity_ratio:.3f}m/s" if u_prime is not None else ""),
                  flush=True)
        return True
