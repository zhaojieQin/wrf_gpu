# Copied from src/gpuwrf/wrf_lbm_coupling/vertical_interp.py
# 改动要同步两边。此副本用于 lbm_adapter（无 gpuwrf 包依赖）。

"""Vertical coordinate conversion: WRF eta-levels → physical height z.

Design: approach A (interpolate in eta-space on demand).

Given WRF geopotential (ph_total = phb + ph', m² s⁻²), builds a per-column
z profile and interpolates any 3-D WRF field to a set of target physical
heights supplied by the LBM side.

All functions are pure (no side effects). Inputs/outputs are numpy arrays so
they work before JAX device arrays are transferred.

Vertical index convention (WRF model order):
  k=0  → lowest model level (surface)
  k=nz → model top          (for w-level arrays of length nz+1)

Shape conventions:
  ph_total  : (nz+1, ny, nx)   w-level geopotential, m² s⁻²
  u, v, theta : (nz, ny, nx)   mass-level fields
  w           : (nz+1, ny, nx) w-level field
  terrain_height : (ny, nx)    m
  eta_levels     : (nz+1,)     ZNW, 1.0→0.0

References
----------
WRF ARW Tech Note §2: z_w = (PHB + PH) / g, z_mass = 0.5*(z_w[k] + z_w[k+1])
"""

from __future__ import annotations

import warnings
from bisect import bisect_right
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

G = 9.81  # m s⁻², WRF uses exactly 9.81
_KAPPA = 0.4  # von Kármán constant


# ---------------------------------------------------------------------------
# z-profile helpers
# ---------------------------------------------------------------------------

def compute_z_w(ph_total: np.ndarray) -> np.ndarray:
    """Convert total geopotential to physical height on w-levels.

    Parameters
    ----------
    ph_total : (nz+1, ny, nx)  m² s⁻²  (= phb + ph_perturbation)

    Returns
    -------
    z_w : (nz+1, ny, nx)  m

    Derivation: z = Φ / g  where Φ = PHB + PH (total geopotential).
    """
    return np.asarray(ph_total, dtype=np.float64) / G


def compute_z_mass(z_w: np.ndarray) -> np.ndarray:
    """Compute mass-level heights by averaging adjacent w-level heights.

    Parameters
    ----------
    z_w : (nz+1, ny, nx)  m  from :func:`compute_z_w`

    Returns
    -------
    z_mass : (nz, ny, nx)  m   k=0 is lowest mass level
    """
    return 0.5 * (z_w[:-1] + z_w[1:])


# ---------------------------------------------------------------------------
# Core interpolation
# ---------------------------------------------------------------------------

def _interp_column_1d(
    field_col: np.ndarray,
    z_col: np.ndarray,
    z_target: float,
    clamp: bool = True,
) -> tuple[float, bool, bool]:
    """Interpolate one column of a 1-D WRF field to z_target.

    Parameters
    ----------
    field_col : (nz,) or (nz+1,)  field values on z_col levels
    z_col     : (nz,) or (nz+1,)  monotonically increasing heights (m)
    z_target  : target height (m)
    clamp     : if True, clamp out-of-range z_target to boundary values.

    Returns
    -------
    value       : interpolated value
    below_floor : True if z_target was below z_col[0]
    above_ceil  : True if z_target was above z_col[-1]
    """
    nk = len(z_col)
    below_floor = z_target < z_col[0]
    above_ceil = z_target > z_col[-1]

    if below_floor:
        return float(field_col[0]), True, False
    if above_ceil:
        return float(field_col[-1]), False, True

    # bisect_right returns the index AFTER the last value <= z_target,
    # i.e. z_col[idx-1] <= z_target < z_col[idx].
    idx = bisect_right(z_col, z_target)
    idx = min(idx, nk - 1)   # safety clamp (can only hit nk if z_target==z_col[-1])

    if idx == 0:
        return float(field_col[0]), False, False

    z_lo, z_hi = z_col[idx - 1], z_col[idx]
    dz = z_hi - z_lo
    if dz <= 0.0:
        return float(field_col[idx - 1]), False, False

    alpha = (z_target - z_lo) / dz
    value = float(field_col[idx - 1]) * (1.0 - alpha) + float(field_col[idx]) * alpha
    return value, False, False


def interp_field_to_heights(
    field: np.ndarray,
    z_levels: np.ndarray,
    target_z: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    terrain_height: np.ndarray,
    dx_wrf: float,
    dy_wrf: float,
) -> np.ndarray:
    """Interpolate a 3-D WRF field to LBM node positions.

    Implements approach A: for each LBM node (x, y, z_target), look up the
    WRF column at (j_wrf, i_wrf), build the z profile, invert z_target → η,
    and interpolate the field value.

    Parameters
    ----------
    field        : (nk, ny, nx)  WRF field on mass levels or w-levels
    z_levels     : (nk, ny, nx)  corresponding physical heights (m);
                   use z_mass for mass-level fields, z_w for w-level fields
    target_z     : (N,)  LBM node heights above MSL (m)
    target_x     : (N,)  LBM node x positions in WRF grid coords (m from SW corner)
    target_y     : (N,)  LBM node y positions in WRF grid coords (m from SW corner)
    terrain_height : (ny, nx)  WRF terrain height (m MSL)
    dx_wrf       : WRF d03 grid spacing in x (m)
    dy_wrf       : WRF d03 grid spacing in y (m)

    Returns
    -------
    values : (N,)  interpolated field values at LBM node positions

    Raises
    ------
    ValueError
        If any LBM node falls outside the WRF horizontal grid.

    Notes
    -----
    - Out-of-range vertical targets are clamped to the boundary layer value
      with a one-time warning each direction.
    - Horizontal interpolation is nearest-neighbour (WRF d03 resolution is
      typically ≪ LBM domain; bilinear can be added later if needed).
    """
    field = np.asarray(field, dtype=np.float64)
    z_levels = np.asarray(z_levels, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    target_x = np.asarray(target_x, dtype=np.float64)
    target_y = np.asarray(target_y, dtype=np.float64)
    terrain_height = np.asarray(terrain_height, dtype=np.float64)

    nk, ny, nx = field.shape
    n_nodes = len(target_z)
    if not (len(target_x) == len(target_y) == n_nodes):
        raise ValueError("target_z, target_x, target_y must have the same length")

    # Horizontal bounds check
    x_max = (nx - 1) * dx_wrf
    y_max = (ny - 1) * dy_wrf
    out_of_bounds = (
        (target_x < 0) | (target_x > x_max) |
        (target_y < 0) | (target_y > y_max)
    )
    if out_of_bounds.any():
        bad = np.where(out_of_bounds)[0]
        raise ValueError(
            f"{len(bad)} LBM nodes are outside the WRF d03 horizontal grid "
            f"(x∈[0, {x_max:.1f}] m, y∈[0, {y_max:.1f}] m). "
            f"First bad node: x={target_x[bad[0]]:.1f}, y={target_y[bad[0]]:.1f}"
        )

    # Grid indices (nearest-neighbour)
    j_idx = np.clip(np.round(target_y / dy_wrf).astype(int), 0, ny - 1)
    i_idx = np.clip(np.round(target_x / dx_wrf).astype(int), 0, nx - 1)

    values = np.empty(n_nodes, dtype=np.float64)
    warn_below = False
    warn_above = False

    for n in range(n_nodes):
        j, i = j_idx[n], i_idx[n]
        z_col = z_levels[:, j, i]          # (nk,) height profile at this column
        val, below, above = _interp_column_1d(field[:, j, i], z_col, target_z[n])
        values[n] = val
        warn_below = warn_below or below
        warn_above = warn_above or above

    if warn_below:
        warnings.warn(
            "Some LBM nodes are below the lowest WRF mass level (z < z_mass[0]). "
            "Clamping to lowest level value.",
            stacklevel=2,
        )
    if warn_above:
        warnings.warn(
            "Some LBM nodes are above the highest WRF mass level (z > z_mass[-1]). "
            "Clamping to top level value.",
            stacklevel=2,
        )

    return values


# ---------------------------------------------------------------------------
# Convenience: interpolate u, v, w, theta in one call
# ---------------------------------------------------------------------------

@dataclass
class WRFColumnData:
    """Container for per-step WRF fields needed for vertical interpolation.

    Attributes
    ----------
    ph_total     : (nz+1, ny, nx) m² s⁻²  total geopotential (phb + ph')
    u            : (nz, ny, nx)   m s⁻¹
    v            : (nz, ny, nx)   m s⁻¹
    w            : (nz+1, ny, nx) m s⁻¹   (w-level)
    theta        : (nz, ny, nx)   K        (WRF θ − 300, perturbation potential temp)
    terrain_height : (ny, nx)     m        from grid.terrain_height
    dx_wrf       : float          m
    dy_wrf       : float          m
    """

    ph_total: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    theta: np.ndarray
    terrain_height: np.ndarray
    dx_wrf: float
    dy_wrf: float


@dataclass
class InterpolatedFields:
    """Vertically interpolated WRF fields at LBM node positions."""

    u: np.ndarray      # (N,) m s⁻¹
    v: np.ndarray      # (N,) m s⁻¹
    w: np.ndarray      # (N,) m s⁻¹
    theta: np.ndarray  # (N,) K  (WRF perturbation potential temperature θ−300)


# ---------------------------------------------------------------------------
# MOST surface-layer extrapolation (Path 2)
# ---------------------------------------------------------------------------

class MOSTSurfaceFields(NamedTuple):
    """Per-column surface-layer fields needed for MOST extrapolation.

    All arrays are (ny, nx), already on the CPU (np.ndarray).

    Attributes
    ----------
    ustar      : friction velocity (m s⁻¹), from carry.diag.ustar
    roughness_m: aerodynamic roughness length z0 (m), from carry.diag.roughness_m
    psim       : dimensionless stability function for momentum ψ_m(z/L).
                 Computed by surface_layer_with_diagnostics() at each coupling
                 step, time-aligned with ustar (path 2: full MOST with stability).
                 psim > 0 in unstable conditions, psim < 0 in stable conditions.
    """

    ustar: np.ndarray       # (ny, nx)  m s⁻¹
    roughness_m: np.ndarray  # (ny, nx)  m
    psim: np.ndarray        # (ny, nx)  dimensionless


def most_extrapolate(
    z_agl: float,
    ustar: float,
    z0: float,
    psim: float,
    kappa: float = _KAPPA,
) -> float:
    """Monin-Obukhov similarity extrapolation for wind speed below z_mass[0].

    Path 2: complete MOST with stability correction.
    psim is sourced from surface_layer_with_diagnostics(), time-aligned with
    ustar from the same carry. Only applied for nodes where z_agl < z_mass[0].

    Formula
    -------
    U(z) = (u* / κ) × [ln(z / z0) - ψ_m(z/L)]

    Derivation: MOST log-linear velocity profile (Monin & Obukhov 1954).
    ψ_m > 0 (unstable): wind increases slower than log → U < log profile
    ψ_m < 0 (stable):   wind increases faster than log → U > log profile

    Parameters
    ----------
    z_agl   : float  height above ground level (m); must be > z0
    ustar   : float  friction velocity (m s⁻¹); must be > 0
    z0      : float  roughness length (m); must be > 0
    psim    : float  stability function ψ_m evaluated at z/L
    kappa   : float  von Kármán constant (default 0.4)

    Returns
    -------
    U_mag : float  wind speed magnitude (m s⁻¹); always >= 0

    Notes
    -----
    - Caller is responsible for z_agl > z0 (raises ValueError otherwise).
    - Returns 0.0 if ustar <= 0 (calm / numerical zero).
    - NaN/inf in psim → fallback to neutral (psim = 0) with warning.
    """
    if z_agl <= z0:
        raise ValueError(
            f"MOST extrapolation requires z_agl ({z_agl:.4f} m) > z0 ({z0:.4f} m). "
            "LBM node is below or at the roughness height."
        )
    if ustar <= 0.0:
        return 0.0
    if not np.isfinite(psim):
        warnings.warn(
            f"psim is not finite ({psim!r}); falling back to neutral (psim=0).",
            stacklevel=2,
        )
        psim = 0.0
    log_term = np.log(z_agl / z0)
    u_mag = (ustar / kappa) * (log_term - psim)
    return float(max(u_mag, 0.0))


def _most_extrapolate_uv(
    z_agl: float,
    ustar: float,
    z0: float,
    psim: float,
    u_ref: float,
    v_ref: float,
) -> tuple[float, float]:
    """Apply MOST to get u, v at z_agl using direction from (u_ref, v_ref).

    The wind direction is taken from (u_ref, v_ref) (first WRF mass level).
    Only the magnitude is replaced by the MOST estimate. Static-wind
    protection: if |u_ref, v_ref| < 0.1 m/s, returns (0, 0).

    Parameters
    ----------
    z_agl            : float  height AGL of the LBM node (m)
    ustar, z0, psim  : MOST parameters
    u_ref, v_ref     : float  WRF wind at lowest mass level (m s⁻¹)

    Returns
    -------
    u_extrap, v_extrap : float  MOST-extrapolated velocity components (m s⁻¹)
    """
    spd_ref = np.hypot(u_ref, v_ref)
    if spd_ref < 0.1:
        # Static-wind protection: return calm
        return 0.0, 0.0
    u_mag = most_extrapolate(z_agl, ustar, z0, psim)
    scale = u_mag / spd_ref
    return float(u_ref * scale), float(v_ref * scale)


def interp_wrf_to_lbm_nodes(
    wrf_data: WRFColumnData,
    target_z: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    most_fields: MOSTSurfaceFields | None = None,
) -> InterpolatedFields:
    """Interpolate u, v, w, theta to LBM node positions.

    Parameters
    ----------
    wrf_data    : WRFColumnData  per-step WRF fields
    target_z    : (N,) m  LBM node heights above MSL
    target_x    : (N,) m  LBM node x in WRF grid coords (m from SW corner)
    target_y    : (N,) m  LBM node y in WRF grid coords (m from SW corner)
    most_fields : MOSTSurfaceFields | None
        When provided, nodes whose z_agl < z_mass[0] use MOST extrapolation
        (path 2: full MOST with stability correction) instead of clamping.
        ustar and psim must be time-aligned (both from the same carry snapshot).
        If None, falls back to the original clamping behaviour.

    Returns
    -------
    InterpolatedFields  with u, v, w, theta at each LBM node

    Notes
    -----
    - u, v, theta are mass-level fields → use z_mass for interpolation
    - w is a w-level field → use z_w for interpolation
    - theta is passed through unchanged (WRF θ−300 = LBM concentration,
      assuming para.set_reference_temperature(300.0) on the LBM side)
    - MOST extrapolation uses nearest-neighbour horizontal lookup (same as
      the normal interpolation path) for ustar, z0, psim.
    """
    z_w = compute_z_w(wrf_data.ph_total)
    z_mass = compute_z_mass(z_w)

    kwargs = dict(
        target_z=target_z,
        target_x=target_x,
        target_y=target_y,
        terrain_height=wrf_data.terrain_height,
        dx_wrf=wrf_data.dx_wrf,
        dy_wrf=wrf_data.dy_wrf,
    )

    # Standard interpolation (clamping for below-floor nodes)
    u_out = interp_field_to_heights(wrf_data.u, z_mass, **kwargs)
    v_out = interp_field_to_heights(wrf_data.v, z_mass, **kwargs)
    w_out = interp_field_to_heights(wrf_data.w, z_w, **kwargs)
    theta_out = interp_field_to_heights(wrf_data.theta, z_mass, **kwargs)

    # Path 2: MOST extrapolation for nodes below z_mass[0]
    if most_fields is not None:
        u_out, v_out = _apply_most_to_uv(
            u_out, v_out,
            wrf_data, z_mass, z_w,
            target_z, target_x, target_y,
            most_fields,
        )

    return InterpolatedFields(u=u_out, v=v_out, w=w_out, theta=theta_out)


def _apply_most_to_uv(
    u_interp: np.ndarray,
    v_interp: np.ndarray,
    wrf_data: WRFColumnData,
    z_mass: np.ndarray,
    z_w: np.ndarray,
    target_z: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    most_fields: MOSTSurfaceFields,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace clamped u/v with MOST extrapolation for below-floor nodes.

    Path 2: complete MOST with stability correction.
    psim is time-aligned with ustar (both from the same carry snapshot).
    Only nodes with z_agl < z_mass[0, j, i] are replaced; all others keep
    the standard interpolated value from the caller.

    Horizontal lookup uses nearest-neighbour (same convention as
    interp_field_to_heights) for ustar, z0, psim, and the WRF first-layer
    wind direction (u[0,j,i], v[0,j,i]).

    Returns
    -------
    u_out, v_out : (N,) arrays with MOST values patched in where applicable
    """
    target_z = np.asarray(target_z, dtype=np.float64)
    target_x = np.asarray(target_x, dtype=np.float64)
    target_y = np.asarray(target_y, dtype=np.float64)

    _, ny, nx = wrf_data.u.shape
    dx_wrf = wrf_data.dx_wrf
    dy_wrf = wrf_data.dy_wrf
    n_nodes = len(target_z)

    # Nearest-neighbour horizontal indices (same as interp_field_to_heights)
    j_idx = np.clip(np.round(target_y / dy_wrf).astype(int), 0, ny - 1)
    i_idx = np.clip(np.round(target_x / dx_wrf).astype(int), 0, nx - 1)

    ustar_field = np.asarray(most_fields.ustar, dtype=np.float64)
    z0_field = np.asarray(most_fields.roughness_m, dtype=np.float64)
    psim_field = np.asarray(most_fields.psim, dtype=np.float64)
    terrain = np.asarray(wrf_data.terrain_height, dtype=np.float64)

    u_out = u_interp.copy()
    v_out = v_interp.copy()
    n_most = 0

    for n in range(n_nodes):
        j, i = j_idx[n], i_idx[n]
        z_floor = z_mass[0, j, i]   # lowest WRF mass level height (MSL)
        if target_z[n] >= z_floor:
            continue  # normal interpolation region — keep as-is

        # This node is below the lowest WRF mass level: apply MOST
        z_agl = target_z[n] - terrain[j, i]
        z0 = float(z0_field[j, i])
        ustar = float(ustar_field[j, i])
        psim = float(psim_field[j, i])

        # Direction from WRF first mass-level wind (destaggered)
        u_ref = float(wrf_data.u[0, j, i])
        v_ref = float(wrf_data.v[0, j, i])

        if z_agl <= z0:
            # Node is at or below the roughness height — clamp to 0
            u_out[n] = 0.0
            v_out[n] = 0.0
        else:
            u_e, v_e = _most_extrapolate_uv(z_agl, ustar, z0, psim, u_ref, v_ref)
            u_out[n] = u_e
            v_out[n] = v_e
        n_most += 1

    if n_most > 0:
        warnings.warn(
            f"MOST extrapolation (path 2) applied to {n_most}/{n_nodes} LBM nodes "
            "below z_mass[0]. psim from surface_layer_with_diagnostics(), "
            "time-aligned with ustar.",
            stacklevel=3,
        )

    return u_out, v_out


# ---------------------------------------------------------------------------
# Bilinear horizontal interpolation (per-column vertical interpolation in AGL)
# ---------------------------------------------------------------------------
#
# WRF d03 is coarse (3 km) relative to the LBM domain (a few km), so
# nearest-neighbour lookup makes the LBM boundary piecewise constant with
# jumps of several m/s along WRF column edges.  The functions below:
#   1. interpolate each WRF column vertically in height above ground (AGL),
#      so the LBM's flat ground (z=0) maps to every column's own terrain;
#   2. apply MOST below each column's lowest mass level (per column);
#   3. blend the four surrounding columns with bilinear weights.
# Only a handful of WRF columns touch the LBM domain, so the work is
# vectorised per column (np.interp over all nodes using that column).


def destagger_uv(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move WRF Arakawa-C u (…, ny, nx+1) and v (…, ny+1, nx) to mass points.

    Arrays already on the mass grid (same trailing shape for u and v) are
    returned unchanged.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape == v.shape:
        return u, v
    if u.shape[-1] == v.shape[-1] + 1 and v.shape[-2] == u.shape[-2] + 1:
        u_m = 0.5 * (u[..., :-1] + u[..., 1:])
        v_m = 0.5 * (v[..., :-1, :] + v[..., 1:, :])
        return u_m, v_m
    raise ValueError(f"Unexpected u/v shapes for destaggering: u{u.shape}, v{v.shape}")


def _bilinear_corners(target_x, target_y, ny, nx, dx_wrf, dy_wrf):
    """Return [(j, i, weight)] for the 4 mass-grid columns around each node.

    Mass point (j, i) sits at (i*dx, j*dy) measured from the SW mass point.
    """
    target_x = np.asarray(target_x, dtype=np.float64)
    target_y = np.asarray(target_y, dtype=np.float64)
    x_max = (nx - 1) * dx_wrf
    y_max = (ny - 1) * dy_wrf
    out_of_bounds = (target_x < 0) | (target_x > x_max) | (target_y < 0) | (target_y > y_max)
    if out_of_bounds.any():
        bad = np.where(out_of_bounds)[0]
        raise ValueError(
            f"{len(bad)} LBM nodes are outside the WRF d03 horizontal grid "
            f"(x∈[0, {x_max:.1f}] m, y∈[0, {y_max:.1f}] m). "
            f"First bad node: x={target_x[bad[0]]:.1f}, y={target_y[bad[0]]:.1f}"
        )
    fx = target_x / dx_wrf
    fy = target_y / dy_wrf
    i0 = np.clip(np.floor(fx).astype(int), 0, max(nx - 2, 0))
    j0 = np.clip(np.floor(fy).astype(int), 0, max(ny - 2, 0))
    tx = np.clip(fx - i0, 0.0, 1.0)
    ty = np.clip(fy - j0, 0.0, 1.0)
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    return [
        (j0, i0, (1.0 - ty) * (1.0 - tx)),
        (j0, i1, (1.0 - ty) * tx),
        (j1, i0, ty * (1.0 - tx)),
        (j1, i1, ty * tx),
    ]


class _ColumnSource:
    """Pre-processed WRF fields for per-column AGL interpolation."""

    def __init__(self, wrf_data: WRFColumnData, most_fields: MOSTSurfaceFields | None,
                 kappa: float = _KAPPA):
        self.terrain = np.asarray(wrf_data.terrain_height, dtype=np.float64)
        self.ny, self.nx = self.terrain.shape
        z_w = compute_z_w(wrf_data.ph_total)
        z_mass = compute_z_mass(z_w)
        self.z_w_agl = z_w - self.terrain[None]
        self.z_mass_agl = z_mass - self.terrain[None]
        self.u, self.v = destagger_uv(wrf_data.u, wrf_data.v)
        self.w = np.asarray(wrf_data.w, dtype=np.float64)
        self.theta = np.asarray(wrf_data.theta, dtype=np.float64)
        for name, arr in (("u", self.u), ("v", self.v), ("theta", self.theta)):
            if arr.shape[1:] != (self.ny, self.nx):
                raise ValueError(f"{name} shape {arr.shape} does not match terrain {self.terrain.shape}")
        if self.w.shape[1:] != (self.ny, self.nx):
            raise ValueError(f"w shape {self.w.shape} does not match terrain {self.terrain.shape}")
        self.most = most_fields
        self.kappa = kappa

    def column(self, j: int, i: int, z_agl: np.ndarray):
        """u, v, w, theta of column (j, i) at heights z_agl (m AGL), vectorised.

        Out-of-range heights are clamped to the end values (np.interp); below
        the lowest mass level u/v use MOST when surface fields are available.
        """
        zm = self.z_mass_agl[:, j, i]
        zw = self.z_w_agl[:, j, i]
        u = np.interp(z_agl, zm, self.u[:, j, i])
        v = np.interp(z_agl, zm, self.v[:, j, i])
        w = np.interp(z_agl, zw, self.w[:, j, i])
        theta = np.interp(z_agl, zm, self.theta[:, j, i])

        if self.most is not None:
            below = z_agl < zm[0]
            if below.any():
                u1, v1 = self.u[0, j, i], self.v[0, j, i]
                spd = float(np.hypot(u1, v1))
                if spd < 0.1:
                    u[below] = 0.0  # static-wind protection (as _most_extrapolate_uv)
                    v[below] = 0.0
                else:
                    z0 = float(self.most.roughness_m[j, i])
                    ustar = float(self.most.ustar[j, i])
                    psim = float(self.most.psim[j, i])
                    if not np.isfinite(psim):
                        psim = 0.0
                    zb = z_agl[below]
                    if ustar <= 0.0 or z0 <= 0.0:
                        mag = np.zeros_like(zb)
                    else:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            mag = (ustar / self.kappa) * (np.log(zb / z0) - psim)
                        mag = np.where(zb > z0, np.maximum(mag, 0.0), 0.0)
                    u[below] = u1 * mag / spd
                    v[below] = v1 * mag / spd
        return u, v, w, theta


def interp_wrf_to_lbm_nodes_bilinear(
    wrf_data: WRFColumnData,
    target_x: np.ndarray,
    target_y: np.ndarray,
    target_z_agl: np.ndarray,
    most_fields: MOSTSurfaceFields | None = None,
) -> InterpolatedFields:
    """Interpolate u, v, w, theta to LBM nodes: bilinear in x/y, linear in z (AGL).

    Parameters
    ----------
    wrf_data     : WRFColumnData  (u/v may be staggered; they are destaggered here)
    target_x/y   : (N,) m  node positions in WRF grid coords (m from SW mass point)
    target_z_agl : (N,) m  node heights above ground
    most_fields  : optional MOST surface fields for nodes below the first mass level

    Returns
    -------
    InterpolatedFields  u, v, w (m/s) and theta (K, full) at each node
    """
    src = _ColumnSource(wrf_data, most_fields)
    target_z_agl = np.asarray(target_z_agl, dtype=np.float64)
    n = target_z_agl.shape[0]
    out = [np.zeros(n) for _ in range(4)]

    for jc, ic, wc in _bilinear_corners(target_x, target_y, src.ny, src.nx,
                                        wrf_data.dx_wrf, wrf_data.dy_wrf):
        key = jc * src.nx + ic
        for k in np.unique(key):
            m = key == k
            j, i = divmod(int(k), src.nx)
            vals = src.column(j, i, target_z_agl[m])
            wm = wc[m]
            for o, val in zip(out, vals):
                o[m] += wm * val

    return InterpolatedFields(u=out[0], v=out[1], w=out[2], theta=out[3])


def bilinear_terrain(terrain_height: np.ndarray, target_x, target_y,
                     dx_wrf: float, dy_wrf: float) -> np.ndarray:
    """Bilinearly interpolated terrain height (m MSL) at node positions."""
    terrain = np.asarray(terrain_height, dtype=np.float64)
    ny, nx = terrain.shape
    out = np.zeros(np.asarray(target_x).shape[0])
    for jc, ic, wc in _bilinear_corners(target_x, target_y, ny, nx, dx_wrf, dy_wrf):
        out += wc * terrain[jc, ic]
    return out


def build_lbm_grid_table(
    wrf_data: WRFColumnData,
    x_wrf: np.ndarray,
    y_wrf: np.ndarray,
    z_agl: np.ndarray,
    most_fields: MOSTSurfaceFields | None = None,
    fields: tuple[str, ...] = ("u", "v", "w"),
) -> tuple[np.ndarray, ...]:
    """Fields on a tensor-product LBM grid, shape (ny_l, nx_l, nz_l), float32.

    ``fields`` selects any of "u", "v", "w", "theta" (default u, v, w); tables are
    returned in that order.

    Same interpolation as :func:`interp_wrf_to_lbm_nodes_bilinear`, exploiting
    that the LBM grid is separable: column profiles are evaluated once on the
    1-D z_agl levels and blended with 2-D bilinear weights.

    Parameters
    ----------
    x_wrf : (nx_l,) m  LBM grid x positions in WRF coords
    y_wrf : (ny_l,) m  LBM grid y positions in WRF coords
    z_agl : (nz_l,) m  LBM grid heights above ground
    """
    src = _ColumnSource(wrf_data, most_fields)
    x_wrf = np.asarray(x_wrf, dtype=np.float64)
    y_wrf = np.asarray(y_wrf, dtype=np.float64)
    z_agl = np.asarray(z_agl, dtype=np.float64)

    # Separable corners: run the 2-D corner helper on the x and y axes separately
    cx = _bilinear_corners(x_wrf, np.zeros_like(x_wrf), src.ny, src.nx, wrf_data.dx_wrf, wrf_data.dy_wrf)
    cy = _bilinear_corners(np.zeros_like(y_wrf), y_wrf, src.ny, src.nx, wrf_data.dx_wrf, wrf_data.dy_wrf)
    i_lo, i_hi = cx[0][1], cx[1][1]
    wx_lo, wx_hi = cx[0][2] + cx[2][2], cx[1][2] + cx[3][2]   # y-weights are 1/0 at y=0
    j_lo, j_hi = cy[0][0], cy[2][0]
    wy_lo, wy_hi = cy[0][2] + cy[1][2], cy[2][2] + cy[3][2]

    names = ("u", "v", "w", "theta")
    sel = [names.index(f) for f in fields]
    profiles = {}

    def prof(j, i):
        if (j, i) not in profiles:
            cols = src.column(j, i, z_agl.copy())
            profiles[(j, i)] = np.stack([cols[c] for c in sel]).astype(np.float32)  # (n_fields, nz_l)
        return profiles[(j, i)]

    shape = (y_wrf.size, x_wrf.size, z_agl.size)
    tables = [np.zeros(shape, dtype=np.float32) for _ in sel]
    for jj, wy in ((j_lo, wy_lo), (j_hi, wy_hi)):
        for ii, wx in ((i_lo, wx_lo), (i_hi, wx_hi)):
            weight = (wy[:, None] * wx[None, :]).astype(np.float32)        # (ny_l, nx_l)
            key = jj[:, None] * src.nx + ii[None, :]                        # (ny_l, nx_l)
            for k in np.unique(key):
                m = key == k
                j, i = divmod(int(k), src.nx)
                p = prof(j, i)
                for c in range(len(sel)):
                    tables[c][m] += weight[m][:, None] * p[c][None, :]
    return tuple(tables)
