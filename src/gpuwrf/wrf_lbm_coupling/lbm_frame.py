"""LBMFrame: assembled per-step data packet ready for LBM boundary update.

Combines vertical interpolation (eta→z) and unit conversion (WRF→LBM lattice)
into a single call.  All heavy lifting delegates to vertical_interp and
unit_convert; this module only orchestrates.

Usage
-----
Static fields (sent once at init):
    Build an LBMStaticContext once from wrfinput data and the LBM para object.

Dynamic frames (sent every WRF step, dt=5 s):
    Call build_lbm_frame() with the current carry fields.

Example::

    ctx = LBMStaticContext(
        phb=carry.base_state.phb,
        terrain_height=np.asarray(grid.terrain_height),
        eta_levels=np.asarray(grid.eta_levels),
        dx_wrf=dx_d03,
        dy_wrf=dy_d03,
        ref=LBMReference(dx_lbm=dx_lbm, dt_lbm=dt_lbm),
    )
    lbm_nodes = ctx.get_lbm_node_coords()   # call once, cache result

    frame = build_lbm_frame(
        t_wrf=t_wrf,
        ph_total=np.asarray(carry.state.ph_total),
        u=np.asarray(carry.state.u),
        v=np.asarray(carry.state.v),
        w=np.asarray(carry.state.w),
        theta=np.asarray(carry.state.theta),
        ctx=ctx,
        lbm_nodes=lbm_nodes,
    )
    # frame.u, frame.v, frame.w  → dimensionless LBM lattice velocities
    # frame.theta                → K deviation (= WRF theta, no conversion)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpuwrf.wrf_lbm_coupling.unit_convert import LBMReference, theta_to_lbm, uvw_to_lbm
from gpuwrf.wrf_lbm_coupling.vertical_interp import (
    WRFColumnData,
    InterpolatedFields,
    interp_wrf_to_lbm_nodes,
)


# ---------------------------------------------------------------------------
# Static context (initialised once per run)
# ---------------------------------------------------------------------------

@dataclass
class LBMStaticContext:
    """Per-domain static fields needed for every frame build.

    These fields do not change during the simulation and are sent to the LBM
    side once during the handshake phase.

    Attributes
    ----------
    phb           : (nz+1, ny, nx)  m² s⁻²  base-state geopotential (static)
    terrain_height : (ny, nx)       m        from grid.terrain_height
    eta_levels    : (nz+1,)                  ZNW from grid.eta_levels (1.0→0.0)
    dx_wrf        : float           m        d03 grid spacing in x
    dy_wrf        : float           m        d03 grid spacing in y
    ref           : LBMReference             unit conversion parameters
    lbm_inlet_x   : (N,) optional   m        LBM inlet node x coords (WRF frame)
    lbm_inlet_y   : (N,) optional   m        LBM inlet node y coords (WRF frame)
    lbm_inlet_z   : (N,) optional   m        LBM inlet node z coords (above MSL)
    """

    phb: np.ndarray
    terrain_height: np.ndarray
    eta_levels: np.ndarray
    dx_wrf: float
    dy_wrf: float
    ref: LBMReference
    lbm_inlet_x: np.ndarray | None = None
    lbm_inlet_y: np.ndarray | None = None
    lbm_inlet_z: np.ndarray | None = None

    def set_lbm_inlet_coords(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> None:
        """Register LBM inlet node coordinates in WRF-grid physical space.

        Parameters
        ----------
        x, y : (N,) m  horizontal position from WRF d03 SW corner
        z    : (N,) m  height above MSL
        """
        self.lbm_inlet_x = np.asarray(x, dtype=np.float64)
        self.lbm_inlet_y = np.asarray(y, dtype=np.float64)
        self.lbm_inlet_z = np.asarray(z, dtype=np.float64)

    @property
    def n_nodes(self) -> int:
        """Number of registered LBM inlet nodes."""
        if self.lbm_inlet_x is None:
            return 0
        return len(self.lbm_inlet_x)

    def to_dict(self) -> dict:
        """Serialise static fields for MPI send (numpy arrays only)."""
        _, ny, nx = self.phb.shape
        nz_plus1 = self.phb.shape[0]
        return {
            "phb": self.phb.astype(np.float32),
            "terrain_height": self.terrain_height.astype(np.float32),
            "eta_levels": self.eta_levels.astype(np.float64),
            "dx_wrf": float(self.dx_wrf),
            "dy_wrf": float(self.dy_wrf),
            "nx": nx,
            "ny": ny,
            "nz": nz_plus1 - 1,
            "u_ref": float(self.ref.u_ref),
            "theta_ref": float(self.ref.theta_ref),
        }


# ---------------------------------------------------------------------------
# Per-step frame
# ---------------------------------------------------------------------------

@dataclass
class LBMFrame:
    """Per-step WRF→LBM data packet at LBM inlet node positions.

    Attributes
    ----------
    t_wrf  : float     WRF simulation time (s from forecast start)
    u, v, w : (N,)     dimensionless LBM lattice velocities
    theta  : (N,)      K deviation (= WRF theta, passed through unchanged)
    coords : (N, 3)    physical (x, y, z) of LBM nodes in WRF grid frame (m)
    ref    : LBMReference
    """

    t_wrf: float
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    theta: np.ndarray
    coords: np.ndarray   # (N, 3)
    ref: LBMReference

    def to_dict(self) -> dict:
        """Serialise to plain dict of numpy arrays for MPI send."""
        return {
            "t_wrf": float(self.t_wrf),
            "u": self.u.astype(np.float32),
            "v": self.v.astype(np.float32),
            "w": self.w.astype(np.float32),
            "theta": self.theta.astype(np.float32),
            "coords": self.coords.astype(np.float32),
        }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_lbm_frame(
    t_wrf: float,
    ph_total: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    theta: np.ndarray,
    ctx: LBMStaticContext,
) -> LBMFrame:
    """Build one LBM data frame from the current WRF carry fields.

    Parameters
    ----------
    t_wrf    : float  WRF time in seconds from forecast start
    ph_total : (nz+1, ny, nx)  m² s⁻²  carry.state.ph_total (= phb + ph')
    u        : (nz, ny, nx)    m s⁻¹   carry.state.u
    v        : (nz, ny, nx)    m s⁻¹   carry.state.v
    w        : (nz+1, ny, nx)  m s⁻¹   carry.state.w
    theta    : (nz, ny, nx)    K       carry.state.theta  (= θ_full − 300)
    ctx      : LBMStaticContext         static context (phb, terrain, ref, …)

    Returns
    -------
    LBMFrame with u/v/w in LBM lattice units, theta in K deviation.

    Raises
    ------
    RuntimeError  if LBM inlet coordinates have not been registered via
                  ctx.set_lbm_inlet_coords().
    """
    if ctx.lbm_inlet_x is None:
        raise RuntimeError(
            "LBM inlet node coordinates are not set. "
            "Call ctx.set_lbm_inlet_coords(x, y, z) before building frames."
        )

    wrf_data = WRFColumnData(
        ph_total=np.asarray(ph_total, dtype=np.float64),
        u=np.asarray(u, dtype=np.float64),
        v=np.asarray(v, dtype=np.float64),
        w=np.asarray(w, dtype=np.float64),
        theta=np.asarray(theta, dtype=np.float64),
        terrain_height=np.asarray(ctx.terrain_height, dtype=np.float64),
        dx_wrf=ctx.dx_wrf,
        dy_wrf=ctx.dy_wrf,
    )

    interp: InterpolatedFields = interp_wrf_to_lbm_nodes(
        wrf_data,
        target_z=ctx.lbm_inlet_z,
        target_x=ctx.lbm_inlet_x,
        target_y=ctx.lbm_inlet_y,
    )

    ref = ctx.ref
    coords = np.stack([ctx.lbm_inlet_x, ctx.lbm_inlet_y, ctx.lbm_inlet_z], axis=1)

    return LBMFrame(
        t_wrf=float(t_wrf),
        u=uvw_to_lbm(interp.u, ref),
        v=uvw_to_lbm(interp.v, ref),
        w=uvw_to_lbm(interp.w, ref),
        theta=theta_to_lbm(interp.theta, ref),
        coords=coords,
        ref=ref,
    )
