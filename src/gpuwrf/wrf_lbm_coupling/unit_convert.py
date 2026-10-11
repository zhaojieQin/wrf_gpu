"""WRF physical quantities → LBM lattice units conversion.

All functions are pure (no side effects). Inputs/outputs are numpy arrays.

Unit conventions
----------------
WRF side:
  u, v, w : m s⁻¹
  theta    : K  (WRF perturbation potential temperature θ − 300,
             i.e. θ_full − t0 where t0 = 300 K)

LBM side (VirtualFluids):
  velocity : dimensionless lattice units  u_LB = u_phys / u_ref
             where u_ref = dx_lbm / dt_lbm  (= para.get_velocity_ratio())
  theta    : K deviation from T_ref_LBM
             passed through unchanged when T_ref_LBM = 300 K = WRF t0

References
----------
VirtualFluids Parameter.h:778  u0_ratio = dx / dt
StratifiedBoundaryLayer.cpp:264  para.setVelocityRatio(deltaX / deltaT)
StratifiedBoundaryLayer.cpp:106  concentration = T_phys - T_ref (K deviation)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LBMReference:
    """Reference quantities for WRF → LBM unit conversion.

    Attributes
    ----------
    dx_lbm : float
        LBM grid spacing (m).
    dt_lbm : float
        LBM time step (s).
    u_ref : float
        Reference velocity = dx_lbm / dt_lbm (m s⁻¹).
        Equals para.get_velocity_ratio() in VirtualFluids.
    theta_ref : float
        LBM reference potential temperature (K).
        Must match para.set_reference_temperature() on the LBM side.
        Default 300.0 K = WRF t0, so that WRF theta (θ−300) passes
        through to LBM concentration unchanged.
    """

    dx_lbm: float
    dt_lbm: float
    theta_ref: float = field(default=300.0)

    def __post_init__(self) -> None:
        if self.dt_lbm <= 0:
            raise ValueError(f"dt_lbm must be positive, got {self.dt_lbm}")
        if self.dx_lbm <= 0:
            raise ValueError(f"dx_lbm must be positive, got {self.dx_lbm}")

    @property
    def u_ref(self) -> float:
        """Reference velocity dx_lbm / dt_lbm (m s⁻¹)."""
        return self.dx_lbm / self.dt_lbm

    @classmethod
    def from_velocity_ratio(
        cls,
        velocity_ratio: float,
        dx_lbm: float,
        dt_lbm: float,
        theta_ref: float = 300.0,
    ) -> "LBMReference":
        """Construct from para.get_velocity_ratio().

        Verifies that velocity_ratio == dx_lbm / dt_lbm within 0.1 %.

        Parameters
        ----------
        velocity_ratio : float  para.get_velocity_ratio()  (m s⁻¹)
        dx_lbm         : float  LBM grid spacing (m)
        dt_lbm         : float  LBM time step (s)
        theta_ref      : float  (default 300.0 K)
        """
        ref = cls(dx_lbm=dx_lbm, dt_lbm=dt_lbm, theta_ref=theta_ref)
        expected = dx_lbm / dt_lbm
        if abs(velocity_ratio - expected) / expected > 1e-3:
            raise ValueError(
                f"velocity_ratio={velocity_ratio} does not match "
                f"dx_lbm/dt_lbm={expected:.6g}. "
                "Check LBM para setup."
            )
        return ref


# ---------------------------------------------------------------------------
# Velocity conversion
# ---------------------------------------------------------------------------

def uvw_to_lbm(u_phys: np.ndarray, ref: LBMReference) -> np.ndarray:
    """Convert physical velocity (m s⁻¹) to LBM lattice units.

    Derivation
    ----------
    [lattice] = [m s⁻¹] / [m s⁻¹]   (dimensionless)
    u_LB = u_phys / u_ref
    where u_ref = dx_lbm / dt_lbm

    Parameters
    ----------
    u_phys : ndarray  any shape, m s⁻¹
    ref    : LBMReference

    Returns
    -------
    u_lb : ndarray  same shape, dimensionless lattice velocity
    """
    return np.asarray(u_phys) / ref.u_ref


def lbm_to_uvw(u_lb: np.ndarray, ref: LBMReference) -> np.ndarray:
    """Inverse of :func:`uvw_to_lbm`: LBM units → m s⁻¹."""
    return np.asarray(u_lb) * ref.u_ref


# ---------------------------------------------------------------------------
# Potential temperature conversion
# ---------------------------------------------------------------------------

def theta_to_lbm(theta_wrf: np.ndarray, ref: LBMReference) -> np.ndarray:  # noqa: ARG001
    """Pass WRF perturbation potential temperature to LBM concentration.

    Derivation
    ----------
    WRF stores:   theta_wrf = θ_full − t0  (t0 = 300 K)
    LBM stores:   concentration = T_phys − T_ref_LBM

    If T_ref_LBM = t0 = 300 K (set via para.set_reference_temperature(300.0)),
    then:
        concentration = θ_full − 300 = theta_wrf

    No arithmetic conversion is needed; the arrays are identical in value.

    The ``ref`` argument is accepted for API consistency and to allow future
    extension (e.g. if T_ref_LBM ≠ 300 K).

    Parameters
    ----------
    theta_wrf : ndarray  any shape, K (WRF perturbation potential temperature)
    ref       : LBMReference  (unused when theta_ref == 300.0)

    Returns
    -------
    theta_lb : ndarray  same shape and dtype, K deviation for LBM concentration
    """
    return np.asarray(theta_wrf)
