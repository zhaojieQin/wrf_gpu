"""Surface-layer diagnostics for the WRF→LBM coupling layer.

Recomputes WRF surface-layer diagnostics (ustar, z0, psim) at each coupling
step using surface_layer_with_diagnostics() so that psim is time-aligned with
ustar (both from the same carry snapshot, Path 2 MOST).

All outputs are CPU numpy arrays, ready for MOSTSurfaceFields.

Design
------
- Calls the public ``get_surface_diagnostics`` wrapper in physics_couplers.py,
  which runs _surface_column_view → surface_layer_with_diagnostics internally.
- JAX → numpy transfer: one tuple dispatch via jax.device_get() to minimise
  round-trips (3 fields in one call).
- Optional debug assertions check that recomputed ustar/z0 are consistent with
  carry.diag values (catches timing or state-mismatch bugs early).

Field mapping (SurfaceLayerDiagnostics → MOSTSurfaceFields)
-----------------------------------------------------------
    ustar       ← diag.fluxes.ustar   (m s⁻¹, friction velocity)
    roughness_m ← diag.znt            (m,    aerodynamic roughness length z0)
    psim        ← diag.psim           (dimensionless, stability function)
"""

from __future__ import annotations

import logging
import warnings

import jax
import numpy as np

logger = logging.getLogger(__name__)

from gpuwrf.coupling.physics_couplers import get_surface_diagnostics
from gpuwrf.wrf_lbm_coupling.vertical_interp import MOSTSurfaceFields

# Relative tolerance for debug consistency checks against carry.diag
_RTOL_CHECK = 1e-5


def compute_surface_diagnostics(carry, grid) -> MOSTSurfaceFields:
    """Recompute surface-layer diagnostics and return as MOSTSurfaceFields.

    Calls surface_layer_with_diagnostics() on the current carry state so that
    psim, ustar, and z0 are all derived from the same physics snapshot.  This
    is Path 2 MOST: full stability correction, time-aligned fields.

    Parameters
    ----------
    carry : Carry  WRF model carry (post-surface-adapter so ustar is written)
    grid  : GridSpec  domain grid for theta_m / hydrostatic-p corrections

    Returns
    -------
    MOSTSurfaceFields  with ustar, roughness_m, psim as (ny, nx) numpy arrays

    Notes
    -----
    - Called once per coupling step (every ~5 s); D→H transfer is ~42 KB for
      a 42×42 domain, negligible overhead.
    - In debug mode (__debug__ = True, i.e. python without -O), asserts that
      the recomputed ustar and z0 match carry.diag values.  A mismatch
      indicates a state-timing bug; the assertion failure should NOT be
      silenced.
    """
    diag = get_surface_diagnostics(carry.state, grid)

    # Transfer only the 3 fields we need — one dispatch, not three
    ustar_np, z0_np, psim_np = jax.device_get(
        (diag.fluxes.ustar, diag.znt, diag.psim)
    )
    ustar_np = np.asarray(ustar_np, dtype=np.float64)
    z0_np    = np.asarray(z0_np,    dtype=np.float64)
    psim_np  = np.asarray(psim_np,  dtype=np.float64)

    # NaN guard for psim: replace with 0 (neutral fallback) and log
    n_nan = int(np.isnan(psim_np).sum())
    if n_nan > 0:
        logger.warning(
            "compute_surface_diagnostics: psim contains %d NaN value(s) "
            "(out of %d grid points); replaced with 0 (neutral MOST).",
            n_nan, psim_np.size,
        )
        psim_np = np.where(np.isnan(psim_np), 0.0, psim_np)

    # Debug consistency check: recomputed values should agree with carry.diag
    if __debug__:
        _check_consistency(ustar_np, z0_np, carry)

    return MOSTSurfaceFields(ustar=ustar_np, roughness_m=z0_np, psim=psim_np)


def _check_consistency(ustar_np: np.ndarray, z0_np: np.ndarray, carry) -> None:
    """Assert recomputed fields agree with carry.diag (debug mode only).

    Mismatches typically indicate that the callback is running against a carry
    from the wrong physics stage (e.g. pre-surface-adapter).

    Raises
    ------
    AssertionError  if ustar or roughness_m differ by more than _RTOL_CHECK.
    """
    try:
        diag_ustar = np.asarray(carry.diag.ustar, dtype=np.float64)
        diag_z0    = np.asarray(carry.diag.roughness_m, dtype=np.float64)
    except AttributeError:
        # carry.diag may not exist in test mocks; skip silently
        return

    try:
        np.testing.assert_allclose(
            ustar_np, diag_ustar, rtol=_RTOL_CHECK,
            err_msg=(
                "compute_surface_diagnostics: recomputed ustar differs from "
                "carry.diag.ustar — possible state-timing mismatch."
            ),
        )
        np.testing.assert_allclose(
            z0_np, diag_z0, rtol=_RTOL_CHECK,
            err_msg=(
                "compute_surface_diagnostics: recomputed z0 (znt) differs "
                "from carry.diag.roughness_m — possible state-timing mismatch."
            ),
        )
    except AssertionError as exc:
        # Re-raise with a clear marker so it is not confused with test failures
        raise AssertionError(
            f"[surface_diagnostics consistency check FAILED]\n{exc}"
        ) from exc
