"""Unit tests for surface_diagnostics.py.

Tests use synthetic carry/grid mocks (no GPU, no real WRF state).
All assertions are pure-numpy.

Coverage
--------
  1. compute_surface_diagnostics returns MOSTSurfaceFields with correct dtype
  2. All three fields (ustar, roughness_m, psim) are numpy arrays, not JAX
  3. Output shapes match (ny, nx)
  4. Debug consistency check passes when carry.diag matches
  5. Debug consistency check raises AssertionError when carry.diag differs
  6. No carry.diag attribute → check is skipped silently
  7. get_surface_diagnostics wrapper is importable and callable
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_diag(ny: int = 4, nx: int = 5, ustar_val=0.3, z0_val=0.05, psim_val=-0.2):
    """Build a fake SurfaceLayerDiagnostics-like object with numpy arrays."""

    class FakeFluxes:
        ustar = np.full((ny, nx), ustar_val, dtype=np.float64)

    class FakeDiag:
        fluxes = FakeFluxes()
        znt = np.full((ny, nx), z0_val, dtype=np.float64)
        psim = np.full((ny, nx), psim_val, dtype=np.float64)

    return FakeDiag()


def _make_carry(ny: int = 4, nx: int = 5, ustar_val=0.3, z0_val=0.05):
    """Build a minimal fake carry with diag fields."""

    class FakeDiag:
        ustar = np.full((ny, nx), ustar_val, dtype=np.float64)
        roughness_m = np.full((ny, nx), z0_val, dtype=np.float64)

    class FakeCarry:
        state = MagicMock()
        diag = FakeDiag()

    return FakeCarry()


def _fake_jax_device_get(x):
    """Fake jax.device_get: returns the tuple unchanged (already numpy)."""
    if isinstance(x, tuple):
        return x
    return x


# ---------------------------------------------------------------------------
# Tests: return type and shape
# ---------------------------------------------------------------------------

class TestComputeSurfaceDiagnostics:
    """compute_surface_diagnostics returns well-formed MOSTSurfaceFields."""

    def _run(self, ny=4, nx=5, ustar_val=0.3, z0_val=0.05, psim_val=-0.2):
        """Run compute_surface_diagnostics with fully mocked internals."""
        from gpuwrf.wrf_lbm_coupling.surface_diagnostics import compute_surface_diagnostics

        fake_diag = _make_diag(ny, nx, ustar_val, z0_val, psim_val)
        carry = _make_carry(ny, nx, ustar_val, z0_val)
        grid = MagicMock()

        with (
            patch(
                "gpuwrf.wrf_lbm_coupling.surface_diagnostics.get_surface_diagnostics",
                return_value=fake_diag,
            ),
            patch(
                "gpuwrf.wrf_lbm_coupling.surface_diagnostics.jax.device_get",
                side_effect=_fake_jax_device_get,
            ),
        ):
            return compute_surface_diagnostics(carry, grid)

    def test_returns_most_surface_fields(self):
        from gpuwrf.wrf_lbm_coupling.vertical_interp import MOSTSurfaceFields

        result = self._run()
        assert isinstance(result, MOSTSurfaceFields)

    def test_fields_are_numpy_not_jax(self):
        result = self._run()
        for name, arr in [
            ("ustar", result.ustar),
            ("roughness_m", result.roughness_m),
            ("psim", result.psim),
        ]:
            assert isinstance(arr, np.ndarray), f"{name} should be numpy array"

    def test_shapes_match_ny_nx(self):
        ny, nx = 7, 11
        result = self._run(ny=ny, nx=nx)
        assert result.ustar.shape == (ny, nx)
        assert result.roughness_m.shape == (ny, nx)
        assert result.psim.shape == (ny, nx)

    def test_dtype_float64(self):
        result = self._run()
        assert result.ustar.dtype == np.float64
        assert result.roughness_m.dtype == np.float64
        assert result.psim.dtype == np.float64

    def test_values_propagated(self):
        result = self._run(ustar_val=0.42, z0_val=0.07, psim_val=1.5)
        assert np.allclose(result.ustar, 0.42)
        assert np.allclose(result.roughness_m, 0.07)
        assert np.allclose(result.psim, 1.5)


# ---------------------------------------------------------------------------
# Tests: debug consistency check
# ---------------------------------------------------------------------------

class TestConsistencyCheck:
    """_check_consistency / __debug__ assertion behaviour."""

    def _run_with_mismatch(self, bad_field: str):
        """Run with one field deliberately mismatched in carry.diag."""
        from gpuwrf.wrf_lbm_coupling.surface_diagnostics import compute_surface_diagnostics

        ny, nx = 3, 3
        ustar_val, z0_val = 0.3, 0.05

        fake_diag = _make_diag(ny, nx, ustar_val, z0_val)
        carry = _make_carry(ny, nx, ustar_val, z0_val)

        # Introduce mismatch in the requested field
        if bad_field == "ustar":
            carry.diag.ustar = np.full((ny, nx), ustar_val * 2)  # wrong
        else:
            carry.diag.roughness_m = np.full((ny, nx), z0_val * 2)  # wrong

        grid = MagicMock()
        with (
            patch(
                "gpuwrf.wrf_lbm_coupling.surface_diagnostics.get_surface_diagnostics",
                return_value=fake_diag,
            ),
            patch(
                "gpuwrf.wrf_lbm_coupling.surface_diagnostics.jax.device_get",
                side_effect=_fake_jax_device_get,
            ),
        ):
            return compute_surface_diagnostics(carry, grid)

    def test_consistent_carry_passes(self):
        """Matching carry.diag → no error."""
        from gpuwrf.wrf_lbm_coupling.surface_diagnostics import compute_surface_diagnostics

        ny, nx = 3, 3
        ustar_val, z0_val = 0.3, 0.05
        fake_diag = _make_diag(ny, nx, ustar_val, z0_val)
        carry = _make_carry(ny, nx, ustar_val, z0_val)
        grid = MagicMock()

        with (
            patch(
                "gpuwrf.wrf_lbm_coupling.surface_diagnostics.get_surface_diagnostics",
                return_value=fake_diag,
            ),
            patch(
                "gpuwrf.wrf_lbm_coupling.surface_diagnostics.jax.device_get",
                side_effect=_fake_jax_device_get,
            ),
        ):
            result = compute_surface_diagnostics(carry, grid)
        assert result is not None  # no exception raised

    def test_ustar_mismatch_raises(self):
        """Mismatched ustar → AssertionError with clear label."""
        with pytest.raises(AssertionError, match="consistency check FAILED"):
            self._run_with_mismatch("ustar")

    def test_z0_mismatch_raises(self):
        """Mismatched roughness_m → AssertionError with clear label."""
        with pytest.raises(AssertionError, match="consistency check FAILED"):
            self._run_with_mismatch("roughness_m")

    def test_no_carry_diag_skips_check(self):
        """carry without .diag attribute → check skipped, no error."""
        from gpuwrf.wrf_lbm_coupling.surface_diagnostics import compute_surface_diagnostics

        ny, nx = 3, 3
        fake_diag = _make_diag(ny, nx)

        class CarryNoDiag:
            state = MagicMock()
            # no .diag attribute

        carry = CarryNoDiag()
        grid = MagicMock()

        with (
            patch(
                "gpuwrf.wrf_lbm_coupling.surface_diagnostics.get_surface_diagnostics",
                return_value=fake_diag,
            ),
            patch(
                "gpuwrf.wrf_lbm_coupling.surface_diagnostics.jax.device_get",
                side_effect=_fake_jax_device_get,
            ),
        ):
            result = compute_surface_diagnostics(carry, grid)

        assert isinstance(result.ustar, np.ndarray)


# ---------------------------------------------------------------------------
# Tests: public wrapper importability
# ---------------------------------------------------------------------------

class TestGetSurfaceDiagnosticsWrapper:
    """get_surface_diagnostics is importable and is a callable."""

    def test_importable(self):
        from gpuwrf.coupling.physics_couplers import get_surface_diagnostics

        assert callable(get_surface_diagnostics)

    def test_wrapper_calls_surface_layer_with_diagnostics(self):
        """Smoke-test: wrapper calls through to surface_layer_with_diagnostics."""
        from gpuwrf.coupling.physics_couplers import get_surface_diagnostics

        sentinel = object()

        with (
            patch(
                "gpuwrf.coupling.physics_couplers._surface_column_view",
                return_value=sentinel,
            ) as mock_view,
            patch(
                "gpuwrf.coupling.physics_couplers.surface_layer_with_diagnostics",
                return_value=MagicMock(),
            ) as mock_sld,
        ):
            state = MagicMock()
            grid = MagicMock()
            get_surface_diagnostics(state, grid)

        mock_view.assert_called_once_with(state, grid)
        mock_sld.assert_called_once_with(sentinel)

    def test_wrapper_grid_none_default(self):
        """grid=None is the default (passed through to _surface_column_view)."""
        from gpuwrf.coupling.physics_couplers import get_surface_diagnostics

        with (
            patch(
                "gpuwrf.coupling.physics_couplers._surface_column_view",
                return_value=MagicMock(),
            ) as mock_view,
            patch(
                "gpuwrf.coupling.physics_couplers.surface_layer_with_diagnostics",
                return_value=MagicMock(),
            ),
        ):
            state = MagicMock()
            get_surface_diagnostics(state)

        mock_view.assert_called_once_with(state, None)


# ---------------------------------------------------------------------------
# Tests: compute_surface_fields (MOST fields + hfx from the same snapshot)
# ---------------------------------------------------------------------------

class TestComputeSurfaceFields:
    def _run(self, ny=4, nx=5, hfx_val=42.0):
        from gpuwrf.wrf_lbm_coupling.surface_diagnostics import compute_surface_fields

        fake_diag = _make_diag(ny, nx)
        fake_diag.hfx = np.full((ny, nx), hfx_val, dtype=np.float32)
        carry = _make_carry(ny, nx)
        with (
            patch(
                "gpuwrf.wrf_lbm_coupling.surface_diagnostics.get_surface_diagnostics",
                return_value=fake_diag,
            ),
            patch(
                "gpuwrf.wrf_lbm_coupling.surface_diagnostics.jax.device_get",
                side_effect=_fake_jax_device_get,
            ),
        ):
            return compute_surface_fields(carry, MagicMock())

    def test_returns_most_fields_and_hfx(self):
        from gpuwrf.wrf_lbm_coupling.vertical_interp import MOSTSurfaceFields

        most_fields, hfx = self._run(hfx_val=42.0)
        assert isinstance(most_fields, MOSTSurfaceFields)
        assert hfx.shape == (4, 5)
        assert hfx.dtype == np.float64
        np.testing.assert_allclose(hfx, 42.0)

    def test_most_fields_match_compute_surface_diagnostics(self):
        most_fields, _ = self._run()
        np.testing.assert_allclose(most_fields.ustar, 0.3)
        np.testing.assert_allclose(most_fields.roughness_m, 0.05)
        np.testing.assert_allclose(most_fields.psim, -0.2)
