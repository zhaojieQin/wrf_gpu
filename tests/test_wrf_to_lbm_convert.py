"""Unit tests for WRF → LBM conversion layer.

Tests are pure-numpy (no GPU required) and run on CI.

Coverage:
  Part 1 – vertical_interp.py
    - compute_z_w / compute_z_mass: analytic field round-trip
    - interp_field_to_heights: known-value spot checks
    - Edge cases: below floor, above ceiling, horizontal out-of-bounds
    - Flat-terrain isobaric: z varies only with k, not (j, i)
  Part 2 – unit_convert.py
    - u_to_lbm / lbm_to_u: round-trip at known values
    - theta_to_lbm: passthrough (WRF θ−300 == LBM concentration)
    - LBMReference from para-like dict
  Part 3 – most_extrapolate / MOSTSurfaceFields (path 2)
    - Neutral: psim=0 recovers exact log-law
    - Stable: psim<0 gives higher wind than neutral
    - Unstable: psim>0 gives lower wind than neutral
    - z_agl <= z0 raises ValueError
    - ustar <= 0 returns 0
    - NaN psim falls back to neutral with warning
    - Integration: interp_wrf_to_lbm_nodes with most_fields patches below-floor nodes
    - Direction preservation: u/v direction unchanged, only magnitude scaled
    - Static-wind protection: calm first-layer wind returns (0, 0)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gpuwrf.wrf_lbm_coupling.vertical_interp import (
    WRFColumnData,
    MOSTSurfaceFields,
    compute_z_mass,
    compute_z_w,
    interp_field_to_heights,
    interp_wrf_to_lbm_nodes,
    most_extrapolate,
)
from gpuwrf.wrf_lbm_coupling.unit_convert import (
    LBMReference,
    theta_to_lbm,
    uvw_to_lbm,
    lbm_to_uvw,
)

G = 9.81

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear_geopotential(nz: int, ny: int, nx: int, dz: float = 50.0) -> np.ndarray:
    """Return ph_total = g * k * dz (flat terrain, uniform layer spacing)."""
    # z_w[k] = k * dz  →  ph_total[k] = G * k * dz
    k = np.arange(nz + 1, dtype=np.float64)
    ph = G * k[:, None, None] * dz * np.ones((nz + 1, ny, nx))
    return ph


# ---------------------------------------------------------------------------
# Part 1: vertical_interp
# ---------------------------------------------------------------------------

class TestComputeZW:
    def test_flat_terrain_linear(self):
        nz, ny, nx = 5, 3, 4
        dz = 100.0
        ph = _make_linear_geopotential(nz, ny, nx, dz)
        z_w = compute_z_w(ph)
        assert z_w.shape == (nz + 1, ny, nx)
        # z_w[k] should equal k * dz everywhere
        for k in range(nz + 1):
            np.testing.assert_allclose(z_w[k], k * dz, rtol=1e-10)

    def test_output_dtype_float64(self):
        ph = np.ones((4, 3, 3), dtype=np.float32)
        z_w = compute_z_w(ph)
        assert z_w.dtype == np.float64

    def test_zero_geopotential_gives_zero_height(self):
        ph = np.zeros((3, 2, 2))
        z_w = compute_z_w(ph)
        np.testing.assert_array_equal(z_w, 0.0)


class TestComputeZMass:
    def test_midpoint_averaging(self):
        nz, ny, nx = 4, 2, 2
        dz = 50.0
        ph = _make_linear_geopotential(nz, ny, nx, dz)
        z_w = compute_z_w(ph)
        z_mass = compute_z_mass(z_w)
        assert z_mass.shape == (nz, ny, nx)
        # z_mass[k] = 0.5*(k*dz + (k+1)*dz) = (k + 0.5) * dz
        for k in range(nz):
            np.testing.assert_allclose(z_mass[k], (k + 0.5) * dz, rtol=1e-10)

    def test_shape_reduction(self):
        z_w = np.zeros((11, 5, 7))
        z_mass = compute_z_mass(z_w)
        assert z_mass.shape == (10, 5, 7)


class TestInterpFieldToHeights:
    """Spot-check the core interpolation function."""

    def _setup(self, nz=10, ny=4, nx=4, dz=100.0):
        ph = _make_linear_geopotential(nz, ny, nx, dz)
        z_w = compute_z_w(ph)
        z_mass = compute_z_mass(z_w)
        # Field = k + 1 (1-indexed level number) on mass levels
        field = np.zeros((nz, ny, nx))
        for k in range(nz):
            field[k] = k + 1.0
        return field, z_mass, ph, z_w

    def test_exact_level_hit(self):
        nz, dz = 10, 100.0
        field, z_mass, *_ = self._setup(nz=nz, dz=dz)
        ny, nx = 4, 4
        terrain = np.zeros((ny, nx))
        # Target the exact z of mass level 3 (k=3): z = (3 + 0.5) * 100 = 350 m
        z_target = np.array([350.0])
        x_target = np.array([0.0])
        y_target = np.array([0.0])
        vals = interp_field_to_heights(
            field, z_mass, z_target, x_target, y_target, terrain,
            dx_wrf=1000.0, dy_wrf=1000.0,
        )
        np.testing.assert_allclose(vals[0], 4.0, rtol=1e-10)

    def test_midpoint_interpolation(self):
        """Halfway between level 2 and level 3 should give 2.5."""
        nz, dz = 10, 100.0
        field, z_mass, *_ = self._setup(nz=nz, dz=dz)
        ny, nx = 4, 4
        terrain = np.zeros((ny, nx))
        z2 = (2 + 0.5) * dz   # 250 m, level 2
        z3 = (3 + 0.5) * dz   # 350 m, level 3
        z_mid = 0.5 * (z2 + z3)
        vals = interp_field_to_heights(
            field, z_mass,
            np.array([z_mid]), np.array([0.0]), np.array([0.0]),
            terrain, dx_wrf=1000.0, dy_wrf=1000.0,
        )
        np.testing.assert_allclose(vals[0], 2.5 + 1.0, rtol=1e-10)

    def test_below_floor_clamps_and_warns(self):
        nz, dz = 5, 100.0
        field, z_mass, *_ = self._setup(nz=nz, dz=dz)
        ny, nx = 4, 4
        terrain = np.zeros((ny, nx))
        z_below = np.array([-10.0])  # below z_mass[0] = 50 m
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vals = interp_field_to_heights(
                field, z_mass, z_below,
                np.array([0.0]), np.array([0.0]),
                terrain, dx_wrf=1000.0, dy_wrf=1000.0,
            )
        assert any("below" in str(warning.message).lower() for warning in w)
        np.testing.assert_allclose(vals[0], 1.0, rtol=1e-10)  # clamped to level 0

    def test_above_ceiling_clamps_and_warns(self):
        nz, dz = 5, 100.0
        field, z_mass, *_ = self._setup(nz=nz, dz=dz)
        ny, nx = 4, 4
        terrain = np.zeros((ny, nx))
        z_above = np.array([1e6])  # above z_mass[-1]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vals = interp_field_to_heights(
                field, z_mass, z_above,
                np.array([0.0]), np.array([0.0]),
                terrain, dx_wrf=1000.0, dy_wrf=1000.0,
            )
        assert any("above" in str(warning.message).lower() for warning in w)
        np.testing.assert_allclose(vals[0], float(nz), rtol=1e-10)  # top level

    def test_horizontal_out_of_bounds_raises(self):
        nz, dz = 5, 100.0
        field, z_mass, *_ = self._setup(nz=nz, dz=dz)
        ny, nx = 4, 4
        terrain = np.zeros((ny, nx))
        with pytest.raises(ValueError, match="outside the WRF"):
            interp_field_to_heights(
                field, z_mass,
                np.array([200.0]),
                np.array([99999.0]),   # way outside
                np.array([0.0]),
                terrain, dx_wrf=1000.0, dy_wrf=1000.0,
            )

    def test_multiple_nodes(self):
        nz, dz = 10, 100.0
        field, z_mass, *_ = self._setup(nz=nz, dz=dz)
        ny, nx = 4, 4
        terrain = np.zeros((ny, nx))
        # Query 3 exact level centers
        z_targets = np.array([(k + 0.5) * dz for k in [0, 4, 9]])
        x_targets = np.zeros(3)
        y_targets = np.zeros(3)
        vals = interp_field_to_heights(
            field, z_mass, z_targets, x_targets, y_targets,
            terrain, dx_wrf=1000.0, dy_wrf=1000.0,
        )
        np.testing.assert_allclose(vals, [1.0, 5.0, 10.0], rtol=1e-10)


class TestInterpWRFToLBMNodes:
    """Integration test: interp_wrf_to_lbm_nodes."""

    def test_flat_terrain_uniform_wind(self):
        """Uniform u=5 m/s everywhere → interpolated u = 5 m/s at any height."""
        nz, ny, nx = 8, 4, 4
        dz = 100.0
        ph = _make_linear_geopotential(nz, ny, nx, dz)
        u = 5.0 * np.ones((nz, ny, nx))
        v = -3.0 * np.ones((nz, ny, nx))
        w = 0.1 * np.ones((nz + 1, ny, nx))
        theta = 2.0 * np.ones((nz, ny, nx))  # θ−300 = 2 K

        terrain = np.zeros((ny, nx))
        wrf_data = WRFColumnData(
            ph_total=ph, u=u, v=v, w=w, theta=theta,
            terrain_height=terrain, dx_wrf=1000.0, dy_wrf=1000.0,
        )
        target_z = np.array([200.0, 400.0])
        target_x = np.array([0.0, 500.0])
        target_y = np.array([0.0, 500.0])

        result = interp_wrf_to_lbm_nodes(wrf_data, target_z, target_x, target_y)
        np.testing.assert_allclose(result.u, 5.0, rtol=1e-8)
        np.testing.assert_allclose(result.v, -3.0, rtol=1e-8)
        np.testing.assert_allclose(result.theta, 2.0, rtol=1e-8)

    def test_theta_passthrough_signed(self):
        """theta < 0 (stable layer) should be passed through unchanged."""
        nz, ny, nx = 4, 2, 2
        dz = 100.0
        ph = _make_linear_geopotential(nz, ny, nx, dz)
        theta = np.linspace(-5, 5, nz)[:, None, None] * np.ones((nz, ny, nx))
        u = v = np.zeros((nz, ny, nx))
        w = np.zeros((nz + 1, ny, nx))
        terrain = np.zeros((ny, nx))
        wrf_data = WRFColumnData(
            ph_total=ph, u=u, v=v, w=w, theta=theta,
            terrain_height=terrain, dx_wrf=1000.0, dy_wrf=1000.0,
        )
        # Query midpoint of level 0 (z=50 m, theta=-5 K)
        result = interp_wrf_to_lbm_nodes(
            wrf_data,
            np.array([50.0]),
            np.array([0.0]),
            np.array([0.0]),
        )
        np.testing.assert_allclose(result.theta[0], -5.0, rtol=1e-8)

    def test_vertical_gradient_roundtrip(self):
        """Field = z_mass (i.e. field value == height in m). Round-trip: value ≈ z."""
        nz, ny, nx = 20, 2, 2
        dz = 50.0
        ph = _make_linear_geopotential(nz, ny, nx, dz)
        z_w = compute_z_w(ph)
        z_mass = compute_z_mass(z_w)
        # Set u = z_mass (height-proportional field)
        u = z_mass.copy()
        v = w = np.zeros_like(u)
        theta = np.zeros_like(u)
        w_full = np.zeros((nz + 1, ny, nx))
        terrain = np.zeros((ny, nx))

        wrf_data = WRFColumnData(
            ph_total=ph, u=u, v=v, w=w_full, theta=theta,
            terrain_height=terrain, dx_wrf=1000.0, dy_wrf=1000.0,
        )
        # Query at several heights within bounds
        target_z = np.array([75.0, 200.0, 450.0, 800.0])
        target_x = np.zeros(4)
        target_y = np.zeros(4)
        result = interp_wrf_to_lbm_nodes(wrf_data, target_z, target_x, target_y)
        # Since u==z_mass, interpolated u should ≈ target_z
        np.testing.assert_allclose(result.u, target_z, rtol=1e-6, atol=0.1)


# ---------------------------------------------------------------------------
# Part 2: unit_convert
# ---------------------------------------------------------------------------

class TestLBMReference:
    def test_u_ref_computed_from_dx_dt(self):
        ref = LBMReference(dx_lbm=2.0, dt_lbm=0.1)
        assert ref.u_ref == pytest.approx(20.0)

    def test_from_para_dict(self):
        para_like = {"velocity_ratio": 15.0}
        ref = LBMReference.from_velocity_ratio(para_like["velocity_ratio"], dx_lbm=3.0, dt_lbm=0.2)
        assert ref.u_ref == pytest.approx(15.0)
        assert ref.dx_lbm == pytest.approx(3.0)

    def test_theta_ref_default_300(self):
        ref = LBMReference(dx_lbm=1.0, dt_lbm=0.05)
        assert ref.theta_ref == pytest.approx(300.0)


class TestUVWConvert:
    def test_u_to_lbm_known_value(self):
        """u_phys=10 m/s, u_ref=20 m/s → u_LB=0.5 (dimensionless)."""
        ref = LBMReference(dx_lbm=2.0, dt_lbm=0.1)  # u_ref = 20
        u = np.array([10.0, -5.0, 20.0])
        u_lb = uvw_to_lbm(u, ref)
        np.testing.assert_allclose(u_lb, [0.5, -0.25, 1.0], rtol=1e-10)

    def test_uvw_roundtrip(self):
        ref = LBMReference(dx_lbm=1.5, dt_lbm=0.075)
        rng = np.random.default_rng(42)
        u_phys = rng.uniform(-20.0, 20.0, 100)
        np.testing.assert_allclose(lbm_to_uvw(uvw_to_lbm(u_phys, ref), ref), u_phys, rtol=1e-12)

    def test_zero_wind_stays_zero(self):
        ref = LBMReference(dx_lbm=1.0, dt_lbm=0.05)
        u_lb = uvw_to_lbm(np.zeros(5), ref)
        np.testing.assert_array_equal(u_lb, 0.0)


class TestThetaConvert:
    def test_passthrough(self):
        """theta_to_lbm must return WRF theta unchanged (K deviation)."""
        theta_wrf = np.array([-5.0, 0.0, 3.7, 10.2])
        ref = LBMReference(dx_lbm=1.0, dt_lbm=0.05)
        theta_lb = theta_to_lbm(theta_wrf, ref)
        np.testing.assert_array_equal(theta_lb, theta_wrf)

    def test_dtype_preserved(self):
        theta_wrf = np.array([1.0, 2.0], dtype=np.float32)
        ref = LBMReference(dx_lbm=1.0, dt_lbm=0.05)
        theta_lb = theta_to_lbm(theta_wrf, ref)
        assert theta_lb.dtype == np.float32


# ---------------------------------------------------------------------------
# Part 3: most_extrapolate / MOSTSurfaceFields (path 2)
# ---------------------------------------------------------------------------

class TestMostExtrapolate:
    """Unit tests for most_extrapolate() — MOST log-linear wind profile."""

    def test_neutral_recovers_log_law(self):
        """psim=0 (neutral) must give exactly (u*/κ) * ln(z/z0)."""
        ustar, z0, z_agl = 0.5, 0.1, 10.0
        expected = (ustar / 0.4) * np.log(z_agl / z0)
        result = most_extrapolate(z_agl, ustar, z0, psim=0.0)
        assert abs(result - expected) < 1e-10

    def test_stable_gives_higher_wind(self):
        """psim < 0 (stable) → U(z) > neutral log profile."""
        ustar, z0, z_agl = 0.3, 0.05, 5.0
        u_neutral = most_extrapolate(z_agl, ustar, z0, psim=0.0)
        u_stable  = most_extrapolate(z_agl, ustar, z0, psim=-1.5)
        assert u_stable > u_neutral

    def test_unstable_gives_lower_wind(self):
        """psim > 0 (unstable) → U(z) < neutral log profile."""
        ustar, z0, z_agl = 0.6, 0.1, 15.0
        u_neutral  = most_extrapolate(z_agl, ustar, z0, psim=0.0)
        u_unstable = most_extrapolate(z_agl, ustar, z0, psim=2.0)
        assert u_unstable < u_neutral

    def test_below_roughness_raises(self):
        """z_agl <= z0 must raise ValueError."""
        with pytest.raises(ValueError, match="z0"):
            most_extrapolate(0.05, ustar=0.5, z0=0.1, psim=0.0)

    def test_zero_ustar_returns_zero(self):
        """ustar=0 (calm) must return 0 without errors."""
        result = most_extrapolate(5.0, ustar=0.0, z0=0.1, psim=0.0)
        assert result == 0.0

    def test_nan_psim_falls_back_to_neutral(self):
        """NaN psim must warn and fall back to neutral (psim=0)."""
        ustar, z0, z_agl = 0.4, 0.1, 8.0
        expected_neutral = most_extrapolate(z_agl, ustar, z0, psim=0.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = most_extrapolate(z_agl, ustar, z0, psim=float("nan"))
        assert any("neutral" in str(warning.message).lower() for warning in w)
        assert abs(result - expected_neutral) < 1e-10

    def test_result_always_nonnegative(self):
        """Wind speed must never be negative (large negative psim edge case)."""
        # Very large stable correction can in principle make the formula negative;
        # most_extrapolate must clamp to 0.
        result = most_extrapolate(1.0, ustar=0.01, z0=0.05, psim=-100.0)
        assert result >= 0.0

    def test_known_value(self):
        """Spot-check: ustar=0.4, z0=0.01, z=2m, psim=1.0 (unstable).
        U = (0.4/0.4)*(ln(200) - 1) = ln(200) - 1 ≈ 4.298"""
        result = most_extrapolate(2.0, ustar=0.4, z0=0.01, psim=1.0)
        expected = np.log(2.0 / 0.01) - 1.0   # = ln(200) - 1
        assert abs(result - expected) < 1e-10


class TestMostIntegration:
    """Integration tests: interp_wrf_to_lbm_nodes with MOSTSurfaceFields."""

    def _make_wrf_data(self, nz=10, ny=4, nx=4, dz=50.0, u_val=5.0):
        """Flat terrain, uniform wind u_val, zero v/w/theta."""
        G = 9.81
        k = np.arange(nz + 1, dtype=np.float64)
        ph = G * k[:, None, None] * dz * np.ones((nz + 1, ny, nx))
        u = u_val * np.ones((nz, ny, nx))
        v = np.zeros((nz, ny, nx))
        w = np.zeros((nz + 1, ny, nx))
        theta = np.zeros((nz, ny, nx))
        terrain = np.zeros((ny, nx))
        return WRFColumnData(
            ph_total=ph, u=u, v=v, w=w, theta=theta,
            terrain_height=terrain, dx_wrf=1000.0, dy_wrf=1000.0,
        )

    def test_most_patches_below_floor_nodes(self):
        """Nodes below z_mass[0] get MOST value instead of clamped WRF k=0."""
        wrf_data = self._make_wrf_data(dz=50.0, u_val=8.0)
        # z_mass[0] = 25m AGL (dz=50 → midpoint = 25m)
        # Target 5m AGL — below floor

        ustar = 0.5 * np.ones((4, 4))
        z0    = 0.1 * np.ones((4, 4))
        psim  = np.zeros((4, 4))   # neutral: U = (0.5/0.4)*ln(5/0.1) ≈ 4.88 m/s
        most_fields = MOSTSurfaceFields(ustar=ustar, roughness_m=z0, psim=psim)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_most = interp_wrf_to_lbm_nodes(
                wrf_data,
                target_z=np.array([5.0]),
                target_x=np.array([0.0]),
                target_y=np.array([0.0]),
                most_fields=most_fields,
            )
        # Without MOST, u would be clamped to 8.0 (WRF k=0)
        result_clamp = interp_wrf_to_lbm_nodes(
            wrf_data,
            target_z=np.array([5.0]),
            target_x=np.array([0.0]),
            target_y=np.array([0.0]),
        )
        expected_most = (0.5 / 0.4) * np.log(5.0 / 0.1)
        np.testing.assert_allclose(result_most.u[0], expected_most, rtol=1e-8)
        # MOST value should differ from clamped value
        assert abs(result_most.u[0] - result_clamp.u[0]) > 0.1
        # Warning was issued
        assert any("MOST" in str(ww.message) for ww in w)

    def test_above_floor_nodes_unchanged(self):
        """Nodes above z_mass[0] must not be affected by most_fields."""
        wrf_data = self._make_wrf_data(dz=50.0, u_val=5.0)
        most_fields = MOSTSurfaceFields(
            ustar=np.ones((4, 4)),
            roughness_m=0.1 * np.ones((4, 4)),
            psim=np.zeros((4, 4)),
        )
        # 100m AGL > 25m floor: standard interpolation, not MOST
        result_most = interp_wrf_to_lbm_nodes(
            wrf_data, np.array([100.0]), np.array([0.0]), np.array([0.0]),
            most_fields=most_fields,
        )
        result_plain = interp_wrf_to_lbm_nodes(
            wrf_data, np.array([100.0]), np.array([0.0]), np.array([0.0]),
        )
        np.testing.assert_allclose(result_most.u, result_plain.u, rtol=1e-12)

    def test_direction_preserved(self):
        """MOST must keep the same wind direction as WRF first-layer u/v."""
        nz, ny, nx = 6, 4, 4
        dz = 50.0
        G = 9.81
        k = np.arange(nz + 1, dtype=np.float64)
        ph = G * k[:, None, None] * dz * np.ones((nz + 1, ny, nx))
        # First-layer wind: 3-4-5 right triangle (u=3, v=4, speed=5 m/s)
        u = 3.0 * np.ones((nz, ny, nx))
        v = 4.0 * np.ones((nz, ny, nx))
        w = np.zeros((nz + 1, ny, nx))
        theta = np.zeros((nz, ny, nx))
        terrain = np.zeros((ny, nx))
        wrf_data = WRFColumnData(
            ph_total=ph, u=u, v=v, w=w, theta=theta,
            terrain_height=terrain, dx_wrf=1000.0, dy_wrf=1000.0,
        )
        # z_mass[0] = 25m AGL; target 5m (below floor)
        most_fields = MOSTSurfaceFields(
            ustar=0.4 * np.ones((ny, nx)),
            roughness_m=0.1 * np.ones((ny, nx)),
            psim=np.zeros((ny, nx)),
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = interp_wrf_to_lbm_nodes(
                wrf_data, np.array([5.0]), np.array([0.0]), np.array([0.0]),
                most_fields=most_fields,
            )
        # Direction angle must match (u/v ratio must be 3/4)
        assert abs(result.u[0] / result.v[0] - 3.0 / 4.0) < 1e-10

    def test_static_wind_gives_calm(self):
        """First-layer wind < 0.1 m/s → MOST returns (0, 0)."""
        nz, ny, nx = 6, 4, 4
        dz = 50.0
        G = 9.81
        k = np.arange(nz + 1, dtype=np.float64)
        ph = G * k[:, None, None] * dz * np.ones((nz + 1, ny, nx))
        # Near-calm first layer
        u = 0.05 * np.ones((nz, ny, nx))
        v = np.zeros((nz, ny, nx))
        w = np.zeros((nz + 1, ny, nx))
        theta = np.zeros((nz, ny, nx))
        terrain = np.zeros((ny, nx))
        wrf_data = WRFColumnData(
            ph_total=ph, u=u, v=v, w=w, theta=theta,
            terrain_height=terrain, dx_wrf=1000.0, dy_wrf=1000.0,
        )
        most_fields = MOSTSurfaceFields(
            ustar=0.1 * np.ones((ny, nx)),
            roughness_m=0.1 * np.ones((ny, nx)),
            psim=np.zeros((ny, nx)),
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = interp_wrf_to_lbm_nodes(
                wrf_data, np.array([5.0]), np.array([0.0]), np.array([0.0]),
                most_fields=most_fields,
            )
        assert result.u[0] == 0.0
        assert result.v[0] == 0.0

    def test_no_most_fields_unchanged(self):
        """Passing most_fields=None must give identical result to old signature."""
        wrf_data = self._make_wrf_data(dz=50.0, u_val=5.0)
        target_z = np.array([200.0, 100.0, 300.0])
        target_x = np.array([0.0, 500.0, 1000.0])
        target_y = np.array([0.0, 500.0, 1000.0])
        r1 = interp_wrf_to_lbm_nodes(wrf_data, target_z, target_x, target_y)
        r2 = interp_wrf_to_lbm_nodes(wrf_data, target_z, target_x, target_y, most_fields=None)
        np.testing.assert_array_equal(r1.u, r2.u)
        np.testing.assert_array_equal(r1.v, r2.v)
