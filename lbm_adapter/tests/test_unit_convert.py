"""任务6单元测试：lbm_adapter/unit_convert.py — apply_unit_conversions

运行方法（在 wrf_gpu/ 目录下）:
    python -m pytest lbm_adapter/tests/test_unit_convert.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lbm_adapter.unit_convert import (
    LBMReference,
    apply_unit_conversions,
    uvw_to_lbm,
    theta_to_lbm,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NY, NX = 4, 4
NZ = 10


def _ref(dx=1.0, dt=0.01, theta_ref=300.0) -> LBMReference:
    return LBMReference(dx_lbm=dx, dt_lbm=dt, theta_ref=theta_ref)


def _received(u_val=5.0, v_val=-3.0, w_val=0.1,
              theta_val=310.0, with_extra=False) -> dict:
    shape3 = (NZ, NY, NX)
    d = {
        "__wrf_time_s": 4.0,
        "u":     u_val    * np.ones(shape3, dtype=np.float32),
        "v":     v_val    * np.ones(shape3, dtype=np.float32),
        "w":     w_val    * np.ones((NZ + 1, NY, NX), dtype=np.float32),
        "theta": theta_val * np.ones(shape3, dtype=np.float32),
    }
    if with_extra:
        d["ustar"]       = 0.5  * np.ones((NY, NX), dtype=np.float32)
        d["roughness_m"] = 0.1  * np.ones((NY, NX), dtype=np.float32)
        d["psim"]        = np.zeros((NY, NX), dtype=np.float32)
    return d


# ---------------------------------------------------------------------------
# TestLBMReference
# ---------------------------------------------------------------------------

class TestLBMReference:
    def test_u_ref(self):
        ref = _ref(dx=2.0, dt=0.002)
        assert ref.u_ref == pytest.approx(1000.0)

    def test_negative_dt_raises(self):
        with pytest.raises(ValueError, match="dt_lbm"):
            LBMReference(dx_lbm=1.0, dt_lbm=-0.01)

    def test_negative_dx_raises(self):
        with pytest.raises(ValueError, match="dx_lbm"):
            LBMReference(dx_lbm=-1.0, dt_lbm=0.01)

    def test_from_velocity_ratio_ok(self):
        ref = LBMReference.from_velocity_ratio(
            velocity_ratio=100.0, dx_lbm=1.0, dt_lbm=0.01
        )
        assert ref.u_ref == pytest.approx(100.0)

    def test_from_velocity_ratio_mismatch_raises(self):
        with pytest.raises(ValueError, match="velocity_ratio"):
            LBMReference.from_velocity_ratio(
                velocity_ratio=200.0, dx_lbm=1.0, dt_lbm=0.01
            )


# ---------------------------------------------------------------------------
# TestApplyUnitConversions — theta
# ---------------------------------------------------------------------------

class TestApplyUnitConversionsTheta:
    def test_theta_offset_subtracted(self):
        """theta_full=310 K → concentration = 310 - 300 = 10 K."""
        ref = _ref(theta_ref=300.0)
        received = _received(theta_val=310.0)
        out = apply_unit_conversions(received, ref)
        np.testing.assert_allclose(out["theta"], 10.0, rtol=1e-6)

    def test_theta_offset_uses_theta_ref(self):
        """theta_ref=305 K → concentration = 310 - 305 = 5 K."""
        ref = _ref(theta_ref=305.0)
        received = _received(theta_val=310.0)
        out = apply_unit_conversions(received, ref)
        np.testing.assert_allclose(out["theta"], 5.0, rtol=1e-6)

    def test_theta_not_full_value(self):
        """输出不应是 full θ (~300-470 K)，应是偏差量（<200 K）。"""
        ref = _ref(theta_ref=300.0)
        received = _received(theta_val=320.0)
        out = apply_unit_conversions(received, ref)
        assert out["theta"].max() < 200.0, (
            f"theta max={out['theta'].max():.1f} — 疑似未减去 theta_ref"
        )

    def test_theta_output_dtype_float64(self):
        ref = _ref()
        out = apply_unit_conversions(_received(), ref)
        assert out["theta"].dtype == np.float64

    def test_theta_shape_preserved(self):
        ref = _ref()
        received = _received()
        out = apply_unit_conversions(received, ref)
        assert out["theta"].shape == received["theta"].shape

    def test_input_not_mutated(self):
        """apply_unit_conversions 不修改 received 原地数据。"""
        ref = _ref()
        received = _received(theta_val=350.0)
        theta_before = received["theta"].copy()
        apply_unit_conversions(received, ref)
        np.testing.assert_array_equal(received["theta"], theta_before)


# ---------------------------------------------------------------------------
# TestApplyUnitConversions — velocity
# ---------------------------------------------------------------------------

class TestApplyUnitConversionsVelocity:
    def test_u_scaled_by_u_ref(self):
        """u_lbm = u_phys / u_ref。"""
        ref = _ref(dx=1.0, dt=0.01)   # u_ref = 100 m/s
        received = _received(u_val=5.0)
        out = apply_unit_conversions(received, ref)
        np.testing.assert_allclose(out["u"], 5.0 / 100.0, rtol=1e-6)

    def test_v_scaled_by_u_ref(self):
        ref = _ref(dx=1.0, dt=0.01)
        received = _received(v_val=-3.0)
        out = apply_unit_conversions(received, ref)
        np.testing.assert_allclose(out["v"], -3.0 / 100.0, rtol=1e-6)

    def test_w_scaled_by_u_ref(self):
        ref = _ref(dx=1.0, dt=0.01)
        received = _received(w_val=0.2)
        out = apply_unit_conversions(received, ref)
        np.testing.assert_allclose(out["w"], 0.2 / 100.0, rtol=1e-6)

    def test_velocity_dtype_float64(self):
        ref = _ref()
        out = apply_unit_conversions(_received(), ref)
        for key in ("u", "v", "w"):
            assert out[key].dtype == np.float64, key

    def test_velocity_shape_preserved(self):
        ref = _ref()
        received = _received()
        out = apply_unit_conversions(received, ref)
        for key in ("u", "v", "w"):
            assert out[key].shape == received[key].shape, key

    def test_zero_velocity_stays_zero(self):
        ref = _ref()
        received = _received(u_val=0.0, v_val=0.0, w_val=0.0)
        out = apply_unit_conversions(received, ref)
        for key in ("u", "v", "w"):
            np.testing.assert_allclose(out[key], 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# TestApplyUnitConversions — passthrough keys
# ---------------------------------------------------------------------------

class TestApplyUnitConversionsPassthrough:
    def test_control_key_passed_through(self):
        """__wrf_time_s 不被修改。"""
        ref = _ref()
        received = _received()
        out = apply_unit_conversions(received, ref)
        assert out["__wrf_time_s"] == pytest.approx(4.0)

    def test_most_fields_passed_through(self):
        """ustar/roughness_m/psim 不被转换。"""
        ref = _ref()
        received = _received(with_extra=True)
        out = apply_unit_conversions(received, ref)
        np.testing.assert_allclose(out["ustar"], 0.5)
        np.testing.assert_allclose(out["roughness_m"], 0.1)
        np.testing.assert_allclose(out["psim"], 0.0)

    def test_all_keys_preserved(self):
        ref = _ref()
        received = _received(with_extra=True)
        out = apply_unit_conversions(received, ref)
        assert set(out.keys()) == set(received.keys())

    def test_extra_unknown_key_passed_through(self):
        ref = _ref()
        received = _received()
        received["__extra"] = 42
        out = apply_unit_conversions(received, ref)
        assert out["__extra"] == 42


# ---------------------------------------------------------------------------
# TestApplyUnitConversions — round-trip
# ---------------------------------------------------------------------------

class TestApplyUnitConversionsRoundTrip:
    def test_theta_round_trip(self):
        """θ_full=350 K → concentration=50 K → add back theta_ref → 350 K。"""
        ref = _ref(theta_ref=300.0)
        theta_full = 350.0 * np.ones((NZ, NY, NX))
        received = {"theta": theta_full.astype(np.float32),
                    "__wrf_time_s": 0.0}
        out = apply_unit_conversions(received, ref)
        recovered = out["theta"] + ref.theta_ref
        np.testing.assert_allclose(recovered, 350.0, rtol=1e-6)

    def test_velocity_round_trip(self):
        """u_phys → u_lbm → u_phys 还原。"""
        from lbm_adapter.unit_convert import lbm_to_uvw
        ref = _ref(dx=2.0, dt=0.002)   # u_ref=1000 m/s
        u_phys = np.array([10.0, -5.0, 0.5])
        u_lb = uvw_to_lbm(u_phys, ref)
        u_back = lbm_to_uvw(u_lb, ref)
        np.testing.assert_allclose(u_back, u_phys, rtol=1e-12)
