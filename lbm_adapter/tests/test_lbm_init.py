"""任务7单元测试：lbm_adapter/lbm_init.py

纯 numpy，无 VirtualFluids，无 MPI。
sim 用 Mock 替代。

运行方法（在 wrf_gpu/ 目录下）:
    python -m pytest lbm_adapter/tests/test_lbm_init.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lbm_adapter.lbm_init import (
    WRFFrame,
    WRFDoubleBuffer,
    _make_wrf_frame,
    init_lbm_from_wrf,
    make_wrf_initial_condition,
)
from lbm_adapter.wrf_to_lbm import WRFStaticContext
from lbm_adapter.unit_convert import LBMReference

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

G = 9.81
NZ, NY, NX = 10, 4, 4
DZ = 500.0
N_BC = 5


def _make_ph(nz=NZ, ny=NY, nx=NX, dz=DZ):
    k = np.arange(nz + 1, dtype=np.float64)
    return (G * k[:, None, None] * dz * np.ones((nz + 1, ny, nx))).astype(np.float32)


def _make_received(t_wrf=0.0, u_val=5.0, theta_val=310.0, nz=NZ, ny=NY, nx=NX):
    return {
        "__wrf_time_s": t_wrf,
        "u":     u_val  * np.ones((nz,     ny, nx), dtype=np.float32),
        "v":    -2.0    * np.ones((nz,     ny, nx), dtype=np.float32),
        "w":     0.1    * np.ones((nz + 1, ny, nx), dtype=np.float32),
        "theta": theta_val * np.ones((nz, ny, nx), dtype=np.float32),
        "ph":    _make_ph(nz, ny, nx),
    }


def _make_static_ctx(ny=NY, nx=NX):
    return WRFStaticContext(
        terrain_height=np.zeros((ny, nx)),
        dx_wrf=1000.0,
        dy_wrf=1000.0,
    )


def _make_ref(dx=1.0, dt=0.01, theta_ref=300.0):
    return LBMReference(dx_lbm=dx, dt_lbm=dt, theta_ref=theta_ref)


def _bc_coords(n=N_BC):
    """BC 节点在 WRF 网格内（x∈[0,3000], y∈[0,3000], z∈[500,4000]）。"""
    return np.column_stack([
        np.zeros(n),
        np.zeros(n),
        np.linspace(500.0, 4000.0, n),
    ])


def _make_mock_sim(n_bc=N_BC, n_sl=N_BC):
    sim = MagicMock()
    sim.get_freestream_bc_coordinates.return_value = _bc_coords(n_bc)
    sl_coords_3d = np.column_stack([np.zeros(n_sl), np.zeros(n_sl), np.zeros(n_sl)])
    sim.get_surface_layer_bc_coordinates.return_value = sl_coords_3d
    return sim


# ---------------------------------------------------------------------------
# TestWRFFrame
# ---------------------------------------------------------------------------

class TestWRFFrame:
    def test_construction(self):
        n = 3
        f = WRFFrame(
            t_wrf=2.0,
            u=np.ones(n) * 0.01,
            v=np.zeros(n),
            w=np.zeros(n),
            theta=np.ones(n) * 10.0,
            hfx=None,
        )
        assert f.t_wrf == pytest.approx(2.0)
        assert f.u.shape == (n,)
        assert f.hfx is None

    def test_construction_with_hfx(self):
        n = 3
        hfx = np.full((4, 4), 50.0)
        f = WRFFrame(2.0, np.ones(n) * 0.01, np.zeros(n), np.zeros(n),
                     np.ones(n) * 10.0, hfx)
        assert f.hfx.shape == (4, 4)

    def test_is_namedtuple(self):
        f = WRFFrame(0.0, np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), None)
        assert isinstance(f, tuple)


# ---------------------------------------------------------------------------
# TestWRFDoubleBuffer
# ---------------------------------------------------------------------------

class TestWRFDoubleBuffer:
    def _make_buf(self, t0=0.0, t1=2.0, n=4):
        f0 = WRFFrame(t0, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), None)
        f1 = WRFFrame(t1, np.ones(n), np.ones(n), np.ones(n), np.ones(n) * 10.0, None)
        return WRFDoubleBuffer(buf=[f0, f1])

    def test_t0_t1(self):
        buf = self._make_buf(t0=0.0, t1=2.0)
        assert buf.t0 == pytest.approx(0.0)
        assert buf.t1 == pytest.approx(2.0)

    def test_needs_advance_false(self):
        buf = self._make_buf(t0=0.0, t1=2.0)
        assert not buf.needs_advance(1.0)

    def test_needs_advance_true_at_t1(self):
        buf = self._make_buf(t0=0.0, t1=2.0)
        assert buf.needs_advance(2.0)

    def test_needs_advance_true_beyond(self):
        buf = self._make_buf(t0=0.0, t1=2.0)
        assert buf.needs_advance(3.0)

    def test_interp_at_t0_returns_f0(self):
        buf = self._make_buf(t0=0.0, t1=2.0)
        u, v, w, theta, hfx = buf.interp(0.0)
        np.testing.assert_allclose(u, 0.0, atol=1e-12)

    def test_interp_at_t1_returns_f1(self):
        buf = self._make_buf(t0=0.0, t1=2.0)
        u, v, w, theta, hfx = buf.interp(2.0)
        np.testing.assert_allclose(u, 1.0, rtol=1e-12)

    def test_interp_midpoint(self):
        buf = self._make_buf(t0=0.0, t1=2.0)
        u, v, w, theta, hfx = buf.interp(1.0)
        np.testing.assert_allclose(u, 0.5, rtol=1e-12)
        np.testing.assert_allclose(theta, 5.0, rtol=1e-12)

    def test_interp_clamped_below(self):
        buf = self._make_buf(t0=2.0, t1=4.0)
        u, v, w, theta, hfx = buf.interp(0.0)  # before t0 → clamp to f0
        np.testing.assert_allclose(u, 0.0, atol=1e-12)

    def test_interp_clamped_above(self):
        buf = self._make_buf(t0=0.0, t1=2.0)
        u, v, w, theta, hfx = buf.interp(99.0)  # after t1 → clamp to f1
        np.testing.assert_allclose(u, 1.0, rtol=1e-12)

    def test_advance(self):
        buf = self._make_buf(t0=0.0, t1=2.0, n=3)
        new_frame = WRFFrame(4.0, np.full(3, 2.0), np.zeros(3), np.zeros(3), np.zeros(3), None)
        buf.advance(new_frame)
        assert buf.t0 == pytest.approx(2.0)
        assert buf.t1 == pytest.approx(4.0)
        np.testing.assert_allclose(buf.buf[0].u, 1.0)
        np.testing.assert_allclose(buf.buf[1].u, 2.0)

    def test_interp_zero_dt_returns_f1(self):
        """dt=0 时夹紧到 f1 值（alpha=1.0）。"""
        buf = self._make_buf(t0=2.0, t1=2.0)
        u, v, w, theta, hfx = buf.interp(2.0)
        np.testing.assert_allclose(u, 1.0, rtol=1e-12)

    def test_interp_hfx_none_when_both_none(self):
        """两帧 hfx 均为 None → interp 返回 None。"""
        buf = self._make_buf()
        _, _, _, _, hfx = buf.interp(1.0)
        assert hfx is None

    def test_interp_hfx_interpolated(self):
        """两帧均有 hfx → 线性插值。"""
        n = 3
        hfx0 = np.zeros((2, 2))
        hfx1 = np.ones((2, 2)) * 100.0
        f0 = WRFFrame(0.0, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), hfx0)
        f1 = WRFFrame(2.0, np.ones(n), np.ones(n), np.ones(n), np.zeros(n), hfx1)
        buf = WRFDoubleBuffer(buf=[f0, f1])
        _, _, _, _, hfx = buf.interp(1.0)
        np.testing.assert_allclose(hfx, 50.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# TestMakeWRFFrame
# ---------------------------------------------------------------------------

class TestMakeWRFFrame:
    def test_timestamp_preserved(self):
        received = _make_received(t_wrf=4.0)
        frame = _make_wrf_frame(received, _make_static_ctx(), _bc_coords(), _make_ref())
        assert frame.t_wrf == pytest.approx(4.0)

    def test_output_shape(self):
        received = _make_received()
        frame = _make_wrf_frame(received, _make_static_ctx(), _bc_coords(N_BC), _make_ref())
        for field in ("u", "v", "w", "theta"):
            assert getattr(frame, field).shape == (N_BC,), field
        assert frame.hfx is None  # no hfx in received

    def test_hfx_none_when_no_sl_coords(self):
        """hfx present in received but no sl_coords → hfx stored as None."""
        received = _make_received()
        received["hfx"] = np.full((NY, NX), 30.0, dtype=np.float32)
        frame = _make_wrf_frame(received, _make_static_ctx(), _bc_coords(N_BC), _make_ref())
        assert frame.hfx is None

    def test_hfx_interpolated_to_sl_nodes(self):
        """hfx present and sl_coords provided → hfx shape is (n_sl,)."""
        N_SL = 3
        received = _make_received()
        received["hfx"] = np.full((NY, NX), 30.0, dtype=np.float32)
        sl_coords = np.column_stack([np.zeros(N_SL), np.zeros(N_SL)])
        frame = _make_wrf_frame(received, _make_static_ctx(), _bc_coords(N_BC), _make_ref(),
                                 sl_coords=sl_coords)
        assert frame.hfx is not None
        assert frame.hfx.shape == (N_SL,)
        np.testing.assert_allclose(frame.hfx, 30.0, rtol=1e-5)

    def test_velocity_stays_physical(self):
        """u_phys=5 m/s → frame.u = 5 m/s（VF update_freestream_bc 内部再除 velocityRatio）。"""
        ref = _make_ref(dx=1.0, dt=0.01)  # u_ref = 100 m/s，不应作用于速度
        received = _make_received(u_val=5.0)
        frame = _make_wrf_frame(received, _make_static_ctx(), _bc_coords(), ref)
        np.testing.assert_allclose(frame.u, 5.0, rtol=1e-5)

    def test_hfx_without_sl_coords_warns(self):
        received = _make_received()
        received["hfx"] = np.full((NY, NX), 30.0, dtype=np.float32)
        with pytest.warns(UserWarning, match="sl_coords is None"):
            frame = _make_wrf_frame(received, _make_static_ctx(), _bc_coords(), _make_ref())
        assert frame.hfx is None

    def test_theta_is_deviation_not_full(self):
        """theta_full=310 K → concentration = 310 - 300 = 10 K。"""
        ref = _make_ref(theta_ref=300.0)
        received = _make_received(theta_val=310.0)
        frame = _make_wrf_frame(received, _make_static_ctx(), _bc_coords(), ref)
        np.testing.assert_allclose(frame.theta, 10.0, rtol=1e-4)

    def test_output_dtype_float64(self):
        received = _make_received()
        frame = _make_wrf_frame(received, _make_static_ctx(), _bc_coords(), _make_ref())
        for field in ("u", "v", "w", "theta"):
            assert getattr(frame, field).dtype == np.float64, field


# ---------------------------------------------------------------------------
# TestInitLbmFromWrf
# ---------------------------------------------------------------------------

class TestInitLbmFromWrf:
    def test_returns_double_buffer(self):
        sim = _make_mock_sim()
        r0 = _make_received(t_wrf=0.0)
        r1 = _make_received(t_wrf=2.0)
        db = init_lbm_from_wrf(sim, r0, r1, _make_static_ctx(), _make_ref())
        assert isinstance(db, WRFDoubleBuffer)

    def test_buf_timestamps(self):
        sim = _make_mock_sim()
        r0 = _make_received(t_wrf=0.0)
        r1 = _make_received(t_wrf=2.0)
        db = init_lbm_from_wrf(sim, r0, r1, _make_static_ctx(), _make_ref())
        assert db.t0 == pytest.approx(0.0)
        assert db.t1 == pytest.approx(2.0)

    def test_update_freestream_bc_called_once(self):
        """sim.update_freestream_bc 用 step=0 的场调用恰好一次。"""
        sim = _make_mock_sim()
        r0 = _make_received(t_wrf=0.0)
        r1 = _make_received(t_wrf=2.0)
        init_lbm_from_wrf(sim, r0, r1, _make_static_ctx(), _make_ref())
        sim.update_freestream_bc.assert_called_once()

    def test_update_freestream_bc_uses_step0_physical(self):
        """update_freestream_bc 传入 step=0 的物理速度（m/s），不是 step=1，也不是格子单位。"""
        sim = _make_mock_sim()
        r0 = _make_received(t_wrf=0.0, u_val=5.0)
        r1 = _make_received(t_wrf=2.0, u_val=99.0)  # 明显不同
        ref = _make_ref(dx=1.0, dt=0.01)  # u_ref=100
        init_lbm_from_wrf(sim, r0, r1, _make_static_ctx(), ref)
        u_passed = sim.update_freestream_bc.call_args[0][0]
        np.testing.assert_allclose(u_passed, 5.0, rtol=1e-4)

    def test_get_freestream_bc_coordinates_called(self):
        sim = _make_mock_sim()
        r0 = _make_received(t_wrf=0.0)
        r1 = _make_received(t_wrf=2.0)
        init_lbm_from_wrf(sim, r0, r1, _make_static_ctx(), _make_ref(), level=0)
        sim.get_freestream_bc_coordinates.assert_called_once_with(level=0)

    def test_custom_level_passed(self):
        sim = _make_mock_sim()
        r0 = _make_received(t_wrf=0.0)
        r1 = _make_received(t_wrf=2.0)
        init_lbm_from_wrf(sim, r0, r1, _make_static_ctx(), _make_ref(), level=1)
        sim.get_freestream_bc_coordinates.assert_called_once_with(level=1)
        assert sim.update_freestream_bc.call_args[1].get("level") == 1

    def test_high_mach_warns(self):
        """u=50 m/s, u_ref=100 m/s → Ma_lb = 0.5·√3 ≈ 0.87 > 0.3 → warning。"""
        sim = _make_mock_sim()
        r0 = _make_received(t_wrf=0.0, u_val=50.0)
        r1 = _make_received(t_wrf=2.0, u_val=50.0)
        with pytest.warns(UserWarning, match="Mach"):
            init_lbm_from_wrf(sim, r0, r1, _make_static_ctx(), _make_ref(dx=1.0, dt=0.01))

    def test_buf_shapes(self):
        sim = _make_mock_sim(n_bc=N_BC)
        r0 = _make_received(t_wrf=0.0)
        r1 = _make_received(t_wrf=2.0)
        db = init_lbm_from_wrf(sim, r0, r1, _make_static_ctx(), _make_ref())
        for frame in db.buf:
            for field in ("u", "v", "w", "theta"):
                assert getattr(frame, field).shape == (N_BC,), field


# ---------------------------------------------------------------------------
# TestMakeWRFInitialCondition
# ---------------------------------------------------------------------------

class TestMakeWRFInitialCondition:
    """make_wrf_initial_condition 返回一个 ic(x, y, z) → (rho, vx, vy, vz) 闭包。"""

    def _make_ic(self, u_val=5.0, lbm_origin_x=0.0, lbm_origin_y=0.0):
        ref = _make_ref(dx=1.0, dt=0.01)  # velocity_ratio = 100 m/s
        received = _make_received(u_val=u_val)
        ctx = _make_static_ctx()
        return make_wrf_initial_condition(received, ctx, ref,
                                          lbm_origin_x=lbm_origin_x,
                                          lbm_origin_y=lbm_origin_y)

    def test_returns_callable(self):
        ic = self._make_ic()
        assert callable(ic)

    def test_output_is_4_tuple(self):
        ic = self._make_ic()
        result = ic(0.0, 0.0, 500.0)
        assert len(result) == 4

    def test_rho_is_zero_deviation(self):
        """rho 是 VF 的密度偏差 Δρ，参考态为 0.0（不是总密度 1.0）。"""
        ic = self._make_ic()
        rho, vx, vy, vz = ic(0.0, 0.0, 1000.0)
        assert rho == pytest.approx(0.0)

    def test_velocity_in_lattice_units(self):
        """u_phys=5 m/s, velocity_ratio=100 → vx ≈ 0.05。"""
        ic = self._make_ic(u_val=5.0)
        rho, vx, vy, vz = ic(0.0, 0.0, 1000.0)
        # 均匀场，任意高度（范围内）都应返回 5/100 = 0.05
        assert vx == pytest.approx(0.05, rel=1e-3)

    def test_uniform_field_same_at_all_heights(self):
        """均匀速度场 → 不同高度的插值结果相同（范围内）。"""
        ic = self._make_ic(u_val=3.0)
        results = [ic(0.0, 0.0, z) for z in (250.0, 1000.0, 3000.0)]
        vx_vals = [r[1] for r in results]
        for vx in vx_vals:
            assert vx == pytest.approx(3.0 / 100.0, rel=1e-3)

    def test_out_of_bounds_z_clamped(self):
        """超出垂直范围的 z 夹紧，不抛异常。"""
        ic = self._make_ic(u_val=4.0)
        # z=0（低于 z_mass[0] ≈ 250m）→ clamp 到最低层
        rho, vx, vy, vz = ic(0.0, 0.0, 0.0)
        assert np.isfinite(vx)
        # z 极大 → clamp 到最高层
        rho2, vx2, vy2, vz2 = ic(0.0, 0.0, 1e6)
        assert np.isfinite(vx2)

    def test_lbm_origin_offset_applied(self):
        """lbm_origin_x/y 偏移：(x=0,y=0) in LBM == (lbm_origin_x, lbm_origin_y) in WRF。

        用均匀场验证：无论 offset 多少，均匀场的插值结果不变。
        """
        ic_no_offset  = self._make_ic(u_val=6.0, lbm_origin_x=0.0,    lbm_origin_y=0.0)
        ic_with_offset = self._make_ic(u_val=6.0, lbm_origin_x=1000.0, lbm_origin_y=1000.0)
        _, vx1, _, _ = ic_no_offset(0.0, 0.0, 1000.0)
        _, vx2, _, _ = ic_with_offset(0.0, 0.0, 1000.0)
        # 均匀场 → 相同
        assert vx1 == pytest.approx(vx2, rel=1e-5)

    def test_terrain_agl_to_msl(self):
        """非零地形：z_agl=500m 对应 z_msl = terrain + 500m。

        设置 terrain=300m，u = z_mass（MSL）→ 插值结果 ≈ z_msl = 800。
        在 LBM 格子单位下：vx = 800 / velocity_ratio。
        """
        nz, ny, nx, dz = NZ, NY, NX, DZ
        h_terrain = 300.0
        G_ = 9.81
        k = np.arange(nz + 1, dtype=np.float64)
        ph_total = (G_ * (h_terrain + k[:, None, None] * dz)
                    * np.ones((nz + 1, ny, nx))).astype(np.float32)
        z_w = ph_total.astype(np.float64) / G_
        z_mass_arr = 0.5 * (z_w[:-1] + z_w[1:])
        # u = z_mass（MSL） 让插值结果等于 z_msl
        u = z_mass_arr.astype(np.float32)
        received = {
            "__wrf_time_s": 0.0,
            "u":     u,
            "v":     np.zeros_like(u),
            "w":     np.zeros((nz + 1, ny, nx), dtype=np.float32),
            "theta": 310.0 * np.ones((nz, ny, nx), dtype=np.float32),
            "ph":    ph_total,
        }
        terrain = h_terrain * np.ones((ny, nx))
        ctx = WRFStaticContext(terrain_height=terrain, dx_wrf=1000.0, dy_wrf=1000.0)
        ref = _make_ref(dx=1.0, dt=0.01)  # velocity_ratio = 100
        ic = make_wrf_initial_condition(received, ctx, ref)
        z_agl = 500.0
        # z_msl = 300 + 500 = 800m → u_phys ≈ 800 m/s → vx ≈ 8.0 (LBM units)
        rho, vx, vy, vz = ic(0.0, 0.0, z_agl)
        expected_vx = (h_terrain + z_agl) / 100.0  # 8.0
        assert vx == pytest.approx(expected_vx, rel=1e-2)
