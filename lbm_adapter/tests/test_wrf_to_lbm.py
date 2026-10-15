"""任务5单元测试：lbm_adapter/wrf_to_lbm.py

纯 numpy，无 MPI，无 JAX，无 GPU。
合成数据量级与真实 WRF d03 一致（ph O(10^5)，theta ~300-470 K）。

运行方法（在 wrf_gpu/ 目录下）:
    python -m pytest lbm_adapter/tests/test_wrf_to_lbm.py -v
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lbm_adapter.wrf_to_lbm import (
    WRFStaticContext,
    make_wrf_column_data,
    interp_to_bc_nodes,
    extract_most_fields,
    WRFColumnData,
    InterpolatedFields,
    MOSTSurfaceFields,
)
from lbm_adapter.vertical_interp import compute_z_w, compute_z_mass

G = 9.81  # m s⁻²

# ---------------------------------------------------------------------------
# 真实量级的合成数据：z_w ∈ [0, 15000 m]，dz ~ 500 m/层
# 对应 ph_total = G * z_w，范围 O(10^4-10^5)
# ---------------------------------------------------------------------------
REAL_DZ = 500.0     # m per layer（15km / 30层）
REAL_NZ = 30        # 层数
REAL_NY = 4
REAL_NX = 4
REAL_DX = 1000.0    # d03 格距 m


def _make_ph_total(nz: int, ny: int, nx: int, dz: float = REAL_DZ) -> np.ndarray:
    """均匀层距：z_w[k] = k * dz，ph_total[k] = G * k * dz。
    nz=30, dz=500 → z_w[30] = 15000m，ph_total[30] = 147150 m²/s²。
    """
    k = np.arange(nz + 1, dtype=np.float64)
    return G * k[:, None, None] * dz * np.ones((nz + 1, ny, nx))


def _make_static_ctx(nz=REAL_NZ, ny=REAL_NY, nx=REAL_NX,
                     dx_wrf=REAL_DX, dy_wrf=REAL_DX) -> WRFStaticContext:
    return WRFStaticContext(
        terrain_height=np.zeros((ny, nx)),
        dx_wrf=dx_wrf,
        dy_wrf=dy_wrf,
    )


def _make_received(nz=REAL_NZ, ny=REAL_NY, nx=REAL_NX, dz=REAL_DZ,
                   u_val=5.0, v_val=-3.0, w_val=0.1, theta_val=310.0,
                   with_most=True) -> dict:
    """构造与 mock_receiver 接收格式相同的合成 dict。
    ph 是 ph_total（O(10^5)），theta 是 full θ（~300-470 K）。
    """
    ph = _make_ph_total(nz, ny, nx, dz).astype(np.float32)
    d: dict = {
        "__wrf_time_s": 4.0,
        "u":     u_val    * np.ones((nz,     ny, nx), dtype=np.float32),
        "v":     v_val    * np.ones((nz,     ny, nx), dtype=np.float32),
        "w":     w_val    * np.ones((nz + 1, ny, nx), dtype=np.float32),
        "theta": theta_val * np.ones((nz,   ny, nx), dtype=np.float32),
        "ph":    ph,
    }
    if with_most:
        d["ustar"]       = 0.5  * np.ones((ny, nx), dtype=np.float32)
        d["roughness_m"] = 0.1  * np.ones((ny, nx), dtype=np.float32)
        d["psim"]        = np.zeros((ny, nx), dtype=np.float32)
    return d


# ---------------------------------------------------------------------------
# TestWRFStaticContext
# ---------------------------------------------------------------------------

class TestWRFStaticContext:
    def test_from_dict(self):
        ny, nx = 3, 4
        d = {"terrain_height": np.zeros((ny, nx), dtype=np.float32),
             "dx_wrf": 333.0, "dy_wrf": 333.0}
        ctx = WRFStaticContext.from_dict(d)
        assert ctx.terrain_height.dtype == np.float64
        assert ctx.dx_wrf == pytest.approx(333.0)

    def test_no_phb_field(self):
        """WRFStaticContext 不包含 phb（ph 已是总位势）。"""
        ctx = _make_static_ctx()
        assert not hasattr(ctx, "phb")

    def test_terrain_shape(self):
        ctx = _make_static_ctx(ny=5, nx=6)
        assert ctx.terrain_height.shape == (5, 6)


# ---------------------------------------------------------------------------
# TestPhTotalRange：验证合成 ph 量级正确
# ---------------------------------------------------------------------------

class TestPhTotalRange:
    def test_ph_range_realistic(self):
        """ph_total 范围应为 O(10^4-10^5)（地面 ~0，顶部 ~147000 m²/s²）。"""
        ph = _make_ph_total(REAL_NZ, REAL_NY, REAL_NX, REAL_DZ)
        assert ph.min() == pytest.approx(0.0)
        assert ph.max() == pytest.approx(G * REAL_NZ * REAL_DZ, rel=1e-6)
        # nz=30, dz=500 → max = 9.81 * 30 * 500 = 147150
        assert ph.max() > 1e4

    def test_zw_from_ph_range(self):
        """z_w = ph / g，地面 0m，顶部 REAL_NZ * REAL_DZ m。"""
        ph = _make_ph_total(REAL_NZ, REAL_NY, REAL_NX, REAL_DZ)
        z_w = compute_z_w(ph)
        assert z_w.min() == pytest.approx(0.0, abs=1e-6)
        assert z_w.max() == pytest.approx(REAL_NZ * REAL_DZ, rel=1e-6)

    def test_zw_top_in_realistic_range(self):
        """模型顶部高度应在 10-20 km 范围内（真实 WRF d03 典型值）。"""
        ph = _make_ph_total(REAL_NZ, REAL_NY, REAL_NX, REAL_DZ)
        z_w = compute_z_w(ph)
        z_top = float(z_w[-1, 0, 0])
        assert 10000.0 <= z_top <= 20000.0


# ---------------------------------------------------------------------------
# TestMakeWRFColumnData
# ---------------------------------------------------------------------------

class TestMakeWRFColumnData:
    def test_ph_total_direct_use(self):
        """ph_total = received['ph']，不叠加 phb。"""
        received = _make_received()
        ctx = _make_static_ctx()
        wrf_data = make_wrf_column_data(received, ctx)
        np.testing.assert_allclose(
            wrf_data.ph_total,
            received["ph"].astype(np.float64),
            rtol=1e-5,
        )

    def test_ph_total_range_realistic(self):
        """组装后的 ph_total 应在 O(10^4-10^5) 范围。"""
        received = _make_received()
        ctx = _make_static_ctx()
        wrf_data = make_wrf_column_data(received, ctx)
        assert wrf_data.ph_total.max() > 1e4

    def test_field_shapes(self):
        nz, ny, nx = REAL_NZ, REAL_NY, REAL_NX
        received = _make_received(nz=nz, ny=ny, nx=nx)
        ctx = _make_static_ctx(nz=nz, ny=ny, nx=nx)
        wrf_data = make_wrf_column_data(received, ctx)
        assert wrf_data.u.shape      == (nz, ny, nx)
        assert wrf_data.v.shape      == (nz, ny, nx)
        assert wrf_data.theta.shape  == (nz, ny, nx)
        assert wrf_data.w.shape      == (nz + 1, ny, nx)
        assert wrf_data.ph_total.shape == (nz + 1, ny, nx)

    def test_output_dtype_float64(self):
        received = _make_received()
        ctx = _make_static_ctx()
        wrf_data = make_wrf_column_data(received, ctx)
        for attr in ("ph_total", "u", "v", "w", "theta"):
            assert getattr(wrf_data, attr).dtype == np.float64, attr

    def test_theta_full_range(self):
        """theta 是 full θ，范围约 300-470 K（不是偏差量 ~0-170 K）。"""
        received = _make_received(theta_val=320.0)
        ctx = _make_static_ctx()
        wrf_data = make_wrf_column_data(received, ctx)
        # full θ 应在 300-470 K 范围内
        assert wrf_data.theta.min() >= 200.0   # 宽松下界（避免误判）
        assert wrf_data.theta.max() <= 600.0   # 宽松上界
        # 具体值应约等于 theta_val=320
        np.testing.assert_allclose(wrf_data.theta, 320.0, rtol=1e-5)

    def test_control_keys_ignored(self):
        received = _make_received()
        received["__extra_control"] = 999.0
        ctx = _make_static_ctx()
        wrf_data = make_wrf_column_data(received, ctx)
        assert wrf_data is not None

    def test_dx_dy_passed_through(self):
        received = _make_received()
        ctx = _make_static_ctx(dx_wrf=500.0, dy_wrf=400.0)
        wrf_data = make_wrf_column_data(received, ctx)
        assert wrf_data.dx_wrf == pytest.approx(500.0)
        assert wrf_data.dy_wrf == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# TestExtractMostFields
# ---------------------------------------------------------------------------

class TestExtractMostFields:
    def test_extracts_when_present(self):
        received = _make_received(with_most=True)
        most = extract_most_fields(received)
        assert most is not None
        assert most.ustar.shape == (REAL_NY, REAL_NX)

    def test_returns_none_when_missing(self):
        received = _make_received(with_most=False)
        assert extract_most_fields(received) is None

    def test_partial_missing_returns_none(self):
        received = _make_received(with_most=False)
        received["ustar"] = np.ones((REAL_NY, REAL_NX))
        assert extract_most_fields(received) is None

    def test_dtype_float64(self):
        received = _make_received(with_most=True)
        most = extract_most_fields(received)
        assert most.ustar.dtype == np.float64
        assert most.roughness_m.dtype == np.float64
        assert most.psim.dtype == np.float64


# ---------------------------------------------------------------------------
# TestInterpToBcNodes
# ---------------------------------------------------------------------------

class TestInterpToBcNodes:
    def _setup(self, u_val=5.0, v_val=-3.0, theta_val=310.0):
        received = _make_received(u_val=u_val, v_val=v_val, theta_val=theta_val)
        ctx = _make_static_ctx()
        return make_wrf_column_data(received, ctx)

    def test_uniform_field_at_any_height(self):
        """均匀速度场 → BC 节点处插值结果 == 原值。"""
        wrf_data = self._setup(u_val=7.0, v_val=-2.0)
        # 目标高度在 z_mass 范围内（dz=500m，z_mass[0]=250m，z_mass[-1]=14750m）
        bc_coords = np.array([[0.0, 0.0, 500.0],
                               [0.0, 0.0, 2000.0],
                               [0.0, 0.0, 8000.0]])
        result = interp_to_bc_nodes(wrf_data, bc_coords)
        np.testing.assert_allclose(result.u, 7.0,  rtol=1e-6)
        np.testing.assert_allclose(result.v, -2.0, rtol=1e-6)

    def test_theta_output_is_full_temperature(self):
        """interp_to_bc_nodes 输出的 theta 是 full θ（~300-470 K），不是偏差。"""
        wrf_data = self._setup(theta_val=315.0)
        bc_coords = np.array([[0.0, 0.0, 1000.0],
                               [0.0, 0.0, 5000.0]])
        result = interp_to_bc_nodes(wrf_data, bc_coords)
        # 应约等于 315 K（full θ），不是 15 K（偏差）
        np.testing.assert_allclose(result.theta, 315.0, rtol=1e-5)
        assert result.theta.min() > 200.0  # 不可能是偏差量（<100 K）

    def test_theta_not_offset(self):
        """验证 theta 没有被减去 300：如果误减了，结果会是 ~10-170 K。"""
        wrf_data = self._setup(theta_val=320.0)
        bc_coords = np.array([[0.0, 0.0, 1000.0]])
        result = interp_to_bc_nodes(wrf_data, bc_coords)
        # 若正确（full θ），值约 320；若误减了 300，值约 20
        assert result.theta[0] > 200.0, (
            f"theta={result.theta[0]:.1f} K — 疑似已减去 300，应保持 full θ"
        )

    def test_bc_coords_shape_check(self):
        wrf_data = self._setup()
        with pytest.raises(ValueError, match="n_bc, 3"):
            interp_to_bc_nodes(wrf_data, np.ones((5, 2)))

    def test_bc_coords_1d_raises(self):
        wrf_data = self._setup()
        with pytest.raises(ValueError):
            interp_to_bc_nodes(wrf_data, np.array([0.0, 0.0, 1000.0]))

    def test_vertical_gradient_round_trip(self):
        """u = z_mass 时，插值到目标高度应还原目标高度值。"""
        nz, ny, nx, dz = REAL_NZ, REAL_NY, REAL_NX, REAL_DZ
        ph = _make_ph_total(nz, ny, nx, dz)
        z_w = compute_z_w(ph)
        z_mass = compute_z_mass(z_w)
        u = z_mass.copy().astype(np.float32)
        received = {
            "__wrf_time_s": 0.0,
            "u":     u,
            "v":     np.zeros_like(u),
            "w":     np.zeros((nz + 1, ny, nx), dtype=np.float32),
            "theta": 310.0 * np.ones((nz, ny, nx), dtype=np.float32),
            "ph":    ph.astype(np.float32),
        }
        ctx = _make_static_ctx(nz=nz, ny=ny, nx=nx)
        wrf_data = make_wrf_column_data(received, ctx)

        # 在 z_mass 范围内取目标高度（不越界）
        z_floor = float(z_mass[0, 0, 0])
        z_top   = float(z_mass[-1, 0, 0])
        targets = np.array([z_floor + 50, z_floor + 1000, z_floor + 5000])
        targets = targets[targets < z_top]
        bc_coords = np.column_stack([np.zeros(len(targets)),
                                     np.zeros(len(targets)),
                                     targets])
        result = interp_to_bc_nodes(wrf_data, bc_coords)
        np.testing.assert_allclose(result.u, targets, rtol=1e-4, atol=1.0)

    def test_with_most_fields_patches_below_floor(self):
        """提供 most_fields 时，低于 z_mass[0] 的节点不再被 clamp 到最低层。"""
        nz, ny, nx, dz = REAL_NZ, REAL_NY, REAL_NX, REAL_DZ
        received = _make_received(u_val=8.0)
        ctx = _make_static_ctx()
        wrf_data = make_wrf_column_data(received, ctx)
        # z_mass[0] = dz/2 = 250m
        most = MOSTSurfaceFields(
            ustar=0.5 * np.ones((ny, nx)),
            roughness_m=0.1 * np.ones((ny, nx)),
            psim=np.zeros((ny, nx)),
        )
        bc_coords = np.array([[0.0, 0.0, 5.0]])  # 5m < 250m
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result_most  = interp_to_bc_nodes(wrf_data, bc_coords, most_fields=most)
        result_plain = interp_to_bc_nodes(wrf_data, bc_coords)
        assert abs(result_most.u[0] - result_plain.u[0]) > 0.5

    def test_output_is_interpolated_fields(self):
        wrf_data = self._setup()
        bc_coords = np.array([[0.0, 0.0, 1000.0]])
        result = interp_to_bc_nodes(wrf_data, bc_coords)
        assert isinstance(result, InterpolatedFields)
        for attr in ("u", "v", "w", "theta"):
            assert hasattr(result, attr)

    def test_multiple_bc_nodes_shape(self):
        wrf_data = self._setup()
        n_bc = 7
        bc_coords = np.column_stack([
            np.linspace(0, 3000, n_bc),
            np.zeros(n_bc),
            np.linspace(500, 8000, n_bc),
        ])
        result = interp_to_bc_nodes(wrf_data, bc_coords)
        for attr in ("u", "v", "w", "theta"):
            assert getattr(result, attr).shape == (n_bc,)

    def test_out_of_bounds_x_raises(self):
        """BC 节点超出 WRF 水平网格范围 → ValueError。"""
        received = _make_received(nx=4)
        ctx = _make_static_ctx(nx=4, dx_wrf=1000.0)  # x_max = 3000m
        wrf_data = make_wrf_column_data(received, ctx)
        bc_coords = np.array([[9999.0, 0.0, 1000.0]])
        with pytest.raises(ValueError, match="outside"):
            interp_to_bc_nodes(wrf_data, bc_coords)

    def test_agl_to_msl_with_terrain(self):
        """z_coord_is_agl=True：AGL 高度自动加上地形高度后再插值。

        设置地形高度 h_terrain = 300 m，均匀速度场 u = z_mass（MSL）。
        AGL=250 m 的节点对应 MSL = 300+250 = 550 m，插值结果应约为 550。
        若未加地形（即把 AGL 当 MSL），则插值到 250 m，结果约为 250。
        """
        nz, ny, nx, dz = REAL_NZ, REAL_NY, REAL_NX, REAL_DZ
        h_terrain = 300.0  # m MSL

        # ph_total 从 h_terrain 开始（地形抬高）
        k = np.arange(nz + 1, dtype=np.float64)
        ph_total = (G * (h_terrain + k[:, None, None] * dz)
                    * np.ones((nz + 1, ny, nx))).astype(np.float32)

        z_w = ph_total.astype(np.float64) / G
        z_mass = 0.5 * (z_w[:-1] + z_w[1:])  # z_mass[0] = h_terrain + dz/2 = 550m

        # u = z_mass（MSL），这样插值结果等于目标 MSL 高度
        u = z_mass.astype(np.float32)

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
        wrf_data = make_wrf_column_data(received, ctx)

        # AGL = 500m → MSL = 300 + 500 = 800m → u 插值结果 ≈ 800
        z_agl = 500.0
        bc_coords_agl = np.array([[0.0, 0.0, z_agl]])

        result_agl = interp_to_bc_nodes(wrf_data, bc_coords_agl, z_coord_is_agl=True)
        result_msl = interp_to_bc_nodes(wrf_data, bc_coords_agl, z_coord_is_agl=False)

        expected_msl = h_terrain + z_agl  # 800.0
        # AGL 模式：加了地形，插值到 800m → u ≈ 800
        np.testing.assert_allclose(result_agl.u[0], expected_msl, rtol=1e-3,
                                   err_msg="AGL mode should give MSL-equivalent result")
        # MSL 模式：直接用 500m 插值 → u ≈ 500，但 500 < z_mass[0]=550 → clamp 到 550
        # 无论如何，两者结果应不同
        assert abs(result_agl.u[0] - result_msl.u[0]) > 100.0, (
            f"AGL and MSL modes should differ for non-zero terrain: "
            f"agl={result_agl.u[0]:.1f}, msl={result_msl.u[0]:.1f}"
        )
