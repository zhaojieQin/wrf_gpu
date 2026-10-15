"""LBM侧 WRF场接收与空间插值接口（任务5）.

本模块负责将 MPI 接收到的 WRF numpy 字段 dict 转换为
VirtualFluids BC 节点处的插值结果。

依赖：纯 numpy，无 JAX，无 mpi4py。
可在 LBM 进程独立 import，无需 gpuwrf 包。

公开接口
--------
make_wrf_column_data(received)        -> WRFColumnData
interp_to_bc_nodes(wrf_data, bc_coords, most_fields=None) -> InterpolatedFields

WRF 侧发送的字段（由 coupling_adapter.py 的 subdomain_config 决定）
-----------------------------------------------------------------
3D 场（nz, ny, nx）:
  u, v, theta   — 质量层 (mass level)
    theta 是 full moist potential temperature θ_m（~300-470 K），NOT θ'
    减 300 在任务 6 的 apply_unit_conversions 里做。

3D 场（nz+1, ny, nx）:
  w             — w层 (w-level)
  ph            — 总位势 φ_total (m² s⁻²)，= state.ph_total = state.ph
                  范围 O(10^4-10^5)，对应 z_w = ph/g ∈ [0, 15000] m

2D 场（ny, nx）:
  hfx  — 地表感热通量 H₀（W m⁻²），驱动 VF SurfaceLayerBoundaryCondition
         稳定/不稳定大气下的 Monin-Obukhov 相似理论。
         正值=向上（不稳定），负值=向下（稳定）。

控制键（不对应 MPI 数组包）:
  __wrf_time_s  — 当前步 WRF 物理时间（秒）

设计说明
--------
WRF State.ph 是 ph_total（state.py 中 ph 是 ph_total 的 property alias）。
不需要 phb：ph_total 直接用于计算 z_w = ph_total / g。

bc_coords 约定
--------------
sim.get_velocity_bc_coordinates(level=0) 返回 (n_bc, 3) 数组，
列顺序：[x_m, y_m, z_msl_m]，均以 m 为单位，
x/y 以 WRF d03 西南角为原点。

theta 输出约定
--------------
interp_to_bc_nodes 输出的 theta 是 full moist potential temperature（K），
范围约 300-470 K。减 300 并转换到 LBM 格子单位在任务 6 的
apply_unit_conversions(received, ref) 里完成。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# 直接 import 复制到 lbm_adapter/ 的副本（无 gpuwrf 包依赖）
from lbm_adapter.vertical_interp import (
    WRFColumnData,
    InterpolatedFields,
    MOSTSurfaceFields,
    interp_wrf_to_lbm_nodes,
)
from lbm_adapter.unit_convert import LBMReference, uvw_to_lbm, theta_to_lbm


# ---------------------------------------------------------------------------
# WRF 静态上下文（每次运行初始化一次）
# ---------------------------------------------------------------------------

@dataclass
class WRFStaticContext:
    """每次运行初始化一次的静态量。

    Attributes
    ----------
    terrain_height : (ny, nx)  m   WRF d03 地形高度
    dx_wrf         : float     m   d03 x方向格距
    dy_wrf         : float     m   d03 y方向格距
    z0_mean        : float     m   d03 陆地点粗糙度长度全域平均值（从 wrfinput_d03/ZNT 读取）

    Notes
    -----
    不含 phb：WRF 发送的 ph 已经是总位势 ph_total（= PHB + PH'），
    直接用于 z_w = ph / g，无需叠加 phb。

    z0_mean 在 surfaceLayerBoundaryCondition 设置时传入 VF（init 阶段），
    运行时不再更新。
    """
    terrain_height: np.ndarray
    dx_wrf: float
    dy_wrf: float
    z0_mean: float = 0.1  # m, 典型陆地粗糙度

    @classmethod
    def from_dict(cls, d: dict) -> "WRFStaticContext":
        """从字典构造（键名与 LBMStaticContext.to_dict() 兼容）。"""
        return cls(
            terrain_height=np.asarray(d["terrain_height"], dtype=np.float64),
            dx_wrf=float(d["dx_wrf"]),
            dy_wrf=float(d["dy_wrf"]),
            z0_mean=float(d.get("z0_mean", 0.1)),
        )


# ---------------------------------------------------------------------------
# 公开接口 1：make_wrf_column_data
# ---------------------------------------------------------------------------

def make_wrf_column_data(received: dict, static_ctx: WRFStaticContext) -> WRFColumnData:
    """将 MPI 接收的 numpy 字段 dict 组装为 WRFColumnData。

    Parameters
    ----------
    received    : 从 mock_receiver / recv_loop 收到的 dict，包含键：
                  'u', 'v', 'w', 'theta', 'ph'（总位势，O(10^5) m² s⁻²）,
                  可选 'ustar'/'roughness_m'/'psim'，
                  控制键 '__wrf_time_s' 等会被忽略。
    static_ctx  : WRFStaticContext，含 terrain_height/dx_wrf/dy_wrf。

    Returns
    -------
    WRFColumnData 可直接传给 interp_to_bc_nodes。

    Notes
    -----
    ph 字段是 ph_total（state.ph = state.ph_total，WRF 代码 state.py:815）。
    范围 O(10^4-10^5) m² s⁻²，对应 z_w = ph / g ∈ [0, ~15000] m。

    theta 字段是 full moist potential temperature θ_m（~300-470 K）。
    不在此处减 300；减 300 在任务 6 的 apply_unit_conversions 里做。
    """
    return WRFColumnData(
        ph_total=np.asarray(received["ph"], dtype=np.float64),
        u=np.asarray(received["u"], dtype=np.float64),
        v=np.asarray(received["v"], dtype=np.float64),
        w=np.asarray(received["w"], dtype=np.float64),
        theta=np.asarray(received["theta"], dtype=np.float64),
        terrain_height=static_ctx.terrain_height,
        dx_wrf=static_ctx.dx_wrf,
        dy_wrf=static_ctx.dy_wrf,
    )


# ---------------------------------------------------------------------------
# MOST 场提取辅助
# ---------------------------------------------------------------------------

def extract_most_fields(received: dict) -> MOSTSurfaceFields | None:
    """从 received dict 提取 MOST 地表诊断场（若存在）。

    若 received 中缺少 ustar/roughness_m/psim 任意一个，返回 None。
    """
    keys = ("ustar", "roughness_m", "psim")
    if not all(k in received for k in keys):
        return None
    return MOSTSurfaceFields(
        ustar=np.asarray(received["ustar"], dtype=np.float64),
        roughness_m=np.asarray(received["roughness_m"], dtype=np.float64),
        psim=np.asarray(received["psim"], dtype=np.float64),
    )


def extract_hfx(received: dict) -> np.ndarray | None:
    """从 received dict 提取地表感热通量 hfx（若存在）。

    Parameters
    ----------
    received : MPI 接收的 dict

    Returns
    -------
    hfx : (ny, nx) float64，单位 W m⁻²，正值向上（不稳定），负值向下（稳定）。
          若 'hfx' 键不存在，返回 None。
    """
    if "hfx" not in received:
        return None
    return np.asarray(received["hfx"], dtype=np.float64)


# ---------------------------------------------------------------------------
# 公开接口 2：interp_to_bc_nodes
# ---------------------------------------------------------------------------

def interp_to_bc_nodes(
    wrf_data: WRFColumnData,
    bc_coords: np.ndarray,
    most_fields: "MOSTSurfaceFields | None" = None,
    z_coord_is_agl: bool = True,
) -> InterpolatedFields:
    """将 WRF 场空间插值到 LBM BC 节点位置。

    Parameters
    ----------
    wrf_data   : WRFColumnData，由 make_wrf_column_data() 构造。
    bc_coords  : (n_bc, 3) float64，列顺序 [x_m, y_m, z_m]。
                 由 sim.get_velocity_bc_coordinates(level=0) 返回。
                 x/y 以 WRF d03 西南角为原点（m）。
    most_fields : MOSTSurfaceFields | None。
                  若提供，对低于 z_mass[0] 的节点用 MOST 代替线性外推。
    z_coord_is_agl : bool，默认 True。
                 True  — bc_coords[:,2] 是 AGL（VirtualFluids 默认，从地面 0 开始）。
                         函数内部按 WRF 最近邻格点查 terrain_height，转为 MSL 后插值。
                 False — bc_coords[:,2] 已经是 MSL（海拔），直接使用。

    Returns
    -------
    InterpolatedFields 包含：
      u, v, w  — 物理速度（m s⁻¹），未转换为 LBM 格子单位
      theta    — full moist potential temperature（K），范围约 300-470 K
                 注意：这不是偏差量。减 300 并转换 LBM 单位在任务 6
                 的 apply_unit_conversions(received, ref) 里完成。

    Raises
    ------
    ValueError  若 bc_coords 不是 (n_bc, 3) 形状，或节点超出 WRF 水平网格。
    """
    bc_coords = np.asarray(bc_coords, dtype=np.float64)
    if bc_coords.ndim != 2 or bc_coords.shape[1] != 3:
        raise ValueError(
            f"bc_coords must be (n_bc, 3), got shape {bc_coords.shape}"
        )
    target_x = bc_coords[:, 0]
    target_y = bc_coords[:, 1]
    target_z_in = bc_coords[:, 2]

    if z_coord_is_agl:
        # AGL → MSL：按节点的水平位置查最近邻 WRF 格点的地形高度并相加。
        # 与 interp_field_to_heights 内部的水平查找方式保持一致（nearest-neighbour）。
        terrain = np.asarray(wrf_data.terrain_height, dtype=np.float64)
        ny, nx = terrain.shape
        j_idx = np.clip(
            np.round(target_y / wrf_data.dy_wrf).astype(int), 0, ny - 1
        )
        i_idx = np.clip(
            np.round(target_x / wrf_data.dx_wrf).astype(int), 0, nx - 1
        )
        terrain_at_node = terrain[j_idx, i_idx]  # (n_bc,) 每节点对应的地形高度
        target_z = target_z_in + terrain_at_node
    else:
        target_z = target_z_in

    return interp_wrf_to_lbm_nodes(
        wrf_data,
        target_z=target_z,
        target_x=target_x,
        target_y=target_y,
        most_fields=most_fields,
    )


# ---------------------------------------------------------------------------
# 公开接口 3：interp_hfx_to_sl_nodes
# ---------------------------------------------------------------------------

def interp_hfx_to_sl_nodes(
    hfx_2d: np.ndarray,
    sl_coords: np.ndarray,
    static_ctx: WRFStaticContext,
) -> np.ndarray:
    """将 WRF hfx (ny, nx) 双线性插值到 surfaceLayerBC 节点位置。

    Parameters
    ----------
    hfx_2d     : (ny, nx) float64，W m⁻²，WRF 原始网格
    sl_coords  : (n_sl, 2) float64，列顺序 [x_m, y_m]，
                 以 WRF d03 西南角为原点（m）
    static_ctx : WRFStaticContext，提供 dx_wrf/dy_wrf

    Returns
    -------
    hfx_nodes : (n_sl,) float64，W m⁻²，在各 surfaceLayerBC 节点处的插值值
    """
    hfx_2d = np.asarray(hfx_2d, dtype=np.float64)
    sl_coords = np.asarray(sl_coords, dtype=np.float64)
    ny, nx = hfx_2d.shape
    dx = static_ctx.dx_wrf
    dy = static_ctx.dy_wrf

    x_nodes = sl_coords[:, 0]
    y_nodes = sl_coords[:, 1]

    # Fractional grid indices (0-based)
    xi = x_nodes / dx
    yi = y_nodes / dy

    # Floor indices, clamped to valid range
    i0 = np.clip(np.floor(xi).astype(int), 0, nx - 2)
    j0 = np.clip(np.floor(yi).astype(int), 0, ny - 2)
    i1 = i0 + 1
    j1 = j0 + 1

    # Fractional weights
    tx = np.clip(xi - i0, 0.0, 1.0)
    ty = np.clip(yi - j0, 0.0, 1.0)

    # Bilinear interpolation: hfx[j, i] (row=y, col=x)
    hfx_nodes = (
        (1.0 - ty) * (1.0 - tx) * hfx_2d[j0, i0]
        + (1.0 - ty) * tx        * hfx_2d[j0, i1]
        + ty         * (1.0 - tx) * hfx_2d[j1, i0]
        + ty         * tx         * hfx_2d[j1, i1]
    )
    return hfx_nodes
