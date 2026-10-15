"""任务7：用 WRF step=0 数据初始化 LBM（VirtualFluids）。

公开接口
--------
init_lbm_from_wrf(sim, para, received_step0, received_step1,
                  static_ctx, ref, level=0)
    -> WRFDoubleBuffer

设计说明
--------
VirtualFluids 初始化需要：
  1. para.set_initial_condition(fn)  — 在 sim 构造前设置初始场
  2. sim.init_timers()               — sim 构造后、第一步前调用

本模块的职责：
  - 接收 WRF step=0 和 step=1 两帧数据（用于立即开始时间插值）
  - 从 step=0 插值到 BC 节点，用于 update_velocity_bc（首帧 BC）
  - 保存 buf[0]=step0, buf[1]=step1 到 WRFDoubleBuffer，供主循环使用
  - 不负责构造 sim/para（由调用方在 sim 构造前完成 set_initial_condition）

WRFDoubleBuffer
---------------
保存两个 WRF 时间帧（buf[0], buf[1]）及对应的物理时间戳，
用于主循环中的双缓冲时间插值。
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from lbm_adapter.wrf_to_lbm import (
    WRFStaticContext,
    make_wrf_column_data,
    interp_to_bc_nodes,
    extract_most_fields,
    extract_hfx,
    interp_hfx_to_sl_nodes,
    WRFColumnData,
    InterpolatedFields,
)
from lbm_adapter.unit_convert import LBMReference, apply_unit_conversions
from lbm_adapter.vertical_interp import (
    compute_z_w,
    compute_z_mass,
    _interp_column_1d,
)


# ---------------------------------------------------------------------------
# 双缓冲数据结构
# ---------------------------------------------------------------------------

class WRFFrame(NamedTuple):
    """单个 WRF 时间帧（在 BC 节点处插值完毕）。

    速度保持物理单位：sim.update_freestream_bc 接收 m/s，
    并在 C++ 内部（Simulation::updateFreestreamBC）自行除以 velocityRatio。

    Attributes
    ----------
    t_wrf     : float  WRF 物理时间（秒）
    u, v, w   : (n_bc,)  物理速度（m s⁻¹）
    theta     : (n_bc,)  LBM 浓度（K 偏差，= θ_full − theta_ref）
    hfx       : (n_sl,) or None  地表感热通量（W m⁻²），已插值到 surfaceLayerBC 节点
    """
    t_wrf: float
    u: np.ndarray              # (n_bc,) m/s
    v: np.ndarray              # (n_bc,) m/s
    w: np.ndarray              # (n_bc,) m/s
    theta: np.ndarray          # (n_bc,) K deviation
    hfx: np.ndarray | None     # (n_sl,) W m⁻², or None if not sent


@dataclass
class WRFDoubleBuffer:
    """双缓冲：两个相邻 WRF 时间帧，用于线性时间插值。

    约定：buf[0].t_wrf <= buf[1].t_wrf。

    Methods
    -------
    interp(t_lbm)  -> InterpolatedBCFields
        在 [buf[0].t_wrf, buf[1].t_wrf] 之间做线性插值。
    advance(new_frame)
        丢弃 buf[0]，buf[1] 升为 buf[0]，new_frame 存入 buf[1]。
    needs_advance(t_lbm)
        返回 True 如果 t_lbm >= buf[1].t_wrf（需要换页）。
    """
    buf: list  # [WRFFrame, WRFFrame]

    @property
    def t0(self) -> float:
        return self.buf[0].t_wrf

    @property
    def t1(self) -> float:
        return self.buf[1].t_wrf

    def needs_advance(self, t_lbm: float) -> bool:
        """LBM 当前物理时间超出 buf[1] → 需要换页。"""
        return t_lbm >= self.t1

    def interp(self, t_lbm: float):
        """在两帧之间线性插值，返回 (u, v, w, theta, hfx)。

        如果 t_lbm 超出 [t0, t1]，夹紧到边界（不外推）。

        Returns
        -------
        u, v, w : (n_bc,)  物理速度（m s⁻¹）
        theta   : (n_bc,)  K deviation
        hfx     : (ny, nx) W m⁻²，或 None（若两帧均无 hfx）
        """
        dt = self.t1 - self.t0
        if dt <= 0.0:
            alpha = 1.0
        else:
            alpha = float(np.clip((t_lbm - self.t0) / dt, 0.0, 1.0))
        f0, f1 = self.buf[0], self.buf[1]
        u = (1.0 - alpha) * f0.u     + alpha * f1.u
        v = (1.0 - alpha) * f0.v     + alpha * f1.v
        w = (1.0 - alpha) * f0.w     + alpha * f1.w
        theta = (1.0 - alpha) * f0.theta + alpha * f1.theta
        if f0.hfx is not None and f1.hfx is not None:
            hfx = (1.0 - alpha) * f0.hfx + alpha * f1.hfx
        elif f1.hfx is not None:
            hfx = f1.hfx
        else:
            hfx = f0.hfx
        return u, v, w, theta, hfx

    def advance(self, new_frame: WRFFrame) -> None:
        """换页：buf[0] ← buf[1]，buf[1] ← new_frame。"""
        self.buf[0] = self.buf[1]
        self.buf[1] = new_frame


# ---------------------------------------------------------------------------
# 辅助：received dict → WRFFrame（在 BC 节点处插值 + 单位转换）
# ---------------------------------------------------------------------------

def _make_wrf_frame(
    received: dict,
    static_ctx: WRFStaticContext,
    bc_coords: np.ndarray,
    ref: LBMReference,
    sl_coords: np.ndarray | None = None,
) -> WRFFrame:
    """将一个 received dict 转换为 WRFFrame（BC 节点插值 + 单位转换）。

    Parameters
    ----------
    received   : 从 MPI 接收的原始 dict（含 '__wrf_time_s'）
    static_ctx : WRFStaticContext（地形/格距/z0_mean）
    bc_coords  : (n_bc, 3) BC 节点坐标 [x_m, y_m, z_msl_m]
    ref        : LBMReference（dx_lbm, dt_lbm, theta_ref）
    sl_coords  : (n_sl, 2) surfaceLayerBC 节点坐标 [x_m, y_m]，或 None。
                 若为 None，hfx 不做空间插值，直接存 None。

    Returns
    -------
    WRFFrame  含物理时间戳；u/v/w 为物理速度（m/s），theta 为 K 偏差；
              hfx 已插值到 (n_sl,) 或 None
    """
    t_wrf = float(received.get("__wrf_time_s", 0.0))

    # 1. 组装 WRFColumnData
    wrf_data = make_wrf_column_data(received, static_ctx)

    # 2. 提取可选 MOST 场（已废弃，但保留兼容性）
    most_fields = extract_most_fields(received)

    # 3. 插值到 BC 节点（物理单位）
    interped: InterpolatedFields = interp_to_bc_nodes(
        wrf_data, bc_coords, most_fields=most_fields
    )

    # 4. 单位转换：只转换 theta（θ_full → K 偏差）。
    #    u/v/w 保持 m/s：sim.update_freestream_bc 在 C++ 内部除以 velocityRatio，
    #    这里若再转为格子单位会重复换算（速度被缩小 dx/dt 倍）。
    converted = apply_unit_conversions({"theta": interped.theta}, ref)

    # 5. 提取 hfx，双线性插值到 surfaceLayerBC 节点（若 sl_coords 提供）
    hfx_raw = extract_hfx(received)
    if hfx_raw is not None and sl_coords is not None:
        hfx = interp_hfx_to_sl_nodes(hfx_raw, sl_coords, static_ctx)
    else:
        if hfx_raw is not None:
            warnings.warn(
                "_make_wrf_frame: received 'hfx' but sl_coords is None; "
                "surface heat flux will NOT be updated for this frame.",
                stacklevel=2,
            )
        hfx = None

    return WRFFrame(
        t_wrf=t_wrf,
        u=np.asarray(interped.u, dtype=np.float64),
        v=np.asarray(interped.v, dtype=np.float64),
        w=np.asarray(interped.w, dtype=np.float64),
        theta=converted["theta"],
        hfx=hfx,
    )


# Lattice Mach number above which LBM compressibility errors become significant
_MACH_WARN = 0.3


def _log_bc_velocity_diagnostics(frame: WRFFrame, ref: LBMReference) -> None:
    """打印 BC 速度的物理值与格子单位换算，格子 Mach 数过大时告警。"""
    speed = np.sqrt(frame.u**2 + frame.v**2 + frame.w**2)
    u_max = float(speed.max()) if speed.size else 0.0
    u_lb_max = u_max / ref.u_ref
    mach = u_lb_max * math.sqrt(3.0)
    print(
        f"[lbm_init] BC velocity: max|U|={u_max:.3f} m/s  "
        f"u_ref(dx/dt)={ref.u_ref:.3f} m/s  max|U_lb|={u_lb_max:.4f}  Ma_lb={mach:.3f}",
        flush=True,
    )
    if mach > _MACH_WARN:
        warnings.warn(
            f"BC lattice Mach number {mach:.3f} exceeds {_MACH_WARN}; "
            "LBM compressibility error may be significant.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# 公开接口：init_lbm_from_wrf
# ---------------------------------------------------------------------------

def init_lbm_from_wrf(
    sim,
    received_step0: dict,
    received_step1: dict,
    static_ctx: WRFStaticContext,
    ref: LBMReference,
    level: int = 0,
    bc_coords_override: np.ndarray | None = None,
    sl_coords_override: np.ndarray | None = None,
) -> WRFDoubleBuffer:
    """用 WRF step=0 数据初始化 LBM BC，并建立双缓冲。

    调用约定
    --------
    此函数必须在以下操作之后调用：
      - sim 已构造（含 para, cuda_memory_manager, grid_builder, bc_factory）
      - sim.init_timers() 已调用
    此函数之后，调用方可直接进入主循环。

    Parameters
    ----------
    sim             : gpu.Simulation  已初始化的 VirtualFluids 仿真对象
    received_step0  : dict  WRF step=0 数据（含 '__wrf_time_s'）
    received_step1  : dict  WRF step=1 数据（含 '__wrf_time_s'）
    static_ctx      : WRFStaticContext（含 z0_mean）
    ref             : LBMReference
    level           : int  多网格层（通常为 0，最细层）
    bc_coords_override : (n_bc, 3) float64 或 None。
        若提供，直接用此坐标（x/y 已在 WRF 参考系，z 为 AGL），
        跳过内部的 sim.get_velocity_bc_coordinates 调用。
        用于调用方已做了水平 offset 转换（中心对齐等）的场景。
    sl_coords_override : (n_sl, 2) float64 或 None。
        若提供，直接用此坐标（x/y 已在 WRF 参考系），
        跳过内部的 sim.get_surface_layer_bc_coordinates 调用。

    Returns
    -------
    WRFDoubleBuffer  buf[0]=step0 帧，buf[1]=step1 帧，可直接用于主循环。

    Side effects
    ------------
    调用 sim.update_freestream_bc(u, v, w, level=level)（u/v/w 为 m/s），
    将 step=0 插值结果设置为首帧 BC。
    """
    # 1. 速度 BC 节点坐标（WRF 参考系，z=AGL）
    if bc_coords_override is not None:
        bc_coords = np.asarray(bc_coords_override, dtype=np.float64)
    else:
        bc_coords = np.asarray(sim.get_freestream_bc_coordinates(level=level), dtype=np.float64)

    # 2. surfaceLayerBC 节点坐标 x/y（WRF 参考系）
    if sl_coords_override is not None:
        sl_coords = np.asarray(sl_coords_override, dtype=np.float64)
        if sl_coords.ndim == 2 and sl_coords.shape[1] == 3:
            sl_coords = sl_coords[:, :2]  # 只取 x/y
    elif hasattr(sim, "get_surface_layer_bc_coordinates"):
        sl_coords_3d = np.asarray(sim.get_surface_layer_bc_coordinates(level=level), dtype=np.float64)
        sl_coords = sl_coords_3d[:, :2]
    else:
        sl_coords = None

    # 3. 构造两帧
    frame0 = _make_wrf_frame(received_step0, static_ctx, bc_coords, ref, sl_coords=sl_coords)
    frame1 = _make_wrf_frame(received_step1, static_ctx, bc_coords, ref, sl_coords=sl_coords)

    # 4. 用 step=0 设置首帧 freestream BC
    sim.update_freestream_bc(frame0.u, frame0.v, frame0.w, level=level)

    # 4b. 单位诊断：BC 速度为 m/s，VF 内部换算为格子单位后检查 Mach 数
    _log_bc_velocity_diagnostics(frame0, ref)

    # 5. 建立双缓冲
    return WRFDoubleBuffer(buf=[frame0, frame1])


# ---------------------------------------------------------------------------
# WRF 三维场初始条件
# ---------------------------------------------------------------------------

def make_wrf_initial_condition(
    received_step0: dict,
    static_ctx: WRFStaticContext,
    ref: LBMReference,
    lbm_origin_x: float = 0.0,
    lbm_origin_y: float = 0.0,
):
    """返回用 WRF step=0 三维场构造的 VF 初始条件函数。

    返回值是一个 Python 函数 ``ic(x, y, z) -> (rho, vx, vy, vz)``，
    可直接传给 ``para.set_initial_condition(ic)``。

    Parameters
    ----------
    received_step0  : dict  WRF step=0 数据（含 'u','v','w','theta','ph'）
    static_ctx      : WRFStaticContext（terrain_height, dx_wrf, dy_wrf）
    ref             : LBMReference（dx_lbm, dt_lbm）
    lbm_origin_x    : float  LBM 域西南角在 WRF 坐标系中的 x 偏移（m），默认 0
    lbm_origin_y    : float  LBM 域西南角在 WRF 坐标系中的 y 偏移（m），默认 0

    Returns
    -------
    ic : callable  ``ic(x, y, z) -> tuple[float, float, float, float]``
         x, y, z 是 LBM 域内的物理坐标（m），z 为 AGL。
         返回 (rho, vx, vy, vz)，速度已转为 LBM 格子单位。

    Notes
    -----
    - WRF 场在构造时预先计算，闭包里只做查表和列插值。
    - 水平：最近邻查找（与 interp_field_to_heights 一致）
    - 垂直：线性插值，超出范围夹紧（clamp）
    - z AGL → MSL：加当前格点地形高度
    - u, v, w 除以 velocity_ratio（= dx/dt）转为 LBM 无量纲速度
    - rho 固定返回 1.0（VF 不可压缩参考态）
    """
    # 预计算：只做一次
    wrf_data = make_wrf_column_data(received_step0, static_ctx)
    z_w    = compute_z_w(wrf_data.ph_total)       # (nz+1, ny, nx) MSL
    z_mass = compute_z_mass(z_w)                   # (nz,   ny, nx) MSL

    u_3d     = np.asarray(wrf_data.u,     dtype=np.float64)
    v_3d     = np.asarray(wrf_data.v,     dtype=np.float64)
    w_3d     = np.asarray(wrf_data.w,     dtype=np.float64)
    terrain  = np.asarray(wrf_data.terrain_height, dtype=np.float64)
    ny, nx   = terrain.shape
    dx_wrf   = wrf_data.dx_wrf
    dy_wrf   = wrf_data.dy_wrf
    velocity_ratio = ref.dx_lbm / ref.dt_lbm  # m/s → LBM 格子单位的分母

    def ic(x: float, y: float, z: float):
        """LBM 初始条件：(x, y, z_agl) → (rho, vx, vy, vz)。"""
        # LBM 坐标 → WRF 坐标
        x_wrf = x + lbm_origin_x
        y_wrf = y + lbm_origin_y

        # 水平最近邻格点
        j = int(np.clip(round(y_wrf / dy_wrf), 0, ny - 1))
        i = int(np.clip(round(x_wrf / dx_wrf), 0, nx - 1))

        # AGL → MSL
        z_msl = z + terrain[j, i]

        # 垂直列插值（质量层：u, v; w 层：w）
        u_val, _, _ = _interp_column_1d(u_3d[:, j, i], z_mass[:, j, i], z_msl)
        v_val, _, _ = _interp_column_1d(v_3d[:, j, i], z_mass[:, j, i], z_msl)
        w_val, _, _ = _interp_column_1d(w_3d[:, j, i], z_w[:, j, i],    z_msl)

        # 物理速度 → LBM 格子单位
        vx = u_val / velocity_ratio
        vy = v_val / velocity_ratio
        vz = w_val / velocity_ratio

        return (1.0, vx, vy, vz)

    return ic
