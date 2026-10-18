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
    interp_wrf_to_lbm_nodes_bilinear,
    build_lbm_grid_table,
)


# ---------------------------------------------------------------------------
# 双缓冲数据结构
# ---------------------------------------------------------------------------

class WRFFrame(NamedTuple):
    """单个 WRF 时间帧（在 BC 节点处插值完毕）。

    速度保持物理单位：sim.update_velocity_bc / update_freestream_bc 均接收 m/s，
    并在 C++ 内部（Simulation::updateFreestreamBC）自行除以 velocityRatio。

    Attributes
    ----------
    t_wrf     : float  WRF 物理时间（秒）
    u, v, w   : (n_bc,)  物理速度（m s⁻¹）
    theta     : (n_bc,)  LBM 浓度（K 偏差，= θ_full − theta_ref）
    hfx       : (n_sl,) or None  地表感热通量（W m⁻²），已插值到 surfaceLayerBC 节点
    theta_ad  : (n_ad,) or None  θ − theta_ref（K），已插值到温度 Dirichlet 边界节点
                （温度场开启时；节点集合与速度边界可能不同，单独插值）
    """
    t_wrf: float
    u: np.ndarray              # (n_bc,) m/s
    v: np.ndarray              # (n_bc,) m/s
    w: np.ndarray              # (n_bc,) m/s
    theta: np.ndarray          # (n_bc,) K deviation
    hfx: np.ndarray | None     # (n_sl,) W m⁻², or None if not sent
    theta_ad: np.ndarray | None = None  # (n_ad,) K deviation, or None


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

    def interp_theta_ad(self, t_lbm: float):
        """温度 Dirichlet 节点上的 θ − θ_ref 时间插值；两帧都没有时返回 None。"""
        f0, f1 = self.buf[0], self.buf[1]
        if f0.theta_ad is None or f1.theta_ad is None:
            return f1.theta_ad if f1.theta_ad is not None else f0.theta_ad
        dt = self.t1 - self.t0
        alpha = 1.0 if dt <= 0.0 else float(np.clip((t_lbm - self.t0) / dt, 0.0, 1.0))
        return (1.0 - alpha) * f0.theta_ad + alpha * f1.theta_ad

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
    ad_coords: np.ndarray | None = None,
) -> WRFFrame:
    """将一个 received dict 转换为 WRFFrame（BC 节点插值 + 单位转换）。

    ad_coords : (n_ad, 3) 温度 Dirichlet 边界节点坐标（WRF 参考系 x/y，z 为 AGL），或 None。
                给定时在这些节点上另外插值 θ，存入 WRFFrame.theta_ad（θ − theta_ref）。

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
    #    u/v/w 保持 m/s：update_velocity_bc / update_freestream_bc 在 C++ 内部除以 velocityRatio，
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

    # 6. 温度 Dirichlet 节点上的 θ（温度场开启时）
    theta_ad = None
    if ad_coords is not None:
        interped_ad = interp_to_bc_nodes(wrf_data, ad_coords, most_fields=most_fields)
        theta_ad = np.asarray(interped_ad.theta, dtype=np.float64) - ref.theta_ref

    return WRFFrame(
        t_wrf=t_wrf,
        u=np.asarray(interped.u, dtype=np.float64),
        v=np.asarray(interped.v, dtype=np.float64),
        w=np.asarray(interped.w, dtype=np.float64),
        theta=converted["theta"],
        hfx=hfx,
        theta_ad=theta_ad,
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
# BC 模式封装：velocity（固定速度）/ freestream（入流/出流切换）
# 两种模式的节点集合与顺序相同（均来自 builder->getVelocitySize），只是 VF 接口不同。
# ---------------------------------------------------------------------------

_BC_MODES = ("velocity", "freestream")


def get_bc_coordinates(sim, bc_mode: str = "freestream", level: int = 0) -> np.ndarray:
    """开放边界节点坐标 (n_bc, 3) [x, y, z]（LBM 坐标，m）。"""
    if bc_mode == "velocity":
        coords = sim.get_velocity_bc_coordinates(level=level)
    elif bc_mode == "freestream":
        coords = sim.get_freestream_bc_coordinates(level=level)
    else:
        raise ValueError(f"Unknown bc_mode {bc_mode!r} (expected one of {_BC_MODES})")
    return np.asarray(coords, dtype=np.float64)


def update_bc(sim, u, v, w, bc_mode: str = "freestream", level: int = 0) -> None:
    """把 u/v/w（m/s）写入开放边界；VF 在 C++ 内部除以 velocityRatio。"""
    if bc_mode == "velocity":
        sim.update_velocity_bc(u, v, w, level=level)
    elif bc_mode == "freestream":
        sim.update_freestream_bc(u, v, w, level=level)
    else:
        raise ValueError(f"Unknown bc_mode {bc_mode!r} (expected one of {_BC_MODES})")


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
    bc_mode: str = "freestream",
    ad_coords_override: np.ndarray | None = None,
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
    ad_coords_override : (n_ad, 3) float64 或 None。
        温度 Dirichlet 边界节点坐标（WRF 参考系 x/y，z 为 AGL）。给定时插值 θ 并
        调用 sim.update_ad_dirichlet_bc 设置首帧温度边界；None 表示无温度场。

    Returns
    -------
    WRFDoubleBuffer  buf[0]=step0 帧，buf[1]=step1 帧，可直接用于主循环。

    Side effects
    ------------
    按 bc_mode 调用 sim.update_velocity_bc / update_freestream_bc（u/v/w 为 m/s），
    将 step=0 插值结果设置为首帧 BC。
    """
    # 1. 速度 BC 节点坐标（WRF 参考系，z=AGL）
    if bc_coords_override is not None:
        bc_coords = np.asarray(bc_coords_override, dtype=np.float64)
    else:
        bc_coords = get_bc_coordinates(sim, bc_mode, level=level)

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
    ad_coords = None if ad_coords_override is None else np.asarray(ad_coords_override, dtype=np.float64)
    frame0 = _make_wrf_frame(received_step0, static_ctx, bc_coords, ref,
                             sl_coords=sl_coords, ad_coords=ad_coords)
    frame1 = _make_wrf_frame(received_step1, static_ctx, bc_coords, ref,
                             sl_coords=sl_coords, ad_coords=ad_coords)

    # 4. 用 step=0 设置首帧开放边界（velocity 或 freestream）
    update_bc(sim, frame0.u, frame0.v, frame0.w, bc_mode, level=level)

    # 4b. 单位诊断：BC 速度为 m/s，VF 内部换算为格子单位后检查 Mach 数
    _log_bc_velocity_diagnostics(frame0, ref)

    # 4c. 首帧温度 Dirichlet 边界（θ − θ_ref，K）
    if frame0.theta_ad is not None:
        sim.update_ad_dirichlet_bc(frame0.theta_ad, level=level)
        print(f"[lbm_init] AD Dirichlet nodes: {frame0.theta_ad.size}, θ−θ_ref range "
              f"[{frame0.theta_ad.min():.2f}, {frame0.theta_ad.max():.2f}] K", flush=True)

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
    domain_length_x: float | None = None,
    domain_length_y: float | None = None,
    domain_length_z: float | None = None,
):
    """返回用 WRF step=0 三维场构造的 VF 初始条件函数。

    返回值是一个 Python 函数 ``ic(x, y, z) -> (rho, vx, vy, vz)``，
    可直接传给 ``para.set_initial_condition(ic)``。

    Parameters
    ----------
    received_step0  : dict  WRF step=0 数据（含 'u','v','w','theta','ph'，可选 MOST 场）
    static_ctx      : WRFStaticContext（terrain_height, dx_wrf, dy_wrf）
    ref             : LBMReference（dx_lbm, dt_lbm）
    lbm_origin_x    : float  LBM 域西南角在 WRF 坐标系中的 x 偏移（m），默认 0
    lbm_origin_y    : float  LBM 域西南角在 WRF 坐标系中的 y 偏移（m），默认 0
    domain_length_x/y/z : float | None  LBM 域尺寸（m）。三者都给定时，预先在 LBM
                      网格上算好查找表（快）；否则每次调用单点插值（慢，仅用于测试）。

    Returns
    -------
    ic : callable  ``ic(x, y, z) -> tuple[float, float, float, float]``
         x, y, z 是 LBM 域内的物理坐标（m），z 为 AGL。
         返回 (rho, vx, vy, vz)，速度已转为 LBM 格子单位。

    Notes
    -----
    - 与边界条件使用同一套插值（interp_wrf_to_lbm_nodes_bilinear）：
      u/v 去交错，水平双线性，每个 WRF 格柱按离地高度线性插值，
      最低质量层以下用 MOST（若 received_step0 带 ustar/roughness_m/psim）。
    - 查找表节点与 VF 网格一致：x_i = -dx/2 + i*dx，共 round(L/dx)+2 个点；
      ic 内按最近的表节点取值。
    - u, v, w 除以 velocity_ratio（= dx/dt）转为 LBM 无量纲速度
    - rho 返回 0.0：VF 初始条件的 rho 是密度偏差 Δρ（GridProvider 默认 0、
      initialization_functions.log_law 也返回 0）。返回 1.0 会使总密度变为 2，
      而速度边界的动量项 6·w·u 按 ρ=1 推导，边界处实际速度会被减半。
    """
    wrf_data = make_wrf_column_data(received_step0, static_ctx)
    most_fields = extract_most_fields(received_step0)
    velocity_ratio = ref.dx_lbm / ref.dt_lbm  # m/s → LBM 格子单位的分母

    if None in (domain_length_x, domain_length_y, domain_length_z):
        # 单点回退路径：与边界条件同一插值函数，逐点调用
        def ic(x: float, y: float, z: float):
            f = interp_wrf_to_lbm_nodes_bilinear(
                wrf_data,
                target_x=np.array([x + lbm_origin_x]),
                target_y=np.array([y + lbm_origin_y]),
                target_z_agl=np.array([z]),
                most_fields=most_fields,
            )
            return (0.0, float(f.u[0]) / velocity_ratio,
                    float(f.v[0]) / velocity_ratio, float(f.w[0]) / velocity_ratio)

        return ic

    # 查找表路径：LBM 网格是张量积网格，先算整张表，ic 只做取值
    dx = ref.dx_lbm
    axes = []
    for length in (domain_length_x, domain_length_y, domain_length_z):
        n = int(round(length / dx)) + 2
        axes.append(-0.5 * dx + dx * np.arange(n))
    x_l, y_l, z_l = axes
    x_wrf = np.clip(x_l + lbm_origin_x, 0.0, (wrf_data.terrain_height.shape[1] - 1) * wrf_data.dx_wrf)
    y_wrf = np.clip(y_l + lbm_origin_y, 0.0, (wrf_data.terrain_height.shape[0] - 1) * wrf_data.dy_wrf)
    u_tab, v_tab, w_tab = build_lbm_grid_table(
        wrf_data, x_wrf, y_wrf, np.maximum(z_l, 0.0), most_fields=most_fields,
    )
    inv = 1.0 / velocity_ratio
    u_tab *= inv
    v_tab *= inv
    w_tab *= inv
    nx_l, ny_l, nz_l = x_l.size, y_l.size, z_l.size
    x0, y0, z0 = x_l[0], y_l[0], z_l[0]
    inv_dx = 1.0 / dx
    print(f"[lbm_init] IC table built: {ny_l}x{nx_l}x{nz_l}, "
          f"max|u|={float(np.abs(u_tab).max()) * velocity_ratio:.2f} m/s", flush=True)

    def ic(x: float, y: float, z: float):
        """LBM 初始条件：(x, y, z_agl) → (rho, vx, vy, vz)，查表。"""
        ix = min(max(int(round((x - x0) * inv_dx)), 0), nx_l - 1)
        iy = min(max(int(round((y - y0) * inv_dx)), 0), ny_l - 1)
        iz = min(max(int(round((z - z0) * inv_dx)), 0), nz_l - 1)
        return (0.0, float(u_tab[iy, ix, iz]), float(v_tab[iy, ix, iz]), float(w_tab[iy, ix, iz]))

    return ic


# ---------------------------------------------------------------------------
# WRF 三维温度场初始条件（温度场开启时）
# ---------------------------------------------------------------------------

def estimate_theta_ref(
    received_step0: dict,
    static_ctx: WRFStaticContext,
    dx_lbm: float,
    lbm_origin_x: float,
    lbm_origin_y: float,
    domain_length_x: float,
    domain_length_y: float,
    domain_length_z: float,
    stride_xy: int = 8,
    stride_z: int = 4,
) -> float:
    """LBM 体积内 WRF θ 的平均值（K），用作自动 θ_ref。

    插值方式与 make_wrf_initial_temperature 相同（水平双线性 + 每列按离地高度插值），
    只是在 LBM 网格上按 stride 抽稀，代价很小。
    """
    wrf_data = make_wrf_column_data(received_step0, static_ctx)
    most_fields = extract_most_fields(received_step0)

    axes = []
    for length, stride in ((domain_length_x, stride_xy), (domain_length_y, stride_xy),
                           (domain_length_z, stride_z)):
        n = int(round(length / dx_lbm)) + 2
        axes.append((-0.5 * dx_lbm + dx_lbm * np.arange(n))[::stride])
    x_l, y_l, z_l = axes
    ny_w, nx_w = wrf_data.terrain_height.shape
    x_wrf = np.clip(x_l + lbm_origin_x, 0.0, (nx_w - 1) * wrf_data.dx_wrf)
    y_wrf = np.clip(y_l + lbm_origin_y, 0.0, (ny_w - 1) * wrf_data.dy_wrf)
    (theta_tab,) = build_lbm_grid_table(
        wrf_data, x_wrf, y_wrf, np.maximum(z_l, 0.0),
        most_fields=most_fields, fields=("theta",),
    )
    theta_ref = float(np.mean(theta_tab, dtype=np.float64))
    print(f"[lbm_init] auto θ_ref = {theta_ref:.3f} K  (θ range in LBM volume "
          f"[{float(theta_tab.min()):.2f}, {float(theta_tab.max()):.2f}] K, "
          f"{theta_tab.size} sample nodes)", flush=True)
    return theta_ref


def make_wrf_initial_temperature(
    received_step0: dict,
    static_ctx: WRFStaticContext,
    ref: LBMReference,
    lbm_origin_x: float,
    lbm_origin_y: float,
    domain_length_x: float,
    domain_length_y: float,
    domain_length_z: float,
):
    """返回 (ic_ad, ic_ref)，分别传给 para.set_initial_condition_AD /
    para.set_initial_local_reference_temperature。

    - ic_ad(x, y, z)  -> θ − θ_ref（K），WRF step=0 的 θ，与速度初始场同一套插值
      （水平双线性、每个格柱按离地高度线性插值），在 LBM 网格上预先建表。
    - ic_ref(x, y, z) -> 同一张表的水平平均廓线（只随 z 变化），作为初始局部参考温度；
      BuoyancyProviderPlanarAverage 之后每一步会用当前的水平平均覆盖它。
    """
    wrf_data = make_wrf_column_data(received_step0, static_ctx)
    most_fields = extract_most_fields(received_step0)

    dx = ref.dx_lbm
    axes = []
    for length in (domain_length_x, domain_length_y, domain_length_z):
        n = int(round(length / dx)) + 2
        axes.append(-0.5 * dx + dx * np.arange(n))
    x_l, y_l, z_l = axes
    ny_w, nx_w = wrf_data.terrain_height.shape
    x_wrf = np.clip(x_l + lbm_origin_x, 0.0, (nx_w - 1) * wrf_data.dx_wrf)
    y_wrf = np.clip(y_l + lbm_origin_y, 0.0, (ny_w - 1) * wrf_data.dy_wrf)
    (theta_tab,) = build_lbm_grid_table(
        wrf_data, x_wrf, y_wrf, np.maximum(z_l, 0.0),
        most_fields=most_fields, fields=("theta",),
    )
    theta_tab -= np.float32(ref.theta_ref)              # θ − θ_ref
    theta_mean = theta_tab.mean(axis=(0, 1))            # (nz_l,) 水平平均廓线

    nx_l, ny_l, nz_l = x_l.size, y_l.size, z_l.size
    x0, y0, z0 = x_l[0], y_l[0], z_l[0]
    inv_dx = 1.0 / dx
    print(f"[lbm_init] θ IC table built: {ny_l}x{nx_l}x{nz_l}, θ−θ_ref range "
          f"[{float(theta_tab.min()):.2f}, {float(theta_tab.max()):.2f}] K "
          f"(θ_ref={ref.theta_ref} K)", flush=True)

    def _idx(v, v0, n):
        return min(max(int(round((v - v0) * inv_dx)), 0), n - 1)

    def ic_ad(x: float, y: float, z: float) -> float:
        return float(theta_tab[_idx(y, y0, ny_l), _idx(x, x0, nx_l), _idx(z, z0, nz_l)])

    def ic_ref(x: float, y: float, z: float) -> float:  # noqa: ARG001
        return float(theta_mean[_idx(z, z0, nz_l)])

    return ic_ad, ic_ref


# ---------------------------------------------------------------------------
# 诊断：温度 Dirichlet 边界值（只打印，不改变计算）
# ---------------------------------------------------------------------------

def diagnose_theta_bc(ad_coords: np.ndarray, theta_ad: np.ndarray, label: str = "",
                      ic_ad=None, origin_xy: tuple[float, float] = (0.0, 0.0),
                      hfx: np.ndarray | None = None) -> None:
    """按面（MX/PX/MY/PY/PZ）打印边界 θ−θ_ref 统计；可选与初始场 ic_ad 逐点比较。

    ad_coords : (n_ad, 3) 节点坐标（x/y 为 WRF 参考系，z 为 AGL）
    ic_ad     : 初始场函数 ic_ad(x, y, z)，参数为 LBM 坐标（x/y 减去 origin_xy）
    hfx       : 地面热通量（W/m²，地面节点上），可选
    """
    c = np.asarray(ad_coords, dtype=np.float64)
    th = np.asarray(theta_ad, dtype=np.float64)
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    dz = np.diff(np.unique(np.round(z, 6)))
    tol = 0.5 * (dz[dz > 0].min() if dz.size and (dz > 0).any() else 1.0)
    faces = {
        "PZ": np.abs(z - z.max()) < tol,
        "MX": np.abs(x - x.min()) < tol,
        "PX": np.abs(x - x.max()) < tol,
        "MY": np.abs(y - y.min()) < tol,
        "PY": np.abs(y - y.max()) < tol,
    }
    ic_vals = None
    if ic_ad is not None:
        ox, oy = origin_xy
        ic_vals = np.array([ic_ad(xi - ox, yi - oy, zi) for xi, yi, zi in c])
    lines = [f"[diag θ_bc] {label} n={th.size}  θ−θ_ref all: [{th.min():.2f}, {th.max():.2f}] K"]
    for name, m in faces.items():
        if not m.any():
            continue
        low = m & (z < 100.0)
        top = m & (z > z.max() - 100.0)
        s = (f"  {name}: n={int(m.sum()):6d}  mean={th[m].mean():7.2f}"
             f"  z<100m={th[low].mean() if low.any() else float('nan'):7.2f}"
             f"  top100m={th[top].mean() if top.any() else float('nan'):7.2f}")
        if ic_vals is not None:
            d = th[m] - ic_vals[m]
            s += f"  | BC−IC: mean={d.mean():+.3f} max|d|={np.abs(d).max():.3f}"
            if low.any():
                s += f" (z<100m mean {(th[low] - ic_vals[low]).mean():+.3f})"
        lines.append(s)
    if hfx is not None and np.size(hfx):
        h = np.asarray(hfx, dtype=np.float64)
        lines.append(f"  HFX (W/m², ground nodes): mean={h.mean():.2f} min={h.min():.2f} max={h.max():.2f}")
    print("\n".join(lines), flush=True)
