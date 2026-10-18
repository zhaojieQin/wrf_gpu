"""任务9：双缓冲主循环 run_lbm_coupled()。

公开接口
--------
run_lbm_coupled(sim, db, recv_loop, ref, n_steps, level=0, dt_lbm=None)

设计说明
--------
主循环逻辑：
  1. 每步计算 t_lbm = lbm_step * dt_lbm（秒）
  2. 调用 recv_loop.maybe_advance(t_lbm, db) — 若需要换页则阻塞等待
  3. 从 db.interp(t_lbm) 拿到时间插值后的 u/v/w/theta（格子单位）
  4. sim.update_velocity_bc(u, v, w, level=level) — 更新 BC
  5. sim.calculate_timestep(lbm_step) — 推进一步

dt_lbm 约定
-----------
若传入 dt_lbm，用 lbm_step * dt_lbm 计算物理时间。
若不传入，调用 para.get_timestep() 获取（VirtualFluids 约定）。
本模块不依赖 para，dt_lbm 由调用方传入（从 LBMReference 读取更合适）。

theta 输出
----------
主循环只更新速度 BC（u/v/w）。theta 作为 LBM concentration（K 偏差）
通过单独的 sim.update_concentration_bc() 或类似接口传入（如果 VF 支持）。
当前版本将 theta 包含在返回值里供调用方使用，但不调用 VF API（待 Task 10 集成）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lbm_adapter.lbm_init import WRFDoubleBuffer, update_bc
from lbm_adapter.recv_loop import WRFRecvLoop


# ---------------------------------------------------------------------------
# 运行统计
# ---------------------------------------------------------------------------

@dataclass
class CouplingStats:
    """主循环运行统计。

    Attributes
    ----------
    n_steps_run    : int    实际执行的 LBM 步数
    n_advances     : int    双缓冲换页次数
    t_wrf_start    : float  循环开始时 buf[0].t_wrf
    t_wrf_end      : float  循环结束时 buf[1].t_wrf（最后接收到的帧时间）
    t_lbm_end      : float  最后一步的 LBM 物理时间（秒）
    """
    n_steps_run: int = 0
    n_advances: int = 0
    t_wrf_start: float = 0.0
    t_wrf_end: float = 0.0
    t_lbm_end: float = 0.0


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def run_lbm_coupled(
    sim,
    db: WRFDoubleBuffer,
    recv_loop: WRFRecvLoop,
    n_steps: int,
    dt_lbm: float,
    level: int = 0,
    start_step: int = 1,
    para=None,
    output_until_step: int | None = None,
    bc_mode: str = "freestream",
    hfx_scale: float | None = None,
) -> CouplingStats:
    """双缓冲耦合主循环。

    Parameters
    ----------
    sim        : gpu.Simulation  已初始化的 VirtualFluids 仿真对象。
                 必须已调用 sim.init_timers()。
    db         : WRFDoubleBuffer  由 init_lbm_from_wrf() 返回的双缓冲。
    recv_loop  : WRFRecvLoop  由 WRFRecvLoop(..., next_step=2) 创建。
    n_steps    : int  总 LBM 步数（从 start_step 到 start_step + n_steps - 1）。
    dt_lbm     : float  LBM 时间步长（秒）。
    level      : int  多网格层（default 0）。
    start_step : int  起始步编号（default 1，与 VirtualFluids 约定一致，0 用于初始状态）。
    hfx_scale  : float | None  地表热通量换算系数 dt/(dx·ρ·cp)，把 W/m² 换成 VF 的格子单位
                 运动学通量；None 时按原值传入（旧行为）。
    para       : gpu.Parameter | None  与 sim 共享的 VF Parameter，仅 output_until_step 需要。
    output_until_step : int | None  此步算完后调用 para.set_timestep_out(0) 关闭 VTK 输出
                 （VF 每步重新读取 timestep_out）。None 表示不干预输出。

    Returns
    -------
    CouplingStats  包含运行统计。

    Notes
    -----
    - t_lbm = lbm_step * dt_lbm（物理时间，单位秒）。
    - 每步先换页（若需要），再插值，再更新 BC，再推进。
    - BC 更新只包含速度（u/v/w）。theta 返回在 stats 之外不使用（待 Task 10）。
    - 若 recv_loop.eof_received == True 且 db.needs_advance，不再等待新帧，
      使用 db.t1 处的值（夹紧）直到仿真结束。
    """
    stats = CouplingStats(
        t_wrf_start=db.t0,
        t_wrf_end=db.t1,
    )

    for lbm_step in range(start_step, start_step + n_steps):
        t_lbm = lbm_step * dt_lbm

        # 1. 换页（若需要）
        advanced = recv_loop.maybe_advance(t_lbm, db)
        if advanced:
            stats.n_advances += 1
            stats.t_wrf_end = db.t1

        # 2. 时间插值
        u, v, w, theta, hfx = db.interp(t_lbm)

        # 3. 更新地表热通量 BC（仅换页时 hfx 变化，但每步更新代价可忽略）
        #    VF 存的是格子单位的运动学热通量：hfx_scale = dt / (dx·ρ·cp)，把 W/m² 换算过去。
        if hfx is not None and hasattr(sim, "update_surface_layer_heat_flux"):
            sim.update_surface_layer_heat_flux(
                hfx * hfx_scale if hfx_scale is not None else hfx, level=level)

        # 3b. 更新温度 Dirichlet 边界（θ − θ_ref，K；温度场开启时）
        theta_ad = db.interp_theta_ad(t_lbm)
        if theta_ad is not None:
            sim.update_ad_dirichlet_bc(theta_ad, level=level)

        # 4. 更新开放边界速度（velocity: 固定速度；freestream: GPU kernel 逐节点判断入流/出流）
        update_bc(sim, u, v, w, bc_mode, level=level)

        # 4. 推进一步
        sim.calculate_timestep(lbm_step)

        # 5. 达到输出上限后关闭 VTK 输出
        if output_until_step is not None and lbm_step == output_until_step:
            para.set_timestep_out(0)

        stats.n_steps_run += 1
        stats.t_lbm_end = t_lbm

    return stats
