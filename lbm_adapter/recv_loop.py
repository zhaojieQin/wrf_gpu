"""任务8：双缓冲 WRF 接收循环（buf[0]/buf[1] 换页 + 阻塞等待）。

公开接口
--------
recv_next_wrf_step(comm, step, source=0) -> dict
    阻塞接收单个 WRF 时间步的完整 dict。
    遵循 MPI 协议：meta_tag=step*1000，data_tag=step*1000+field_idx+1。

WRFRecvLoop
    有状态接收管理器。封装 step 计数、buf 检查、换页逻辑。
    主循环调用 maybe_advance(t_lbm_s, db) 即可，内部自动换页。

设计说明
--------
MPI 协议（与 coupling_adapter.py / mock_receiver.py 一致）：
  - meta_tag   = step * 1000          (comm.recv, pickle dict)
  - data_tag_k = step * 1000 + k + 1  (comm.Recv, numpy array, k=field_idx)
  - EOF_TAG    = 999998
  - READY_TAG  = 999999

字段元数据格式（每个字段在 metadata dict 中的条目）：
    metadata[field_name] = {'shape': (nz, ny, nx), 'dtype': 'float32', 'size_mb': ...}
控制键以 '__' 开头，不对应 MPI 数组包（如 '__wrf_time_s'）。

任务8 专注于：
  1. recv_next_wrf_step : 接收一帧原始数据 dict
  2. WRFRecvLoop.maybe_advance : 若 t_lbm >= buf[1].t_wrf，阻塞换页
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from lbm_adapter.lbm_init import WRFDoubleBuffer, WRFFrame, _make_wrf_frame
from lbm_adapter.wrf_to_lbm import WRFStaticContext
from lbm_adapter.unit_convert import LBMReference

# MPI 协议常量（与 coupling_adapter.py 保持同步）
EOF_TAG   = 999998
READY_TAG = 999999


# ---------------------------------------------------------------------------
# 低层：接收单帧数据
# ---------------------------------------------------------------------------

def recv_next_wrf_step(comm, step: int, source: int = 0,
                       barrier_after_meta: bool = True) -> dict:
    """阻塞接收单个 WRF 时间步的完整 dict。

    协议
    ----
    1. comm.recv(source, tag=step*1000)         → metadata pickle dict
       comm.Barrier()                           → 与 WRFMPISender.send_subdomain 中
                                                  元数据之后的 Barrier 配对
    2. comm.Recv(buf, source, tag=step*1000+k+1) → 第 k 个字段数组（k=0,1,…）
       跳过以 '__' 开头的控制键（它们在 metadata 中携带值，无对应数组）

    Parameters
    ----------
    comm   : MPI communicator（mpi4py MPI.COMM_WORLD 或兼容对象）
    step   : WRF 耦合步编号（0-based）
    source : WRF rank（默认 0）
    barrier_after_meta : 收到元数据后是否调用 comm.Barrier()（默认 True，
             必须与发送端一致；仅在不带 Barrier 的 mock 发送端下关闭）

    Returns
    -------
    received : dict
        控制键（'__wrf_time_s' 等）直接来自 metadata。
        数组键（'u', 'v', 'w', 'theta', 'ph', 'ustar', …）是 numpy arrays。
    """
    meta_tag = step * 1000
    metadata: dict = comm.recv(source=source, tag=meta_tag)
    if barrier_after_meta:
        comm.Barrier()

    received: dict = {}

    # 提取控制键（直接值，不对应数组包）
    for key, val in metadata.items():
        if key.startswith("__"):
            received[key] = val

    # 逐字段接收数组
    field_idx = 0
    for field_name, field_meta in metadata.items():
        if field_name.startswith("__"):
            continue  # 控制键已处理

        shape    = field_meta["shape"]
        dtype    = np.dtype(field_meta["dtype"])
        data_tag = step * 1000 + field_idx + 1

        host_buf = np.empty(shape, dtype=dtype)
        comm.Recv(host_buf, source=source, tag=data_tag)
        received[field_name] = host_buf
        field_idx += 1

    return received


# ---------------------------------------------------------------------------
# 高层：有状态接收管理器
# ---------------------------------------------------------------------------

@dataclass
class WRFRecvLoop:
    """有状态双缓冲换页管理器。

    负责：
    - 追踪下一个待接收的 WRF 步编号
    - 在 t_lbm >= buf[1].t_wrf 时阻塞换页（接收新帧）
    - 将接收到的原始 dict 转换为 WRFFrame（插值 + 单位转换）

    Usage
    -----
    创建后直接在主循环中调用 maybe_advance：

        recv_loop = WRFRecvLoop(comm, static_ctx, ref, bc_coords,
                                next_step=2)  # step=0,1 已用于初始化
        while sim running:
            recv_loop.maybe_advance(t_lbm, db)
            u, v, w, theta = db.interp(t_lbm)
            sim.update_velocity_bc(u, v, w, level=0)
            sim.calculate_timestep(lbm_step)

    Parameters
    ----------
    comm       : MPI communicator
    static_ctx : WRFStaticContext
    ref        : LBMReference
    bc_coords  : (n_bc, 3)  BC 节点坐标（不变）
    sl_coords  : (n_sl, 2)  surfaceLayerBC 节点 x/y 坐标（WRF 参考系），或 None。
                 提供时每帧 hfx 都会插值到这些节点；None 时 hfx 不更新。
    next_step  : int  下一个待接收的 WRF step（初始化消耗了 0 和 1，所以从 2 开始）
    source     : int  WRF rank（默认 0）
    barrier_after_meta : bool  透传给 recv_next_wrf_step（默认 True）
    ad_coords  : (n_ad, 3)  温度 Dirichlet 边界节点坐标（WRF 参考系 x/y，z 为 AGL），或 None。
    eof_received : bool  是否已收到 EOF（内部状态，不要手动设置）
    """
    comm: object
    static_ctx: WRFStaticContext
    ref: LBMReference
    bc_coords: np.ndarray
    next_step: int = 2
    source: int = 0
    sl_coords: np.ndarray | None = None
    barrier_after_meta: bool = True
    ad_coords: np.ndarray | None = None
    eof_received: bool = field(default=False, init=False)

    def maybe_advance(self, t_lbm_s: float, db: WRFDoubleBuffer) -> bool:
        """若 t_lbm_s >= db.t1，阻塞接收下一帧并换页。

        Parameters
        ----------
        t_lbm_s : float  LBM 当前物理时间（秒）
        db       : WRFDoubleBuffer  当前双缓冲（原地修改）

        Returns
        -------
        advanced : bool  True 表示发生了换页（收到新帧），False 表示无需换页。

        Notes
        -----
        若已收到 EOF，不再接收新帧，直接返回 False。
        若新帧时间戳 <= db.t1（乱序/重复），发出警告并跳过。
        """
        if not db.needs_advance(t_lbm_s):
            return False
        if self.eof_received:
            return False

        # 阻塞接收下一帧
        try:
            raw = recv_next_wrf_step(self.comm, self.next_step, self.source,
                                     barrier_after_meta=self.barrier_after_meta)
        except Exception as exc:
            # 检查是否是 EOF 伪装成 dict 的情况（不应发生，但防御性处理）
            raise RuntimeError(
                f"WRFRecvLoop: failed to receive step {self.next_step}: {exc}"
            ) from exc

        new_frame = _make_wrf_frame(raw, self.static_ctx, self.bc_coords, self.ref,
                                    sl_coords=self.sl_coords, ad_coords=self.ad_coords)

        if new_frame.t_wrf <= db.t1:
            warnings.warn(
                f"WRFRecvLoop: received frame t_wrf={new_frame.t_wrf:.3f}s "
                f"<= db.t1={db.t1:.3f}s (out-of-order or duplicate). "
                "Skipping.",
                stacklevel=2,
            )
            self.next_step += 1
            return False

        db.advance(new_frame)
        self.next_step += 1
        return True
