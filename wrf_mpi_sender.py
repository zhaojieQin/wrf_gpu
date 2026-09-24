#!/usr/bin/env python3
"""
WRF → LBM MPI 通信发送端

使用 MPICH + mpi4py + pinned host buffer 方案实现 GPU 数据传输。

设计约束：
- GH200 架构（GPU/CPU 共享内存），所有数据显式拷贝
- Pinned memory 显式释放（避免泄漏）
- GPU 由 CUDA_VISIBLE_DEVICES 指定（启动脚本负责）
- Tag 编码：step*1000 (元数据), step*1000+i+1 (字段 i)
  - 上限：999 个字段/step，4294967 个 steps (MPI tag < 2^31)
"""

from mpi4py import MPI
import cupy as cp
import numpy as np
from jax.dlpack import to_dlpack


class WRFMPISender:
    """WRF 侧 MPI 发送端

    负责将 WRF 计算的 3D 子域通过 MPI 发送到 LBM 进程（GPU Direct 或 pinned buffer）。

    Attributes:
        comm: MPI.COMM_WORLD communicator
        rank: 当前进程的 MPI rank
        dest_rank: 目标接收端 rank（默认 1，LBM 进程）
        threshold_mb: 大数据阈值（保留，当前未使用）
        pending_requests: 待完成的 MPI 请求列表
        has_cuda_aware_mpi: CUDA-aware MPI 支持（当前固定 False）
    """

    def __init__(self, dest_rank: int = 1, threshold_mb: float = 100.0):
        """初始化 MPI 发送端

        Args:
            dest_rank: 目标 rank（LBM 进程）
            threshold_mb: 大数据阈值（MB），保留用于未来优化
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.dest_rank = dest_rank
        self.threshold_mb = threshold_mb

        # 待完成的 MPI 请求：(request, host_buf, pinned_mem)
        self.pending_requests = []

        # CUDA-aware MPI 检测（当前固定使用 pinned buffer）
        self.has_cuda_aware_mpi = self._detect_cuda_aware_mpi()

        if self.rank == 0:
            print(f"[WRFMPISender] Initialized: rank={self.rank}, dest_rank={self.dest_rank}")
            print(f"[WRFMPISender] CUDA-aware MPI: {self.has_cuda_aware_mpi}")
            print(f"[WRFMPISender] Threshold: {self.threshold_mb:.1f} MB (reserved)")

    def send_subdomain(self, subdomain_jax: dict, coupling_step: int) -> int:
        """发送 3D 子域到 LBM

        数据流：
        1. JAX array → CuPy array（显式拷贝）
        2. 发送元数据（字段名、shape、dtype）
        3. 逐字段传输：
           - 分配 pinned host memory
           - GPU → Host（同步 get）
           - MPI Isend（非阻塞）

        Args:
            subdomain_jax: 字段名 -> JAX array 字典
            coupling_step: 耦合步数（用于 MPI tag 编码）

        Returns:
            发送的字段数量

        Raises:
            ValueError: 如果字段数量超过 999
        """
        if len(subdomain_jax) > 999:
            raise ValueError(f"字段数量 {len(subdomain_jax)} 超过上限 999")

        # 1. 准备元数据
        metadata = {}
        for field_name, jax_arr in subdomain_jax.items():
            metadata[field_name] = {
                'shape': jax_arr.shape,
                'dtype': str(jax_arr.dtype),
                'size_mb': jax_arr.nbytes / 1e6
            }

        # 2. 发送元数据（comm.send，小写，pickle）
        meta_tag = coupling_step * 1000
        self.comm.send(metadata, dest=self.dest_rank, tag=meta_tag)

        # 3. 逐字段发送数据
        for field_idx, (field_name, jax_arr) in enumerate(subdomain_jax.items()):
            # JAX → CuPy（显式拷贝，避免 GH200 共享内存下 DLPack 生命周期问题）
            cp_arr = cp.array(jax_arr)

            # 分配 pinned host memory
            pinned_mem = cp.cuda.alloc_pinned_memory(cp_arr.nbytes)
            host_buf = np.frombuffer(pinned_mem, dtype=cp_arr.dtype).reshape(cp_arr.shape)

            # GPU → Host（同步传输，返回时数据已在 host）
            cp_arr.get(out=host_buf)

            # MPI Isend（非阻塞发送）
            data_tag = coupling_step * 1000 + field_idx + 1
            request = self.comm.Isend(host_buf, dest=self.dest_rank, tag=data_tag)

            # 保存 pending request（需要保持 host_buf 和 pinned_mem 引用）
            self.pending_requests.append((request, host_buf, pinned_mem))

        return len(subdomain_jax)

    def wait_all(self) -> None:
        """等待所有 pending 发送完成并释放资源

        关键：显式释放 pinned memory，避免泄漏
        - 每段 ~240 MB，20 段/小时 × 24 小时 = 115 GB 泄漏

        注意：GH200 架构下 pinned memory 释放行为需要验证
        """
        for request, host_buf, pinned_mem in self.pending_requests:
            try:
                # 等待 MPI 发送完成
                request.Wait()
            finally:
                # 显式释放资源（即使 Wait 失败也要释放）
                # del 只删除 Python 引用，不保证立即 cudaFreeHost
                # CuPy 的 pinned memory 通过 MemoryPointer 管理
                # 需要验证 GH200 上是否真正释放
                del host_buf
                del pinned_mem

        self.pending_requests.clear()

    def _detect_cuda_aware_mpi(self) -> bool:
        """检测 MPICH 是否支持 CUDA-aware MPI

        当前实现：固定返回 False（使用 pinned buffer 方案）

        未来优化：
        1. 检查环境变量 MPIR_CVAR_ENABLE_GPU
        2. 尝试发送 GPU pointer 测试

        Returns:
            False（当前版本）
        """
        return False
