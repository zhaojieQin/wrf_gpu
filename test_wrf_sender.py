#!/usr/bin/env python3
"""
WRF 发送端最小测试

模拟 WRF 计算产生的 3D 子域数据，通过 MPI 发送到 LBM 接收端。

用于验证：
1. MPI 通信通道正确性
2. GPU → Host → MPI 数据流完整性
3. Pinned memory 分配/释放稳定性

约束：
- 必须运行在 rank 0
- GPU 由 CUDA_VISIBLE_DEVICES 指定（启动脚本负责）
- 数据形状：测试用假数据 (30,100,101)，真实 WRF 是 (44,42,42)
"""

import sys
from mpi4py import MPI
import jax
import jax.numpy as jnp
from wrf_mpi_sender import WRFMPISender


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 检查 rank
    if rank != 0:
        if rank == 1:
            # Rank 1 静默等待（LBM 接收端）
            pass
        else:
            print(f"[错误] test_wrf_sender.py 必须运行在 rank 0（当前 rank={rank}）")
            sys.exit(1)
        return

    print(f"[WRF Sender Test] 启动")
    print(f"  Rank: {rank}")
    print(f"  JAX devices: {jax.devices()}")
    print()

    # 创建发送端
    sender = WRFMPISender(dest_rank=1, threshold_mb=100.0)

    # 模拟 3 个耦合步
    n_steps = 3
    print(f"开始发送 {n_steps} 个耦合步...\n")

    for step in range(n_steps):
        print(f"{'='*70}")
        print(f"[Step {step}] 准备数据")
        print(f"{'='*70}")

        # 模拟 WRF 子域数据
        # 注意：这里用假数据形状 (30, 100, 101)
        # 真实 WRF 形状：(44, 42, 42) 对应 Switzerland case
        # 跑通后需要换成真实形状
        subdomain = {
            # 3D 风场（staggered grid）
            'u': jnp.ones((30, 100, 101), dtype=jnp.float32) * (step + 1.0),
            'v': jnp.ones((30, 101, 100), dtype=jnp.float32) * (step + 2.0),
            'w': jnp.ones((31, 100, 100), dtype=jnp.float32) * (step + 0.5),

            # 3D 热力学场
            'theta': jnp.ones((30, 100, 100), dtype=jnp.float32) * 300.0 + step,
            'qv': jnp.ones((30, 100, 100), dtype=jnp.float32) * 0.01,
            'p_total': jnp.ones((30, 100, 100), dtype=jnp.float32) * 1013.25,

            # 2D 地表场
            't_skin': jnp.ones((100, 100), dtype=jnp.float32) * 285.0 + step,
            'roughness_m': jnp.ones((100, 100), dtype=jnp.float32) * 0.1,
        }

        # 打印数据统计
        total_size_mb = sum(arr.nbytes for arr in subdomain.values()) / 1e6
        print(f"  字段数量: {len(subdomain)}")
        print(f"  总数据量: {total_size_mb:.2f} MB")
        for field, arr in subdomain.items():
            print(f"    {field}: {arr.shape}, {arr.nbytes/1e6:.2f} MB")

        # 发送
        print(f"\n  发送中...")
        n_fields = sender.send_subdomain(subdomain, step)
        print(f"  ✓ 发送 {n_fields} 个字段到 rank 1")

        # 等待发送完成
        print(f"  等待 MPI 完成...")
        sender.wait_all()
        print(f"  ✓ Step {step} 发送完成\n")

    print(f"{'='*70}")
    print(f"[WRF Sender Test] 完成")
    print(f"  总步数: {n_steps}")
    print(f"  ✓ 所有发送成功")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
