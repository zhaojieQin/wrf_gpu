#!/usr/bin/env python3
"""
LBM 侧 MPI 接收端（Python 测试版）

用于验证 WRF → LBM MPI 通信通道的正确性。

功能：
1. 接收 WRF 发送的元数据和 3D 子域数据
2. 打印字段信息（shape、dtype、数值范围）
3. 可选：将数据传输到 GPU 验证完整性

约束：
- 必须运行在 rank 1
- GPU 由 CUDA_VISIBLE_DEVICES 指定（启动脚本负责）
- 循环次数可配置（默认 3 步验证通道，可选 20 步验证稳定性）
"""

import sys
import argparse
from mpi4py import MPI
import cupy as cp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="LBM MPI 接收端测试")
    parser.add_argument('--steps', type=int, default=3,
                       help='接收的耦合步数（默认 3，验证通道；20 验证稳定性）')
    parser.add_argument('--verify-gpu', action='store_true',
                       help='将数据传输到 GPU 验证完整性')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 检查 rank
    if rank != 1:
        if rank == 0:
            # Rank 0 静默等待（WRF 发送端）
            pass
        else:
            print(f"[错误] mock_receiver.py 必须运行在 rank 1（当前 rank={rank}）")
            sys.exit(1)
        return

    print(f"[LBM Receiver] 启动")
    print(f"  Rank: {rank}")
    print(f"  接收步数: {args.steps}")
    print(f"  GPU 验证: {args.verify_gpu}")
    print(f"  CUDA_VISIBLE_DEVICES: {cp.cuda.runtime.getDevice()}")
    print()

    # 循环接收 N 个 coupling steps
    for step in range(args.steps):
        print(f"{'='*70}")
        print(f"[Step {step}] 开始接收")
        print(f"{'='*70}")

        # 1. 接收元数据（comm.recv，小写，pickle）
        meta_tag = step * 1000
        metadata = comm.recv(source=0, tag=meta_tag)
        print(f"  元数据接收: {len(metadata)} 个字段")

        # 2. 逐字段接收数据
        for field_idx, (field_name, field_meta) in enumerate(metadata.items()):
            shape = field_meta['shape']
            dtype_str = field_meta['dtype']
            size_mb = field_meta['size_mb']

            # 转换 dtype 字符串
            dtype = np.dtype(dtype_str)

            # 分配 host buffer（使用标准 NumPy array）
            host_buf = np.empty(shape, dtype=dtype)

            # MPI Recv（阻塞接收，comm.Recv 大写）
            data_tag = step * 1000 + field_idx + 1
            comm.Recv(host_buf, source=0, tag=data_tag)

            # 打印验证信息
            val_min = host_buf.min()
            val_max = host_buf.max()
            print(f"    {field_name}:")
            print(f"      shape={shape}, dtype={dtype}, size={size_mb:.2f} MB")
            print(f"      range=[{val_min:.6f}, {val_max:.6f}]")

            # 可选：传输到 GPU 验证
            if args.verify_gpu:
                gpu_data = cp.asarray(host_buf)
                gpu_min = float(cp.min(gpu_data))
                gpu_max = float(cp.max(gpu_data))
                print(f"      GPU 验证: [{gpu_min:.6f}, {gpu_max:.6f}]")

                # 验证 CPU 和 GPU 数据一致
                if abs(gpu_min - val_min) > 1e-6 or abs(gpu_max - val_max) > 1e-6:
                    print(f"      [警告] CPU/GPU 数据不一致！")

            # 释放 pinned memory
            del host_buf

        print(f"  ✓ Step {step} 接收完成\n")

    print(f"{'='*70}")
    print(f"[LBM Receiver] 完成")
    print(f"  总步数: {args.steps}")
    print(f"  ✓ 通道验证通过")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
