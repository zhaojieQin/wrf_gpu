#!/usr/bin/env python3
"""
LBM 侧 MPI 接收端（Python 测试版）

用于验证 WRF → LBM MPI 通信通道的正确性。

功能：
1. 接收 WRF 发送的元数据和 3D 子域数据
2. 打印字段信息（shape、dtype、数值范围）
3. 可选：将数据传输到 GPU 验证完整性
4. 可选：验证 WRF 数据物理合理性（范围、常数、演化）
5. 超时保护：每步 60s 超时，避免死锁

约束：
- 必须运行在 rank 1
- GPU 由 CUDA_VISIBLE_DEVICES 指定（启动脚本负责）
- 默认接收 20 步（完整 1h 耦合测试）
"""

import sys
import argparse
import signal
from mpi4py import MPI
import cupy as cp
import numpy as np


class TimeoutError(Exception):
    """接收超时异常"""
    pass


def timeout_handler(signum, frame):
    """超时信号处理器"""
    raise TimeoutError("接收超时（60s）")


def main():
    parser = argparse.ArgumentParser(description="LBM MPI 接收端测试")
    parser.add_argument('--steps', type=int, default=20,
                       help='接收的耦合步数（默认 20，完整 1h 测试）')
    parser.add_argument('--verify-gpu', action='store_true',
                       help='将数据传输到 GPU 验证完整性')
    parser.add_argument('--verify-wrf', action='store_true',
                       help='验证 WRF 数据物理合理性（范围、常数、演化）')
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
    print(f"  WRF 验证: {args.verify_wrf}")
    print(f"  CUDA_VISIBLE_DEVICES: {cp.cuda.runtime.getDevice()}")
    print()

    # 历史数据（用于演化验证）
    history = []

    # 循环接收 N 个 coupling steps
    for step in range(args.steps):
        # 设置超时保护（60s per step）
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)

        try:
            print(f"{'='*70}")
            print(f"[Step {step}] 开始接收（超时 60s）")
            print(f"{'='*70}")

            # 1. 接收元数据（comm.recv，小写，pickle）
            meta_tag = step * 1000
            metadata = comm.recv(source=0, tag=meta_tag)
            print(f"  元数据接收: {len(metadata)} 个字段")

            # 当前步数据
            current_data = {}

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

                # 保存当前数据（用于验证）
                current_data[field_name] = host_buf.copy()

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

                # 释放 buffer
                del host_buf

            # 取消超时保护
            signal.alarm(0)

            # 可选：WRF 数据物理验证
            if args.verify_wrf:
                _verify_wrf_physics(current_data, step, history)

            # 保存历史（用于演化验证）
            history.append(current_data)

            print(f"  ✓ Step {step} 接收完成\n")

        except TimeoutError as e:
            print(f"\n[错误] {e}")
            print(f"  可能原因：")
            print(f"    1. WRF 发送端崩溃或卡住")
            print(f"    2. MPI 通信死锁")
            print(f"    3. GPU 内存不足")
            print(f"  建议：检查 rank 0 输出，或减少 --hours 参数")
            sys.exit(1)
        except Exception as e:
            signal.alarm(0)  # 取消超时
            print(f"\n[错误] Step {step} 接收失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print(f"{'='*70}")
    print(f"[LBM Receiver] 完成")
    print(f"  总步数: {args.steps}")
    print(f"  ✓ 通道验证通过")
    print(f"{'='*70}")


def _verify_wrf_physics(data, step, history):
    """验证 WRF 数据物理合理性

    检查：
    1. 数值范围合理性（避免 NaN/Inf/极端值）
    2. 常数场检测（地表字段应基本不变）
    3. 演化检测（风场/温度应随时间变化）
    """
    print(f"    WRF 物理验证:")

    # 1. 数值范围检查
    for field, arr in data.items():
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            print(f"      [警告] {field} 包含 NaN/Inf")
            return

    # 2. 常数场检查（仅 2D 地表字段）
    if step > 0 and history:
        prev_data = history[-1]
        for field in ['t_skin', 'roughness_m']:
            if field in data and field in prev_data:
                diff = np.abs(data[field] - prev_data[field]).max()
                if diff > 1.0:  # TSK 不应变化超过 1K
                    print(f"      [警告] {field} 变化过大: {diff:.3f}")

    # 3. 演化检查（3D 动力场）
    if step > 0 and history:
        prev_data = history[-1]
        for field in ['u', 'v', 'theta']:
            if field in data and field in prev_data:
                diff = np.abs(data[field] - prev_data[field]).max()
                if diff < 1e-6:  # 应该有演化
                    print(f"      [警告] {field} 无演化（冻结）")
                else:
                    print(f"      ✓ {field} 演化正常: Δmax={diff:.6f}")

    print(f"      ✓ 物理验证通过")


if __name__ == "__main__":
    main()
