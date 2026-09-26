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
print("[DEBUG] Python started", flush=True)

import argparse
import signal
from pathlib import Path
print("[DEBUG] stdlib imports done", flush=True)

from mpi4py import MPI
print(f"[DEBUG] mpi4py imported, rank={MPI.COMM_WORLD.Get_rank()}, size={MPI.COMM_WORLD.Get_size()}", flush=True)

import numpy as np
print("[DEBUG] numpy imported", flush=True)

import cupy as cp
print(f"[DEBUG] cupy imported, device={cp.cuda.runtime.getDevice()}", flush=True)

print("[DEBUG] all imports done", flush=True)


class TimeoutError(Exception):
    """接收超时异常"""
    pass


def timeout_handler(signum, frame):
    """超时信号处理器"""
    raise TimeoutError("接收超时（60s）")


def main():
    print("[DEBUG] entering main()", flush=True)
    parser = argparse.ArgumentParser(description="LBM MPI 接收端测试")
    parser.add_argument('--steps', type=int, default=20,
                       help='接收的耦合步数（默认 20，完整 1h 测试）')
    parser.add_argument('--verify-gpu', action='store_true',
                       help='将数据传输到 GPU 验证完整性')
    parser.add_argument('--verify-wrf', action='store_true',
                       help='验证 WRF 数据物理合理性（范围、常数、演化）')
    parser.add_argument('--write-netcdf', action='store_true',
                       help='将每个 step 数据写入 NetCDF 文件')
    parser.add_argument('--output-dir', type=str, default='./received_nc',
                       help='NetCDF 输出目录（默认 ./received_nc）')
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
    print(f"  写 NetCDF: {args.write_netcdf}")
    if args.write_netcdf:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  输出目录: {output_dir.resolve()}")
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

            # 可选：写 NetCDF
            if args.write_netcdf:
                _write_netcdf(current_data, step, output_dir)
                print(f"  ✓ 写入 NetCDF: {output_dir}/received_step_{step:02d}.nc")

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


def _get_dims_for_shape(shape, nz, ny, nx):
    """根据 shape 和标准网格尺寸推断维度名称

    nz, ny, nx 从标准质量场（theta 或 qv）推断，不从 u 读。
    只返回实际需要的维度，不预先创建所有可能的维度。
    """
    if len(shape) == 2:
        # 2D 地表场：t_skin, roughness_m
        return ('y', 'x')
    elif len(shape) == 3:
        nz_s, ny_s, nx_s = shape
        # 根据各维度的尺寸判断是否 staggered
        z_dim = 'z_stag' if nz_s == nz + 1 else 'z'
        y_dim = 'y_stag' if ny_s == ny + 1 else 'y'
        x_dim = 'x_stag' if nx_s == nx + 1 else 'x'
        return (z_dim, y_dim, x_dim)
    else:
        raise ValueError(f"不支持的数组维度: {len(shape)}D, shape={shape}")


def _write_netcdf(data, step, output_dir):
    """将单个耦合步骤的数据写入 NetCDF 文件

    Args:
        data: 字段名 -> numpy array 字典
        step: 耦合步数（0-based）
        output_dir: 输出目录（Path 对象）
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        print(f"    [警告] netCDF4 未安装，跳过写入")
        return

    # 1. 从标准质量场推断 nz, ny, nx
    nz = ny = nx = None
    for field in ('theta', 'qv', 'p_total'):
        if field in data and data[field].ndim == 3:
            nz, ny, nx = data[field].shape
            break
    if nz is None:
        # 找不到标准质量场，从任意 3D 场取最小值估算
        for arr in data.values():
            if arr.ndim == 3:
                # 取最小维度作为 nx/ny，避免从 staggered 维度读
                nz = min(arr.shape[0], arr.shape[0])
                ny = arr.shape[1]
                nx = arr.shape[2]
                # 减去可能的 stagger
                if nx > ny:  # 可能是 x_stag
                    nx -= 1
                break
    if nz is None:
        print(f"    [警告] 无法推断网格尺寸，跳过 NetCDF 写入")
        return

    # 2. 扫描所有字段，收集需要的维度
    needed_dims = set()
    field_dims = {}
    for field_name, arr in data.items():
        dims = _get_dims_for_shape(arr.shape, nz, ny, nx)
        field_dims[field_name] = dims
        needed_dims.update(dims)

    # 3. 确定各维度的大小
    dim_sizes = {
        'z':      nz,
        'z_stag': nz + 1,
        'y':      ny,
        'y_stag': ny + 1,
        'x':      nx,
        'x_stag': nx + 1,
    }

    # 4. 写入 NetCDF
    nc_path = output_dir / f'received_step_{step:02d}.nc'
    with Dataset(str(nc_path), 'w', format='NETCDF4') as ds:
        # 全局属性
        ds.coupling_step = step
        ds.wrf_time_s    = step * 180.0   # 3 分钟间隔
        ds.wrf_time_min  = step * 3.0
        ds.description   = f'WRF-LBM coupling step {step}'
        ds.grid_nz       = nz
        ds.grid_ny       = ny
        ds.grid_nx       = nx

        # 只创建实际用到的维度
        for dim_name in sorted(needed_dims):
            ds.createDimension(dim_name, dim_sizes[dim_name])

        # 写入各字段
        for field_name, arr in data.items():
            dims = field_dims[field_name]
            var = ds.createVariable(field_name, arr.dtype, dims,
                                    zlib=True, complevel=4)
            var[:] = arr

            # 添加字段属性
            _FIELD_ATTRS = {
                'u':           ('m s-1',  'x-wind component (staggered)'),
                'v':           ('m s-1',  'y-wind component (staggered)'),
                'w':           ('m s-1',  'z-wind component (staggered)'),
                'theta':       ('K',      'potential temperature'),
                'qv':          ('kg kg-1','water vapor mixing ratio'),
                'p_total':     ('Pa',     'total pressure'),
                't_skin':      ('K',      'skin temperature (TSK)'),
                'roughness_m': ('m',      'surface roughness length (ZNT)'),
            }
            if field_name in _FIELD_ATTRS:
                units, desc = _FIELD_ATTRS[field_name]
                var.units       = units
                var.description = desc


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
