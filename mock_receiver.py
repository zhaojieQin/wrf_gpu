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

# Defer cupy import to main() to allow GPU assignment in parent process
print("[DEBUG] all imports done (cupy deferred)", flush=True)


class TimeoutError(Exception):
    """接收超时异常"""
    pass


def timeout_handler(signum, frame):
    """超时信号处理器"""
    raise TimeoutError("接收超时（60s）")


def main(steps=None):
    """LBM receiver main entry point.

    Args:
        steps: Number of coupling steps to receive. If None, read from sys.argv.
    """
    global cp  # Make cupy available to helper functions

    print("[DEBUG] entering main()", flush=True)

    # Import cupy after GPU assignment
    import cupy as cp
    print(f"[DEBUG] cupy imported, device={cp.cuda.runtime.getDevice()}", flush=True)

    # Parse arguments or use provided steps
    if steps is None:
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
        print(f"[DEBUG] args parsed: steps={args.steps}", flush=True)
    else:
        # Direct invocation with steps parameter
        class Args:
            pass
        args = Args()
        args.steps = steps
        args.verify_gpu = False
        args.verify_wrf = True
        args.write_netcdf = True
        args.output_dir = './received_nc_swift'
        print(f"[DEBUG] using provided steps={steps}", flush=True)

    comm = MPI.COMM_WORLD
    print(f"[DEBUG] comm obtained: {comm}", flush=True)

    rank = comm.Get_rank()
    print(f"[DEBUG] rank={rank}", flush=True)

    size = comm.Get_size()
    print(f"[DEBUG] size={size}", flush=True)

    # 检查 rank
    if rank != 1:
        print(f"[DEBUG] rank != 1, rank={rank}", flush=True)
        if rank == 0:
            # Rank 0 静默等待（WRF 发送端）
            pass
        else:
            print(f"[错误] mock_receiver.py 必须运行在 rank 1（当前 rank={rank}）")
            sys.exit(1)
        return

    print("[DEBUG] passed rank check", flush=True)
    print(f"[LBM Receiver] 启动", flush=True)
    print(f"  Rank: {rank}", flush=True)
    print(f"  接收步数: {args.steps}", flush=True)
    print(f"  GPU 验证: {args.verify_gpu}", flush=True)
    print(f"  WRF 验证: {args.verify_wrf}", flush=True)
    print(f"  写 NetCDF: {args.write_netcdf}", flush=True)
    if args.write_netcdf:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  输出目录: {output_dir.resolve()}", flush=True)
    print("[DEBUG] about to call cp.cuda.runtime.getDevice()", flush=True)
    device_id = cp.cuda.runtime.getDevice()
    print(f"[DEBUG] getDevice() returned: {device_id}", flush=True)
    print(f"  CUDA_VISIBLE_DEVICES: {device_id}", flush=True)
    print(flush=True)

    # Wait for sender (rank 0) to be ready before entering receive loop
    print(f"[LBM Receiver] Waiting for rank 0 to be ready (MPI Barrier)...", flush=True)
    comm.Barrier()
    print(f"[LBM Receiver] Barrier passed, waiting for ready signal...", flush=True)

    # Ready 握手：等待发送端编译完成（tag=999999 控制消息专用）
    READY_TAG = 999999
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(600)  # 600 秒超时等待 ready
    try:
        import datetime
        ready_wait_start = datetime.datetime.now()
        print(f"[LBM Receiver] [{ready_wait_start.strftime('%H:%M:%S.%f')[:-3]}] Waiting for ready signal (tag={READY_TAG}, timeout=600s)...", flush=True)
        ready_msg = comm.recv(source=0, tag=READY_TAG)
        signal.alarm(0)  # 取消长超时
        ready_received = datetime.datetime.now()
        wait_time = (ready_received - ready_wait_start).total_seconds()
        print(f"[LBM Receiver] [{ready_received.strftime('%H:%M:%S.%f')[:-3]}] Ready signal received (waited {wait_time:.1f}s): {ready_msg}", flush=True)
    except TimeoutError:
        signal.alarm(0)
        print(f"[错误] Ready signal timeout (600s) - sender (rank 0) 未在 10 分钟内准备好", flush=True)
        print(f"  可能原因：AOT 加载或编译失败", flush=True)
        sys.exit(1)
    except Exception as e:
        signal.alarm(0)
        print(f"[错误] Ready signal 接收失败: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"[LBM Receiver] Ready signal confirmed, starting 60s-timeout receive loop", flush=True)

    # 历史数据（用于演化验证）
    history = []

    # 循环接收 N 个 coupling steps
    for step in range(args.steps):
        # 设置超时保护
        # 第一个 step 需要更长超时（WRF 需要完成 AOT 加载 + 第一个时间步计算）
        # 后续 step 使用标准 60s 超时
        timeout_seconds = 180 if step == 0 else 60
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            print(f"{'='*70}", flush=True)
            print(f"[Step {step}] 开始接收（超时 {timeout_seconds}s）", flush=True)
            print(f"{'='*70}", flush=True)

            # 1. 接收元数据（comm.recv，小写，pickle）
            meta_tag = step * 1000
            print(f"[DEBUG] about to recv metadata, tag={meta_tag}", flush=True)
            metadata = comm.recv(source=0, tag=meta_tag)
            print(f"[DEBUG] comm.recv() returned, tag={meta_tag}", flush=True)
            print(f"[DEBUG] metadata content: {metadata}", flush=True)
            print(f"[DEBUG] metadata received: {len(metadata)} fields", flush=True)
            print(f"  元数据接收: {len(metadata)} 个字段", flush=True)

            # Synchronization barrier to confirm receipt
            print(f"[DEBUG] Entering barrier after metadata receive", flush=True)
            comm.Barrier()
            print(f"[DEBUG] Barrier passed - sender confirmed", flush=True)

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
                print(f"    {field_name}:", flush=True)
                print(f"      shape={shape}, dtype={dtype}, size={size_mb:.2f} MB", flush=True)
                print(f"      range=[{val_min:.6f}, {val_max:.6f}]", flush=True)

                # 可选：传输到 GPU 验证
                if args.verify_gpu:
                    gpu_data = cp.asarray(host_buf)
                    gpu_min = float(cp.min(gpu_data))
                    gpu_max = float(cp.max(gpu_data))
                    print(f"      GPU 验证: [{gpu_min:.6f}, {gpu_max:.6f}]", flush=True)

                    # 验证 CPU 和 GPU 数据一致
                    if abs(gpu_min - val_min) > 1e-6 or abs(gpu_max - val_max) > 1e-6:
                        print(f"      [警告] CPU/GPU 数据不一致！", flush=True)

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
                print(f"  ✓ 写入 NetCDF: {output_dir}/received_step_{step:02d}.nc", flush=True)

            # 最后 3 步额外写 wrfout 格式文件
            if step >= args.steps - 3:
                _write_wrfout(current_data, step, output_dir)

            # 保存历史（用于演化验证）
            history.append(current_data)

            print(f"  ✓ Step {step} 接收完成\n", flush=True)

        except TimeoutError as e:
            print(f"\n[错误] {e}", flush=True)
            print(f"  可能原因：", flush=True)
            print(f"    1. WRF 发送端崩溃或卡住", flush=True)
            print(f"    2. MPI 通信死锁", flush=True)
            print(f"    3. GPU 内存不足", flush=True)
            print(f"  建议：检查 rank 0 输出，或减少 --hours 参数", flush=True)
            sys.exit(1)
        except Exception as e:
            signal.alarm(0)  # 取消超时
            print(f"\n[错误] Step {step} 接收失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print(f"{'='*70}", flush=True)
    print(f"[LBM Receiver] 完成", flush=True)
    print(f"  总步数: {args.steps}", flush=True)
    print(f"  ✓ 通道验证通过", flush=True)
    print(f"{'='*70}", flush=True)

    # EOF handshake
    EOF_TAG = 999998
    print(f"[LBM Receiver] Waiting for EOF signal (tag={EOF_TAG})", flush=True)
    eof = comm.recv(source=0, tag=EOF_TAG)
    assert eof.get("eof") is True, f"Expected EOF signal, got {eof}"
    print(f"[LBM Receiver] EOF received: {eof}", flush=True)

    # Send ACK
    ack_msg = {"ack": True, "total_received": args.steps}
    comm.send(ack_msg, dest=0, tag=EOF_TAG)
    print(f"[LBM Receiver] ACK sent: {ack_msg}", flush=True)


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
        print(f"    [警告] netCDF4 未安装，跳过写入", flush=True)
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
        print(f"    [警告] 无法推断网格尺寸，跳过 NetCDF 写入", flush=True)
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


def _write_wrfout(data, step, output_dir):
    """将接收到的数据写成 WRF 格式 NetCDF 文件

    文件命名：wrfout_d03_SS（SS 为两位步数）
    变量名使用 WRF 标准名（U, V, T, QVAPOR 等）
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        print(f"    [警告] netCDF4 未安装，跳过 wrfout 写入", flush=True)
        return

    # 从标准质量场推断 nz, ny, nx
    nz = ny = nx = None
    for field in ('theta', 'qv'):
        if field in data and data[field].ndim == 3:
            nz, ny, nx = data[field].shape
            break
    if nz is None:
        print(f"    [警告] 无法推断网格尺寸，跳过 wrfout 写入", flush=True)
        return

    # WRF 内部变量名映射
    # theta 在 WRF 中存储为扰动位温 T' = theta - 300
    _WRF_VARMAP = {
        'u':     ('U',      'float32', 'm s-1',   'x-wind component'),
        'v':     ('V',      'float32', 'm s-1',   'y-wind component'),
        'theta': ('T',      'float32', 'K',       'perturbation potential temperature (theta-t0)'),
        'qv':    ('QVAPOR', 'float32', 'kg kg-1', 'Water vapor mixing ratio'),
    }

    fname = output_dir / f'wrfout_d03_{step:02d}'
    with Dataset(str(fname), 'w', format='NETCDF4') as ds:
        # 全局属性（模仿 WRF 格式）
        ds.TITLE  = f'WRF-GPU coupling output step {step}'
        ds.coupling_step = step

        # 创建维度
        ds.createDimension('bottom_top',       nz)
        ds.createDimension('bottom_top_stag',  nz + 1)
        ds.createDimension('south_north',      ny)
        ds.createDimension('south_north_stag', ny + 1)
        ds.createDimension('west_east',        nx)
        ds.createDimension('west_east_stag',   nx + 1)

        # 维度推断：根据 shape 相对于 (nz,ny,nx) 的差异判断 stagger
        def _wrf_dims(shape):
            nz_s, ny_s, nx_s = shape
            z = 'bottom_top_stag'  if nz_s == nz + 1 else 'bottom_top'
            y = 'south_north_stag' if ny_s == ny + 1 else 'south_north'
            x = 'west_east_stag'   if nx_s == nx + 1 else 'west_east'
            return (z, y, x)

        for field_name, arr in data.items():
            if field_name not in _WRF_VARMAP:
                continue
            wrf_name, out_dtype, units, desc = _WRF_VARMAP[field_name]
            dims = _wrf_dims(arr.shape)

            out_arr = arr.astype(out_dtype)
            # theta → T (扰动位温)
            if field_name == 'theta':
                out_arr = out_arr - 300.0

            var = ds.createVariable(wrf_name, out_dtype, dims, zlib=True, complevel=4)
            var[:] = out_arr
            var.units       = units
            var.description = desc

    print(f"  ✓ wrfout 写入: {fname}", flush=True)


def _verify_wrf_physics(data, step, history):
    """验证 WRF 数据物理合理性

    检查：
    1. 数值范围合理性（避免 NaN/Inf/极端值）
    2. 常数场检测（地表字段应基本不变）
    3. 演化检测（风场/温度应随时间变化）
    """
    print(f"    WRF 物理验证:", flush=True)

    # 1. 数值范围检查
    for field, arr in data.items():
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            print(f"      [警告] {field} 包含 NaN/Inf", flush=True)
            return

    # 2. 常数场检查（仅 2D 地表字段）
    if step > 0 and history:
        prev_data = history[-1]
        for field in ['t_skin', 'roughness_m']:
            if field in data and field in prev_data:
                diff = np.abs(data[field] - prev_data[field]).max()
                if diff > 1.0:  # TSK 不应变化超过 1K
                    print(f"      [警告] {field} 变化过大: {diff:.3f}", flush=True)

    # 3. 演化检查（3D 动力场）
    if step > 0 and history:
        prev_data = history[-1]
        for field in ['u', 'v', 'theta']:
            if field in data and field in prev_data:
                diff = np.abs(data[field] - prev_data[field]).max()
                if diff < 1e-6:  # 应该有演化
                    print(f"      [警告] {field} 无演化（冻结）", flush=True)
                else:
                    print(f"      ✓ {field} 演化正常: Δmax={diff:.6f}", flush=True)

    print(f"      ✓ 物理验证通过", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Rank 1] FATAL: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.COMM_WORLD.Abort(1)
        else:
            sys.exit(1)
