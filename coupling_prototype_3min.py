#!/usr/bin/env python3
"""
WRF-LBM 3 分钟耦合原型（真实 MPI 通信）

设计：
- 使用 coupling_interval_minutes 配置触发 3 分钟分段（无 auxhist 开销）
- forecast_fn 回调提取 3D 子域数据
- 通过 MPI 发送到 LBM（rank 1）
- 不修改 gpuwrf 内部代码

WRF-LBM 数据协议 v1.0：
传输内容：3D 子域（不是边界面）
字段列表：
  - u: (nz, ny, nx+1) - staggered on x-faces
  - v: (nz, ny+1, nx) - staggered on y-faces
  - w: (nz+1, ny, nx) - staggered on z-faces
  - theta: (nz, ny, nx) - mass points
  - qv: (nz, ny, nx) - mass points
  - p_total: (nz, ny, nx) - mass points
  - t_skin: (ny, nx) - 2D surface field
  - roughness_m: (ny, nx) - 2D surface field

用法：
    # DRY 模式（验证接口，不发送数据）
    python coupling_prototype_3min.py --dry --hours 1.0

    # MPI 模式（真实发送到 LBM rank 1）
    python coupling_prototype_3min.py --mpi --hours 0.05
"""

import argparse
import sys
from pathlib import Path
import time
import os

# 延迟导入（避免无 JAX 环境报错）
try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

from coupling_alignment import (
    compute_coupling_alignment,
    print_coupling_alignment_report,
    CouplingAlignment,
)


class CoupledForecast3Min:
    """3 分钟耦合原型（基于 coupling_interval_minutes 触发）

    工作原理：
    1. 配置 coupling_interval_minutes=3
    2. daily_pipeline 自动按 3 分钟分段调用 forecast_fn
    3. 每次调用时提取指定 3D 子域并发送到 LBM
    4. 无 auxhist 文件写入开销（移除 ~45s/h）
    """

    def __init__(
        self,
        alignment: CouplingAlignment,
        dt_lbm: float,
        dry_run: bool = False,
        subdomain_config: dict | None = None,
    ):
        """
        Args:
            alignment: 耦合对齐参数（来自 compute_coupling_alignment）
            dt_lbm: LBM 时间步长（秒）
            dry_run: True=验证接口不发送，False=真实 MPI 发送
            subdomain_config: 3D 子域提取配置

        内存安全性说明（选项 B - WRF 计算后等待）:
        1. _extract_subdomain() 返回的是 JAX array view（引用 state）
        2. WRFMPISender.send_subdomain() 内部立即 cp.array() 显式拷贝
        3. cp.asnumpy() 同步传输，返回时数据已在 host
        4. 因此 state 在 _segmented_forecast_fn 中被修改不影响 MPI buffer
        5. wait_all() 只需在 __call__ 返回前调用，无需在发送后立即等待
        """
        self.alignment = alignment
        self.dt_lbm = dt_lbm
        self.dry_run = dry_run
        self.subdomain_config = subdomain_config or {
            'z_range': (0, -1),
            'y_range': (0, -1),
            'x_range': (0, -1),
            'fields': [
                # 3D 风场（staggered grid）
                'u', 'v', 'w',
                # 3D 热力学场
                'theta',         # 位温（LBM 端计算热量平流）
                'qv',            # 水汽混合比
                'p_total',       # 总压力（用于密度加权，如果 LBM 需要）
                # 2D 地表场
                't_skin',        # TSK: 地表温度
                'roughness_m',   # ZNT: 地表粗糙度
            ],
        }
        self.call_count = 0
        self.total_hours = 0.0
        self.history = []

        # 初始化 MPI 发送端
        if not dry_run:
            from wrf_mpi_sender import WRFMPISender
            self.mpi_sender = WRFMPISender(dest_rank=1)
        else:
            self.mpi_sender = None

    def __call__(self, state, namelist, hours):
        """forecast_fn 签名：(State, OperationalNamelist, float) -> State

        注意：
        - hours 参数由 daily_pipeline 传入（应为 0.05，即 3 分钟）
        - 每次推进 N_wrf 步（对于 dt=18s，N=10）
        - 返回推进后的 state
        """
        step_start_time = time.time()
        self.call_count += 1

        print(f"\n{'='*70}")
        print(f"[耦合回调 #{self.call_count}] 开始时间: {time.strftime('%H:%M:%S')}")
        print(f"  模式: {'DRY (验证)' if self.dry_run else 'MPI (发送)'}")
        print(f"  累积时间: {self.total_hours:.3f}h → {self.total_hours + hours:.3f}h")
        print(f"  本次推进: {hours*60:.2f} 分钟 ({hours*3600:.0f}s)")
        print(f"  预期步数: {self.alignment['n_wrf_steps']} WRF 步")
        print(f"{'='*70}")

        # ============================================================
        # 1. 提取子域数据
        # ============================================================
        if not self.dry_run:
            t0 = time.time()
            print(f"\n[步骤 1/4] 提取 3D 子域...")
            subdomain_data = self._extract_subdomain(state)
            t1 = time.time()
            print(f"  ✓ 提取完成 ({t1-t0:.2f}s)")

            # 打印统计信息（触发 GPU 同步）
            for name, arr in subdomain_data.items():
                val_min = float(jnp.min(arr))
                val_max = float(jnp.max(arr))
                print(f"    {name}: shape={arr.shape}, "
                      f"range=[{val_min:.2f}, {val_max:.2f}]")

            # ============================================================
            # 2. 发送到 LBM
            # ============================================================
            t0 = time.time()
            print(f"\n[步骤 2/4] 发送到 LBM rank 1...")
            self._send_to_lbm(subdomain_data, self.call_count)
            t1 = time.time()
            print(f"  ✓ 发送完成 ({t1-t0:.2f}s)")
        else:
            print(f"\n[步骤 1-2/4] [DRY] 跳过子域提取和发送")

        # ============================================================
        # 3. 调用真实预报（分段编译模式）
        # ============================================================
        t0 = time.time()
        print(f"\n[步骤 3/4] 调用 WRF 预报...")
        sys.stdout.flush()  # 强制刷新输出

        from gpuwrf.integration.daily_pipeline import _segmented_forecast_fn
        result = _segmented_forecast_fn(state, namelist, hours)

        t1 = time.time()
        print(f"  ✓ 预报完成 ({t1-t0:.2f}s)")

        # ============================================================
        # 4. 等待 MPI 发送完成（选项 B - WRF 计算后等待）
        # ============================================================
        if not self.dry_run:
            t0 = time.time()
            print(f"\n[步骤 4/4] 等待 MPI 发送完成...")
            self.mpi_sender.wait_all()
            t1 = time.time()
            print(f"  ✓ MPI 发送完成 ({t1-t0:.2f}s)")

        # 总耗时
        step_elapsed = time.time() - step_start_time
        print(f"\n本步总耗时: {step_elapsed:.2f}s")
        print(f"{'='*70}\n")
        sys.stdout.flush()  # 强制刷新输出

        # ============================================================
        # 4. 记录历史
        # ============================================================
        self.history.append({
            'call_count': self.call_count,
            'hours_before': self.total_hours,
            'hours_advance': hours,
            'mode': 'DRY' if self.dry_run else 'WET',
        })

        self.total_hours += hours
        return result

    def _extract_subdomain(self, state):
        """提取指定的 3D 子域，传送给 LBM 端

        根据 subdomain_config 提取完整的 3D 子域（不仅仅是边界面），
        方便 LBM 端进行插值和嵌套计算。

        配置示例：
            # 提取完整 3D 子域：垂直 10-30 层，南北 20-80，东西 50-150
            {
                'z_range': (10, 30),
                'y_range': (20, 80),
                'x_range': (50, 150),
                'fields': ['u', 'v', 'w', 'theta', 'qv']
            }

            # 使用 -1 表示到末尾（全域）
            {
                'z_range': (0, -1),    # 垂直全域
                'y_range': (0, -1),    # 南北全域
                'x_range': (100, 200), # 东西 100-200
                'fields': ['u', 'v']
            }

        返回：
            dict[str, jax.Array]: 字段名 -> 3D 子域数据（保持在 GPU）
                shape = (nz_sub, ny_sub, nx_sub)
                例如：z_range=(10,30), y_range=(20,80), x_range=(50,150)
                      -> shape = (20, 60, 100)
        """
        cfg = self.subdomain_config
        fields = cfg.get('fields', ['u', 'v', 'theta', 'qv'])

        # 获取 WRF state 的完整形状
        nz, ny, nx = state.u.shape

        # 解析 3D 子域范围
        z_start, z_end = cfg.get('z_range', (0, -1))
        y_start, y_end = cfg.get('y_range', (0, -1))
        x_start, x_end = cfg.get('x_range', (0, -1))

        # 处理 -1（表示到末尾）
        if z_end == -1:
            z_end = nz
        if y_end == -1:
            y_end = ny
        if x_end == -1:
            x_end = nx

        # 创建切片
        z_slice = slice(z_start, z_end)
        y_slice = slice(y_start, y_end)
        x_slice = slice(x_start, x_end)

        # 提取 3D 子域
        subdomain = {}
        for field in fields:
            if hasattr(state, field):
                arr = getattr(state, field)
                subdomain[field] = arr[z_slice, y_slice, x_slice]

        return subdomain

    def _send_to_lbm(self, subdomain_data, coupling_step):
        """发送 3D 子域到 LBM（DRY 打印 或 MPI 真实发送）

        Args:
            subdomain_data: 字段名 -> JAX array 字典
            coupling_step: 耦合步数
        """
        # 计算数据大小
        total_elements = sum(v.size for v in subdomain_data.values())
        total_bytes = sum(v.nbytes for v in subdomain_data.values())
        total_mb = total_bytes / 1e6

        # 打印子域配置和形状
        cfg = self.subdomain_config
        print(f"    子域配置: z={cfg.get('z_range', (0, -1))}, "
              f"y={cfg.get('y_range', (0, -1))}, "
              f"x={cfg.get('x_range', (0, -1))}")
        for field, arr in subdomain_data.items():
            print(f"      {field}: shape={arr.shape}, dtype={arr.dtype}")

        print(f"    总数据量: {len(subdomain_data)} 个字段, "
              f"{total_elements} 个元素, {total_mb:.2f} MB")
        print(f"    LBM 侧应运行: {self.alignment['m_lbm_steps']} 步 "
              f"(时长 {self.alignment['actual_coupling_s']}s)")

        # MPI 发送
        if self.mpi_sender:
            n_fields = self.mpi_sender.send_subdomain(subdomain_data, coupling_step)
            print(f"    ✓ MPI Isend: {n_fields} 个字段（非阻塞）")

    def summary(self):
        """打印运行摘要"""
        print(f"\n{'='*70}")
        print(f"耦合原型运行摘要")
        print(f"{'='*70}")
        print(f"模式: {'DRY (验证)' if self.dry_run else 'WET (发送)'}")
        print(f"总调用次数: {self.call_count}")
        print(f"累积时间: {self.total_hours:.3f} 小时")
        print(f"预期 forecast_fn 调用次数: {self.total_hours / self.alignment['actual_coupling_minutes'] * 60:.0f}")
        print(f"实际调用次数: {self.call_count}")
        print(f"{'='*70}\n")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="WRF-LBM 3 分钟耦合原型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # DRY 模式（1h 验证，~3-5 分钟）
  python coupling_prototype_3min.py --dry --hours 1

  # WET 模式（1h 耦合测试）
  python coupling_prototype_3min.py --wet --hours 1

  # 提取指定 3D 子域（方便 LBM 插值）
  # 垂直 10-30 层，南北 20-80，东西 50-150
  python coupling_prototype_3min.py --wet --hours 1 \\
      --z-range 10:30 --y-range 20:80 --x-range 50:150

  # 提取完整域（全部层，全部范围）
  python coupling_prototype_3min.py --wet --hours 1 \\
      --z-range 0:-1 --y-range 0:-1 --x-range 0:-1

  # 自定义提取字段
  python coupling_prototype_3min.py --wet --hours 1 \\
      --fields u,v,w,theta,qv,p

  # 指定 LBM 时间步
  python coupling_prototype_3min.py --wet --hours 1 --dt-lbm 0.05
        """
    )

    # 模式选择
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--dry', action='store_true',
                           help='DRY 模式：验证接口，不提取/发送子域数据')
    mode_group.add_argument('--mpi', action='store_true',
                           help='MPI 模式：提取子域数据并通过 MPI 发送到 LBM rank 1')

    # 预报参数
    parser.add_argument('--hours', type=float, default=1.0,
                       help='预报小时数（默认 1.0）')
    parser.add_argument('--domain', type=str, default='d01',
                       help='域名（默认 d01）')

    # 耦合参数
    parser.add_argument('--dt-wrf', type=float, default=18.0,
                       help='WRF 时间步长（秒，默认 18.0）')
    parser.add_argument('--dt-lbm', type=float, default=0.1,
                       help='LBM 时间步长（秒，默认 0.1，待 VirtualFluids 确认）')
    parser.add_argument('--desired-interval', type=float, default=60.0,
                       help='期望耦合间隔（秒，默认 60.0）')

    # 子域提取配置
    parser.add_argument('--z-range', type=str, default='0:-1',
                       help='垂直层范围 start:end，-1 表示到末尾（默认 0:-1 全域）')
    parser.add_argument('--y-range', type=str, default='0:-1',
                       help='南北范围 start:end，-1 表示到末尾（默认 0:-1 全域）')
    parser.add_argument('--x-range', type=str, default='0:-1',
                       help='东西范围 start:end，-1 表示到末尾（默认 0:-1 全域）')
    parser.add_argument('--fields', type=str, default='u,v,theta,qv',
                       help='提取字段，逗号分隔（默认 u,v,theta,qv）')

    # 路径参数
    repo_root = Path(__file__).parent
    parser.add_argument('--input-dir', type=Path,
                       default=repo_root / 'examples' / 'switzerland_d01',
                       help='输入目录（默认 examples/switzerland_d01）')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='输出目录（默认：DRY=runs/coupling_3min_dry, WET=runs/coupling_3min_wet）')

    args = parser.parse_args()

    # =====================================================
    # 1. 计算耦合对齐参数
    # =====================================================
    print("\n[1/5] 计算耦合时间对齐参数...\n")

    try:
        alignment = compute_coupling_alignment(
            dt_wrf=args.dt_wrf,
            dt_lbm=args.dt_lbm,
            desired_interval_s=args.desired_interval,
        )
    except ValueError as e:
        print(f"❌ 错误：无法计算耦合对齐参数", file=sys.stderr)
        print(f"   {e}", file=sys.stderr)
        sys.exit(1)

    print_coupling_alignment_report(alignment, args.dt_wrf, args.dt_lbm, args.desired_interval)

    # =====================================================
    # 2. 参数验证和默认值
    # =====================================================
    dry_run = args.dry
    hours = args.hours

    # 解析子域配置
    def parse_range(s):
        parts = s.split(':')
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 else -1
        return (start, end)

    subdomain_config = {
        'z_range': parse_range(args.z_range),
        'y_range': parse_range(args.y_range),
        'x_range': parse_range(args.x_range),
        'fields': args.fields.split(','),
    }

    if args.output_dir is None:
        output_dir = repo_root / 'runs' / ('coupling_3min_dry' if dry_run else 'coupling_3min_mpi')
    else:
        output_dir = args.output_dir

    input_dir = args.input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.is_dir():
        print(f"❌ 错误：输入目录不存在: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # 清理输出目录
    if output_dir.exists():
        import shutil
        print(f"\n[2/5] 清理现有输出目录: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 3. 导入 gpuwrf
    # =====================================================
    print(f"\n[3/5] 导入 gpuwrf...")
    t0 = time.time()

    try:
        from gpuwrf.integration.daily_pipeline import (
            execute_daily_pipeline,
            DailyPipelineConfig,
        )
        t1 = time.time()
        print(f"  ✓ 导入完成 ({t1-t0:.2f}s)")
    except ImportError as e:
        print(f"❌ 错误：无法导入 gpuwrf: {e}", file=sys.stderr)
        sys.exit(1)

    # =====================================================
    # 4. 配置管线
    # =====================================================
    print(f"\n[4/5] 配置 WRF-GPU 管线...")

    # 创建耦合实例
    coupled = CoupledForecast3Min(
        alignment=alignment,
        dt_lbm=args.dt_lbm,
        dry_run=dry_run,
        subdomain_config=subdomain_config,
    )

    # 使用 coupling_interval_minutes 直接触发分段（无 auxhist 开销）
    config = DailyPipelineConfig(
        run_id=str(input_dir),
        run_root=input_dir.parent,
        domain=args.domain,
        hours=hours,
        output_dir=output_dir,
        proof_dir=output_dir / "proofs",
        coupling_interval_minutes=alignment['auxhist_interval_minutes'],  # 3 分钟
    )

    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  域: {args.domain}")
    print(f"  预报小时数: {hours}")
    print(f"  耦合间隔: {alignment['actual_coupling_minutes']:.0f} 分钟")
    print(f"  预期 forecast_fn 调用: ~{hours * 60 / alignment['actual_coupling_minutes']:.0f} 次")
    print(f"  模式: {'DRY (验证接口)' if dry_run else 'MPI (真实发送)'}")
    print(f"  子域配置: z={subdomain_config['z_range']}, "
          f"y={subdomain_config['y_range']}, "
          f"x={subdomain_config['x_range']}, "
          f"fields={subdomain_config['fields']}")

    # =====================================================
    # 5. 执行
    # =====================================================
    print(f"\n[5/5] 执行耦合测试...")
    print(f"{'='*70}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"预期总耦合步数: ~{hours * 60 / alignment['actual_coupling_minutes']:.0f} 步")
    print(f"每步推进: {alignment['actual_coupling_minutes']:.1f} 分钟")
    print(f"{'='*70}\n")

    start_time = time.time()

    try:
        result = execute_daily_pipeline(
            config,
            forecast_fn=coupled,  # ← 耦合回调
        )
    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        coupled.summary()
        sys.exit(1)

    elapsed = time.time() - start_time

    # =====================================================
    # 6. 结果验证
    # =====================================================
    coupled.summary()

    print(f"{'='*70}")
    print(f"管线执行结果")
    print(f"{'='*70}")
    verdict = result.get('verdict', 'UNKNOWN')
    print(f"Verdict: {verdict}")
    print(f"Wall clock: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"Output files: {len(result.get('wrfout_files', []))}")

    # 严格验证
    if verdict != 'PIPELINE_GREEN':
        print(f"\n❌ 验证失败：verdict = {verdict}")
        print(f"原因: {result.get('reason', 'N/A')}")
        sys.exit(1)

    if coupled.call_count == 0:
        print(f"\n❌ 验证失败：forecast_fn 未被调用")
        sys.exit(1)

    if not result.get('wrfout_files'):
        print(f"\n❌ 验证失败：未生成 wrfout 文件")
        sys.exit(1)

    # 成功
    print(f"\n{'='*70}")
    print(f"✓ 3 分钟耦合原型{'验证' if dry_run else '测试'}通过！")
    print(f"{'='*70}")
    print(f"✓ verdict = PIPELINE_GREEN")
    print(f"✓ wrfout 文件生成: {len(result.get('wrfout_files', []))} 个")
    print(f"✓ forecast_fn 调用: {coupled.call_count} 次")
    print(f"✓ 耦合间隔: {alignment['actual_coupling_minutes']:.0f} 分钟")
    print(f"✓ 无 auxhist 文件写入（使用 coupling_interval_minutes）")

    if dry_run:
        print(f"\n下一步：运行 MPI 模式")
        print(f"  python {Path(__file__).name} --mpi --hours 1.0")
    else:
        print(f"\n下一步：")
        print(f"  1. 启动完整耦合测试（WRF + LBM）")
        print(f"  2. 使用 run_wrf_lbm_coupling.sh 启动脚本")
        print(f"  3. LBM 侧验证数据范围、常数、演化")

    sys.exit(0)


if __name__ == "__main__":
    main()
