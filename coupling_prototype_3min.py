#!/usr/bin/env python3
"""
WRF-LBM 3 分钟耦合原型（最干净路径）

设计：
- 使用 auxhist 配置触发 3 分钟分段
- forecast_fn 回调提取边界数据
- 模拟发送到 LBM（实际发送逻辑待 LBM 接口确认）
- 不修改 gpuwrf 内部代码

用法：
    # DRY 模式（验证接口，不发送数据）
    python coupling_prototype_3min.py --dry --hours 1

    # WET 模式（提取并模拟发送）
    python coupling_prototype_3min.py --wet --hours 1
"""

import argparse
import sys
from pathlib import Path
import time

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
    """3 分钟耦合原型（基于 auxhist 触发）

    工作原理：
    1. 配置 auxhist interval_minutes=3
    2. daily_pipeline 自动按 3 分钟分段调用 forecast_fn
    3. 每次调用时提取边界数据并发送到 LBM
    """

    def __init__(
        self,
        alignment: CouplingAlignment,
        dt_lbm: float,
        dry_run: bool = False,
    ):
        """
        Args:
            alignment: 耦合对齐参数（来自 compute_coupling_alignment）
            dt_lbm: LBM 时间步长（秒）
            dry_run: True=验证接口不发送，False=提取并发送
        """
        self.alignment = alignment
        self.dt_lbm = dt_lbm
        self.dry_run = dry_run
        self.call_count = 0
        self.total_hours = 0.0
        self.history = []

    def __call__(self, state, namelist, hours):
        """forecast_fn 签名：(State, OperationalNamelist, float) -> State

        注意：
        - hours 参数由 daily_pipeline 传入（应为 0.05，即 3 分钟）
        - 每次推进 N_wrf 步（对于 dt=18s，N=10）
        - 返回推进后的 state
        """
        self.call_count += 1

        print(f"\n{'='*70}")
        print(f"[耦合回调 #{self.call_count}] {'DRY (验证)' if self.dry_run else 'WET (发送)'} 模式")
        print(f"  累积时间: {self.total_hours:.3f}h → {self.total_hours + hours:.3f}h")
        print(f"  本次推进: {hours*60:.2f} 分钟 ({hours*3600:.0f}s)")
        print(f"  hours 参数实际值: {hours:.6f}h ({hours*60:.4f} min)")  # ← 新增：打印实际值
        print(f"  预期步数: {self.alignment['n_wrf_steps']} WRF 步")

        # ============================================================
        # 1. 提取边界数据
        # ============================================================
        if not self.dry_run:
            print(f"  提取边界数据...")
            boundary_data = self._extract_boundary_fields(state)

            # 打印统计信息（触发 GPU 同步）
            for name, arr in boundary_data.items():
                val_min = float(jnp.min(arr))
                val_max = float(jnp.max(arr))
                print(f"    {name}: shape={arr.shape}, "
                      f"range=[{val_min:.2f}, {val_max:.2f}]")

            # ============================================================
            # 2. 发送到 LBM（模拟）
            # ============================================================
            print(f"  模拟发送到 LBM...")
            self._send_to_lbm(boundary_data, self.call_count)
            print(f"  ✓ 发送完成")
        else:
            print(f"  [DRY] 跳过边界提取和发送")

        # ============================================================
        # 3. 调用真实预报（分段编译模式）
        # ============================================================
        print(f"  调用真实预报（分段编译）...")

        from gpuwrf.integration.daily_pipeline import _segmented_forecast_fn
        result = _segmented_forecast_fn(state, namelist, hours)

        print(f"  ✓ 预报完成")
        print(f"{'='*70}\n")

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

    def _extract_boundary_fields(self, state):
        """提取边界层字段（占位实现，待 LBM 需求确认）

        TODO: 待 VirtualFluids 团队确认：
        - 需要哪些物理量？（u, v, w, T, p, rho, qv...）
        - 需要哪个边界？（西/东/南/北/顶/底）
        - 需要边界法向速度还是切向速度？
        - 需要原始模式变量还是诊断变量？
        - 数据格式要求？（单位、坐标系）

        当前占位实现：提取西边界（i=0）的基本流场变量
        """
        # 西边界（最左列）- 占位实现
        boundary = {
            'u': state.u[:, :, 0],          # (nz, ny) 东西向风速
            'v': state.v[:, :, 0],          # (nz, ny) 南北向风速
            'theta': state.theta[:, :, 0],  # (nz, ny) 位温
            'qv': state.qv[:, :, 0],        # (nz, ny) 水汽混合比
        }
        return boundary

    def _send_to_lbm(self, boundary_data, coupling_step):
        """发送边界数据到 LBM（模拟）

        实际实现取决于：
        1. VirtualFluids 的边界条件 API
        2. 进程间通信方式（shm / socket / file）
        3. 数据格式（numpy / binary / NetCDF）

        当前仅模拟：计算数据大小、打印统计
        """
        import numpy as np

        # 转换为 numpy（触发 GPU→CPU）
        boundary_np = {k: np.asarray(v) for k, v in boundary_data.items()}

        # 计算数据大小
        total_elements = sum(v.size for v in boundary_np.values())
        total_bytes = sum(v.nbytes for v in boundary_np.values())
        total_mb = total_bytes / 1e6

        print(f"    边界数据: {len(boundary_np)} 个字段, "
              f"{total_elements} 个元素, {total_mb:.2f} MB")
        print(f"    LBM 侧应运行: {self.alignment['m_lbm_steps']} 步 "
              f"(时长 {self.alignment['actual_coupling_s']}s)")

        # TODO: 实际发送逻辑
        # 方案 1：共享内存
        #   shm = shared_memory.SharedMemory(name=f"wrf_lbm_coupling_{coupling_step}")
        #   np_array = np.ndarray(shape, dtype, buffer=shm.buf)
        #   np_array[:] = boundary_np['u']

        # 方案 2：Unix socket
        #   socket.send(pickle.dumps(boundary_np))

        # 方案 3：文件（简单但慢）
        #   np.savez(f"coupling_step_{coupling_step:04d}.npz", **boundary_np)

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

  # 指定 LBM 时间步
  python coupling_prototype_3min.py --wet --hours 1 --dt-lbm 0.05
        """
    )

    # 模式选择
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--dry', action='store_true',
                           help='DRY 模式：验证接口，不提取/发送边界数据')
    mode_group.add_argument('--wet', action='store_true',
                           help='WET 模式：提取边界数据并模拟发送')

    # 预报参数
    parser.add_argument('--hours', type=int, default=1,
                       help='预报小时数（默认 1）')
    parser.add_argument('--domain', type=str, default='d01',
                       help='域名（默认 d01）')

    # 耦合参数
    parser.add_argument('--dt-wrf', type=float, default=18.0,
                       help='WRF 时间步长（秒，默认 18.0）')
    parser.add_argument('--dt-lbm', type=float, default=0.1,
                       help='LBM 时间步长（秒，默认 0.1，待 VirtualFluids 确认）')
    parser.add_argument('--desired-interval', type=float, default=60.0,
                       help='期望耦合间隔（秒，默认 60.0）')

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

    if args.output_dir is None:
        output_dir = repo_root / 'runs' / ('coupling_3min_dry' if dry_run else 'coupling_3min_wet')
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

    try:
        from gpuwrf.integration.daily_pipeline import (
            execute_daily_pipeline,
            DailyPipelineConfig,
        )
        from gpuwrf.io.auxhist_stream import AuxhistStreamConfig
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
    )

    # 配置 auxhist 触发 3 分钟分段
    config = DailyPipelineConfig(
        run_id=str(input_dir),
        run_root=input_dir.parent,
        domain=args.domain,
        hours=hours,
        output_dir=output_dir,
        proof_dir=output_dir / "proofs",
        auxhist=AuxhistStreamConfig(
            stream_id=1,
            interval_minutes=alignment['auxhist_interval_minutes'],  # 3 分钟
            variables=("T2",),  # 最小变量集（减少 auxhist 文件大小）
        ),
    )

    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  域: {args.domain}")
    print(f"  预报小时数: {hours}")
    print(f"  耦合间隔: {alignment['actual_coupling_minutes']:.0f} 分钟")
    print(f"  预期 forecast_fn 调用: ~{hours * alignment['substeps']} 次")
    print(f"  模式: {'DRY (验证接口)' if dry_run else 'WET (提取+发送)'}")

    # =====================================================
    # 5. 执行
    # =====================================================
    print(f"\n[5/5] 执行耦合测试...")
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
    print(f"Auxhist files: {len(result.get('auxhist_files', []))}")  # ← 垃圾文件

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

    if dry_run:
        print(f"\n下一步：运行 WET 模式")
        print(f"  python {Path(__file__).name} --wet --hours 1")
    else:
        print(f"\n下一步：")
        print(f"  1. 确认 LBM 侧接口（边界条件 API、dt_lbm 实测值）")
        print(f"  2. 实现 _send_to_lbm 真实发送逻辑")
        print(f"  3. 运行完整耦合测试（WRF + LBM 双向联调）")

    print(f"\n⚠️  注意：输出目录有 auxhist 垃圾文件（{len(result.get('auxhist_files', []))} 个）")
    print(f"  位置: {output_dir}/auxhist1_d01_*.nc")
    print(f"  如需清理: rm -f {output_dir}/auxhist1_d01_*.nc")

    sys.exit(0)


if __name__ == "__main__":
    main()
