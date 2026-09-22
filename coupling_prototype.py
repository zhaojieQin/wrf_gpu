#!/usr/bin/env python3
"""
forecast_fn 耦合原型（生产就绪版 - 低内存优化）

用法：
    # DRY 模式（2h 快速验证，约 10-15 分钟）
    python coupling_prototype.py --dry

    # WET 模式（3h 耦合测试，约 15-20 分钟）
    python coupling_prototype.py --wet

    # 自定义参数
    python coupling_prototype.py --dry --hours 1 --input-dir /custom/path

模式说明：
- DRY (验证模式): 在回调里读/写 state，但返回真实预报结果（保持 pipeline 正确性）
- WET (耦合模式): 修改 state 后运行真实预报（完整耦合路径）
  建议：限制 2-3h 预报，避免 +1K 累积导致跑飞

关键设计：
1. 使用 _segmented_forecast_fn（分段编译，O(1) 内存 vs 单片段 O(N) 内存）
2. state.replace 后必须 dealias_state_buffers（避免 donate 崩溃）
3. 统计操作会触发 GPU 同步（仅用于验证，不适合性能测量）

内存优化（实测瑞士 d01）：
- 单片段编译（_default_forecast_fn）：1h 编译 5 分钟，峰值 ~40GB，2h OOM
- 分段编译（_segmented_forecast_fn）：1h 编译 2-3 分钟，峰值 ~15GB，稳定
"""

import argparse
import sys
from pathlib import Path
import jax.numpy as jnp


class CoupledForecastPrototype:
    """
    最小耦合原型，支持 DRY（验证）和 WET（耦合）两种模式

    DRY 模式（dry_run=True）：
        - 读取 state 各字段（验证可访问）
        - 执行 replace 操作（验证接口）
        - 调用真实 forecast 并返回其结果（保持 pipeline 正确性）
        - 注意：也会真跑 forecast，使用分段编译降低内存

    WET 模式（dry_run=False）：
        - 修改 state（theta +1K）
        - 调用 dealias_state_buffers（避免 donate 崩溃）
        - 调用真实 forecast（分段编译）
        - 建议：2-3h 预报，避免累积扰动导致跑飞

    内存优化：
        - 使用 _segmented_forecast_fn（分段编译）而非 _default_forecast_fn（单片段）
        - 单片段：O(N) 内存，1h ~40GB，2h OOM
        - 分段：O(1) 内存，1h ~15GB，可扩展到 24h+
    """

    def __init__(self, perturbation_k=1.0, dry_run=False):
        """
        Args:
            perturbation_k: WET 模式下对 theta 添加的扰动（开尔文）
            dry_run: True = 验证模式（读写接口，返回真实结果）
                     False = 耦合模式（真实修改 state）
        """
        self.perturbation_k = perturbation_k
        self.dry_run = dry_run
        self.call_count = 0
        self.total_hours = 0.0
        self.history = []

    def __call__(self, state, namelist, hours):
        """
        forecast_fn 签名：(State, OperationalNamelist, float) -> State

        注意：
        - 每 substep 调用一次（默认 substeps=1，每小时 1 次）
        - hours 参数是本次推进的小时数（通常 1.0 或分段值）
        - 必须返回推进后的 State（不能返回未推进的 state）
        """
        self.call_count += 1

        print(f"\n{'='*70}")
        print(f"[耦合回调 #{self.call_count}] {'DRY (验证)' if self.dry_run else 'WET (耦合)'} 模式")
        print(f"  累积时间: {self.total_hours:.3f}h → {self.total_hours + hours:.3f}h")
        print(f"  本次推进: {hours:.4f}h ({hours * 60:.2f} 分钟)")

        # ============================================================
        # 1. 读取 state 信息（验证可访问）
        # ============================================================
        # 注意：jnp.min/max/mean 会触发 GPU→CPU 同步
        # 适合接口验证，但会污染性能测量
        theta_before = state.theta
        print(f"  State 信息:")
        print(f"    theta 形状: {theta_before.shape}")
        print(f"    theta 设备: {theta_before.devices()}")
        print(f"    theta 范围: [{float(jnp.min(theta_before)):.2f}, "
              f"{float(jnp.max(theta_before)):.2f}] K  [同步点]")
        print(f"    u 形状: {state.u.shape}")
        print(f"    qv 形状: {state.qv.shape}")

        # ============================================================
        # 2. 准备修改后的 state（或测试接口）
        # ============================================================
        if self.dry_run:
            # DRY 模式：验证 replace 接口，但使用原始 state
            print(f"  [DRY] 验证 state.replace 接口（no-op）...")
            # 测试 replace 方法存在且不崩溃
            _ = state.replace(theta=theta_before)
            print(f"  [DRY] ✓ replace 接口正常")
            state_for_forecast = state  # 使用原始 state
        else:
            # WET 模式：真实修改
            print(f"  [WET] 修改 theta +{self.perturbation_k}K...")
            theta_modified = theta_before + self.perturbation_k

            # 关键：replace 后必须 dealias（避免 donate 崩溃）
            # 原因：State.replace() 可能产生 buffer 别名，
            #       run_forecast_operational 使用 donate_argnums=(0,)
            #       会尝试 donate 同一 buffer 两次导致崩溃
            from gpuwrf.runtime.operational_mode import dealias_state_buffers
            state_modified = dealias_state_buffers(
                state.replace(theta=theta_modified)
            )

            # 验证修改（会触发同步，仅验证用）
            actual_delta = float(jnp.mean(state_modified.theta - theta_before))
            print(f"  [WET] ✓ theta 修改成功，实际扰动: {actual_delta:.6f} K  [同步点]")
            state_for_forecast = state_modified

        # ============================================================
        # 3. 记录历史（不触发同步）
        # ============================================================
        self.history.append({
            'call_count': self.call_count,
            'hours_before': self.total_hours,
            'hours_advance': hours,
            'mode': 'DRY' if self.dry_run else 'WET',
        })

        # ============================================================
        # 4. 调用真实预报（复用 _default_forecast_fn）
        # ============================================================
        print(f"  调用真实预报（分段编译模式）...")

        # 使用 _segmented_forecast_fn 而非 _default_forecast_fn
        # 关键差异：
        #   - _default_forecast_fn → run_forecast_operational → 单片段编译 O(N) 内存
        #   - _segmented_forecast_fn → run_forecast_operational_segmented → 分段编译 O(1) 内存
        #
        # 内存对比（瑞士 d01 实测）：
        #   - 单片段 1h: 编译 5 分钟, 峰值 ~40GB 内存
        #   - 分段 1h: 编译 2-3 分钟, 峰值 ~15GB 内存
        #
        # 编译稳定性：
        #   - 单片段：hours 是 static_argnames，每次 hours 变化重新编译
        #             + State 结构变化（边界更新）也重新编译
        #   - 分段：固定段长度 1h，编译缓存稳定
        from gpuwrf.integration.daily_pipeline import _segmented_forecast_fn

        result = _segmented_forecast_fn(state_for_forecast, namelist, hours)

        print(f"  ✓ 预报完成")
        print(f"{'='*70}\n")

        # ============================================================
        # 5. 更新累积时间并返回
        # ============================================================
        self.total_hours += hours
        return result  # 必须返回预报结果（不能返回未推进的 state）

    def summary(self):
        """打印运行摘要"""
        print(f"\n{'='*70}")
        print(f"耦合原型运行摘要")
        print(f"{'='*70}")
        print(f"模式: {'DRY (验证模式)' if self.dry_run else 'WET (耦合模式)'}")
        print(f"总调用次数: {self.call_count}")
        print(f"累积时间: {self.total_hours:.3f} 小时")
        if not self.dry_run:
            print(f"theta 扰动: {self.perturbation_k} K × {self.call_count} 次")
            print(f"预期累积扰动: ~{self.perturbation_k * self.call_count:.1f} K")
        print(f"{'='*70}\n")


def main():
    """命令行入口"""
    # =====================================================
    # 解析命令行参数
    # =====================================================
    parser = argparse.ArgumentParser(
        description="forecast_fn 耦合原型测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # DRY 模式（2h 快速验证）
  python coupling_prototype.py --dry

  # WET 模式（3h 耦合测试）
  python coupling_prototype.py --wet

  # 自定义参数
  python coupling_prototype.py --wet --hours 2 --perturbation-k 0.5
        """
    )

    # 模式选择（互斥）
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--dry', action='store_true',
                           help='DRY 模式：验证接口（也会真跑 forecast）')
    mode_group.add_argument('--wet', action='store_true',
                           help='WET 模式：完整耦合测试')

    # 预报参数
    parser.add_argument('--hours', type=int, default=None,
                       help='预报小时数（默认：DRY=2, WET=3）')
    parser.add_argument('--perturbation-k', type=float, default=1.0,
                       help='WET 模式 theta 扰动（开尔文，默认 1.0）')
    parser.add_argument('--domain', type=str, default='d01',
                       help='域名（默认 d01）')

    # 路径参数（写死瑞士案例）
    repo_root = Path(__file__).parent
    parser.add_argument('--input-dir', type=Path,
                       default=repo_root / 'examples' / 'switzerland_d01',
                       help='输入目录（默认 examples/switzerland_d01）')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='输出目录（默认：DRY=runs/coupling_dry, WET=runs/coupling_wet）')

    args = parser.parse_args()

    # =====================================================
    # 参数验证和默认值
    # =====================================================
    dry_run = args.dry

    # 默认 hours：DRY=2, WET=3
    if args.hours is None:
        hours = 2 if dry_run else 3
    else:
        hours = args.hours

    # 默认 output_dir：DRY=runs/coupling_dry, WET=runs/coupling_wet
    if args.output_dir is None:
        output_dir = repo_root / 'runs' / ('coupling_dry' if dry_run else 'coupling_wet')
    else:
        output_dir = args.output_dir

    input_dir = args.input_dir.resolve()
    output_dir = output_dir.resolve()

    # 检查输入目录
    if not input_dir.is_dir():
        print(f"❌ 错误：输入目录不存在: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # 清理输出目录（避免 WRFOUT_TARGET_EXISTS 错误）
    if output_dir.exists():
        import shutil
        print(f"清理现有输出目录: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 导入 gpuwrf（延迟导入，避免无 JAX 环境报错）
    # =====================================================
    try:
        from gpuwrf.integration.daily_pipeline import (
            execute_daily_pipeline,
            DailyPipelineConfig,
        )
    except ImportError as e:
        print(f"❌ 错误：无法导入 gpuwrf: {e}", file=sys.stderr)
        print("请确保已安装 gpuwrf 并激活虚拟环境", file=sys.stderr)
        sys.exit(1)

    # =====================================================
    # 创建耦合实例
    # =====================================================
    coupled = CoupledForecastPrototype(
        perturbation_k=args.perturbation_k,
        dry_run=dry_run,
    )

    # =====================================================
    # 配置管线
    # =====================================================
    config = DailyPipelineConfig(
        run_id=str(input_dir),      # 绝对路径
        run_root=input_dir.parent,  # fallback
        domain=args.domain,
        hours=hours,
        output_dir=output_dir,
        proof_dir=output_dir / "proofs",
    )

    # =====================================================
    # 打印运行信息
    # =====================================================
    print(f"\n{'='*70}")
    print(f"启动耦合测试")
    print(f"{'='*70}")
    print(f"模式: {'DRY (验证接口)' if dry_run else 'WET (完整耦合)'}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"域: {args.domain}")
    print(f"预报小时数: {hours}")
    if not dry_run:
        print(f"Theta 扰动: {args.perturbation_k} K")
    print(f"预计耗时: ~{hours * 3:.0f}-{hours * 6:.0f} 分钟 (分段编译模式)")
    print(f"内存优化: 使用分段编译，峰值 ~15-20GB（vs 单片段 40GB+）")
    print(f"{'='*70}\n")

    # =====================================================
    # 执行
    # =====================================================
    try:
        result = execute_daily_pipeline(
            config,
            forecast_fn=coupled,  # ← 关键：传入自定义 forecast_fn
        )

        # =====================================================
        # 打印结果
        # =====================================================
        coupled.summary()

        print(f"\n{'='*70}")
        print(f"管线执行结果")
        print(f"{'='*70}")
        print(f"Verdict: {result.get('verdict', 'N/A')}")
        wall_s = result.get('wall_clock_total_s', 0)
        print(f"Wall clock: {wall_s:.1f} s ({wall_s/60:.1f} min)")
        print(f"Output files: {len(result.get('wrfout_files', []))}")
        if result.get('wrfout_files'):
            print(f"First output: {result['wrfout_files'][0]}")
            print(f"Last output: {result['wrfout_files'][-1]}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*70}\n")

        # =====================================================
        # 7. 严格验证检查
        # =====================================================
        # 检查 verdict（必须是 PIPELINE_GREEN）
        verdict = result.get('verdict', 'UNKNOWN')
        if verdict != 'PIPELINE_GREEN':
            print(f"\n❌ 验证失败：verdict = {verdict}")
            print(f"原因: {result.get('reason', 'N/A')}")
            if 'detail' in result:
                print(f"详情: {result['detail']}")
            sys.exit(1)

        # 检查 wrfout 文件
        wrfout_files = result.get('wrfout_files', [])
        if not wrfout_files:
            print(f"\n❌ 验证失败：未生成 wrfout 文件")
            sys.exit(1)

        # 检查 forecast_fn 调用次数
        if coupled.call_count == 0:
            print(f"\n❌ 验证失败：forecast_fn 未被调用")
            sys.exit(1)

        # 检查累积时间
        if abs(coupled.total_hours - hours) > 0.01:
            print(f"\n❌ 验证失败：累积时间不匹配 ({coupled.total_hours} != {hours})")
            sys.exit(1)

        # 所有检查通过
        if dry_run:
            print("\n" + "="*70)
            print("✓ DRY 验证通过！")
            print("="*70)
            print("✓ verdict = PIPELINE_GREEN")
            print(f"✓ wrfout 文件生成: {len(wrfout_files)} 个")
            print("✓ forecast_fn 被正确调用")
            print("✓ state 读取正常")
            print("✓ replace 接口正常")
            print("✓ 分段编译模式工作正常")
            print("\n下一步：运行 WET 模式")
            print(f"  python {Path(__file__).name} --wet")
        else:
            print("\n" + "="*70)
            print("✓ WET 耦合完成！")
            print("="*70)
            print("✓ verdict = PIPELINE_GREEN")
            print(f"✓ wrfout 文件生成: {len(wrfout_files)} 个")
            print("✓ 完整预报 + theta 修改成功")
            print(f"✓ theta 累积扰动: ~{coupled.call_count * coupled.perturbation_k:.1f}K")
            print(f"\n对比验证:")
            print(f"  1. 运行基准版本（无耦合）")
            print(f"  2. 对比 T 场差异（预期 coupled 比 baseline 暖 ~{coupled.call_count}K）")

        sys.exit(0)

    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        coupled.summary()
        sys.exit(1)


if __name__ == "__main__":
    main()
