#!/usr/bin/env python3
"""
耦合设置静态自检脚本

验证：
1. coupling_alignment 模块导入和计算正确
2. AuxhistStreamConfig 配置合法
3. 对齐参数符合预期
4. 不运行 pipeline，纯静态检查

用法：
    python check_coupling_setup.py
    python check_coupling_setup.py --dt-wrf 10.0 --dt-lbm 0.05
"""

import argparse
import sys
from pathlib import Path


def check_imports():
    """检查必要模块导入"""
    print("\n[1/4] 检查模块导入...")

    try:
        from coupling_alignment import (
            compute_coupling_alignment,
            print_coupling_alignment_report,
        )
        print("  ✓ coupling_alignment 模块导入成功")
    except ImportError as e:
        print(f"  ✗ 无法导入 coupling_alignment: {e}", file=sys.stderr)
        return False

    # gpuwrf 导入（仅检查配置类，不触发 JAX）
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from gpuwrf.integration.daily_pipeline import DailyPipelineConfig
        from gpuwrf.io.auxhist_stream import AuxhistStreamConfig
        print("  ✓ gpuwrf 配置类导入成功")
    except ImportError as e:
        print(f"  ✗ 无法导入 gpuwrf 配置: {e}", file=sys.stderr)
        return False

    return True


def check_alignment_computation(dt_wrf, dt_lbm, desired_interval_s):
    """检查对齐参数计算"""
    print(f"\n[2/4] 计算耦合对齐参数...")
    print(f"  输入: dt_wrf={dt_wrf}s, dt_lbm={dt_lbm}s, desired={desired_interval_s}s")

    from coupling_alignment import (
        compute_coupling_alignment,
        print_coupling_alignment_report,
    )

    try:
        alignment = compute_coupling_alignment(
            dt_wrf=dt_wrf,
            dt_lbm=dt_lbm,
            desired_interval_s=desired_interval_s,
        )
        print("  ✓ 对齐参数计算成功")
        return alignment
    except ValueError as e:
        print(f"  ✗ 对齐参数计算失败: {e}", file=sys.stderr)
        return None


def check_auxhist_config(alignment):
    """检查 auxhist 配置合法性"""
    print(f"\n[3/4] 验证 AuxhistStreamConfig 配置...")

    from gpuwrf.io.auxhist_stream import AuxhistStreamConfig

    try:
        config = AuxhistStreamConfig(
            stream_id=1,
            interval_minutes=alignment['auxhist_interval_minutes'],
            variables=("T2",),  # 最小合法变量集
        )
        print(f"  ✓ AuxhistStreamConfig 配置合法")
        print(f"    stream_id: {config.stream_id}")
        print(f"    interval_minutes: {config.interval_minutes}")
        print(f"    variables: {config.variables}")
        return config
    except Exception as e:
        print(f"  ✗ AuxhistStreamConfig 配置失败: {e}", file=sys.stderr)
        return None


def check_alignment_validity(alignment, dt_wrf):
    """检查对齐参数有效性"""
    print(f"\n[4/4] 验证对齐参数有效性...")

    # 验证整数步
    expected_wrf_time = alignment['n_wrf_steps'] * dt_wrf
    actual_coupling_time = alignment['actual_coupling_s']

    if abs(expected_wrf_time - actual_coupling_time) > 1e-6:
        print(f"  ✗ WRF 步数不对齐: {alignment['n_wrf_steps']} × {dt_wrf}s "
              f"= {expected_wrf_time}s ≠ {actual_coupling_time}s", file=sys.stderr)
        return False

    print(f"  ✓ WRF 步数对齐: {alignment['n_wrf_steps']} 步 × {dt_wrf}s = {actual_coupling_time}s")

    # 验证 substeps
    expected_substeps = 60 // alignment['auxhist_interval_minutes']
    if alignment['substeps'] != expected_substeps:
        print(f"  ✗ substeps 计算错误: 预期 {expected_substeps}, 实际 {alignment['substeps']}",
              file=sys.stderr)
        return False

    print(f"  ✓ substeps 正确: 60 / {alignment['auxhist_interval_minutes']} = {alignment['substeps']}")

    # 验证 60 的因数
    factors_of_60 = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
    if alignment['auxhist_interval_minutes'] not in factors_of_60:
        print(f"  ✗ auxhist_interval_minutes={alignment['auxhist_interval_minutes']} "
              f"不是 60 的因数", file=sys.stderr)
        return False

    print(f"  ✓ auxhist_interval_minutes={alignment['auxhist_interval_minutes']} 是 60 的因数")

    return True


def main():
    parser = argparse.ArgumentParser(description="耦合设置静态自检")
    parser.add_argument('--dt-wrf', type=float, default=18.0,
                       help='WRF 时间步长（秒，默认 18.0）')
    parser.add_argument('--dt-lbm', type=float, default=0.1,
                       help='LBM 时间步长（秒，默认 0.1）')
    parser.add_argument('--desired-interval', type=float, default=60.0,
                       help='期望耦合间隔（秒，默认 60.0）')

    args = parser.parse_args()

    print("="*70)
    print("WRF-LBM 耦合设置静态自检")
    print("="*70)

    # 1. 检查导入
    if not check_imports():
        print("\n❌ 导入检查失败")
        sys.exit(1)

    # 2. 计算对齐参数
    alignment = check_alignment_computation(
        args.dt_wrf,
        args.dt_lbm,
        args.desired_interval,
    )
    if alignment is None:
        print("\n❌ 对齐参数计算失败")
        sys.exit(1)

    # 3. 检查 auxhist 配置
    auxhist_config = check_auxhist_config(alignment)
    if auxhist_config is None:
        print("\n❌ auxhist 配置检查失败")
        sys.exit(1)

    # 4. 验证对齐有效性
    if not check_alignment_validity(alignment, args.dt_wrf):
        print("\n❌ 对齐参数验证失败")
        sys.exit(1)

    # 打印完整报告
    print("\n" + "="*70)
    print("对齐参数详细报告")
    print("="*70)

    from coupling_alignment import print_coupling_alignment_report
    print_coupling_alignment_report(alignment, args.dt_wrf, args.dt_lbm, args.desired_interval)

    # 成功
    print("="*70)
    print("✓ 所有检查通过！耦合设置配置正确")
    print("="*70)
    print()
    print("下一步：运行耦合原型")
    print(f"  python coupling_prototype_3min.py --dry --hours 1")
    print()

    sys.exit(0)


if __name__ == "__main__":
    main()
