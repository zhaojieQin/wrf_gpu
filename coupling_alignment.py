"""WRF-LBM 耦合时间对齐推导

自动计算满足三重约束的耦合参数：
1. WRF 整数步：N_wrf × dt_wrf = actual_coupling_s
2. LBM 整数步：M_lbm × dt_lbm = actual_coupling_s
3. auxhist 约束：actual_coupling_minutes ∈ {60 的因数}
"""

import math
from typing import TypedDict


class CouplingAlignment(TypedDict):
    """耦合时间对齐参数"""
    actual_coupling_s: float              # 实际耦合间隔（秒）
    actual_coupling_minutes: float        # 实际耦合间隔（分钟）
    n_wrf_steps: int                      # WRF 每次耦合的步数
    m_lbm_steps: int                      # LBM 每次耦合的步数
    deviation_s: float                    # 偏离期望值（秒）
    deviation_pct: float                  # 偏离百分比
    auxhist_interval_minutes: int         # auxhist 配置值
    substeps: int                         # 预期 substeps（每小时段数）


def compute_coupling_alignment(
    dt_wrf: float,
    dt_lbm: float,
    desired_interval_s: float,
    *,
    max_deviation_pct: float = 300.0,
) -> CouplingAlignment:
    """计算 WRF-LBM 耦合的时间对齐参数

    Args:
        dt_wrf: WRF 时间步长（秒）
        dt_lbm: LBM 时间步长（秒）
        desired_interval_s: 期望耦合间隔（秒）
        max_deviation_pct: 最大允许偏离百分比（默认 50%，因为受 auxhist 限制）

    Returns:
        CouplingAlignment: 对齐参数字典

    Raises:
        ValueError: 无法找到满足约束的对齐方案

    约束：
        1. actual_coupling_minutes 必须是 60 的因数（auxhist gcd 限制）
        2. actual_coupling_s 必须被 dt_wrf 整除（WRF 整数步）
        3. actual_coupling_s 必须被 dt_lbm 整除（LBM 整数步）

    示例：
        >>> # dt_wrf=18s 期望 60s → 实际 180s（3 分钟）
        >>> alignment = compute_coupling_alignment(18.0, 0.1, 60.0)
        >>> alignment['actual_coupling_minutes']
        3.0
        >>> alignment['n_wrf_steps']
        10
    """

    # 输入验证
    if dt_wrf <= 0 or dt_lbm <= 0:
        raise ValueError(f"Time steps must be positive: dt_wrf={dt_wrf}, dt_lbm={dt_lbm}")
    if desired_interval_s <= 0:
        raise ValueError(f"Desired interval must be positive: {desired_interval_s}s")
    if dt_wrf > desired_interval_s:
        raise ValueError(
            f"WRF time step ({dt_wrf}s) cannot exceed coupling interval ({desired_interval_s}s)"
        )

    # 60 的因数（允许的 segment_minutes）
    factors_of_60 = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]

    # 转换为整数毫秒避免浮点误差
    dt_wrf_ms = int(round(dt_wrf * 1000))
    dt_lbm_ms = int(round(dt_lbm * 1000))

    # 找到满足所有约束的最接近因数
    desired_minutes = desired_interval_s / 60.0

    candidates = []
    for factor in factors_of_60:
        coupling_s = factor * 60.0
        coupling_ms = factor * 60 * 1000

        # 检查是否能被 dt_wrf 整除
        if coupling_ms % dt_wrf_ms != 0:
            continue

        # 检查是否能被 dt_lbm 整除（或向上取整）
        n_wrf = coupling_ms // dt_wrf_ms
        m_lbm = (coupling_ms + dt_lbm_ms - 1) // dt_lbm_ms  # 向上取整

        # 验证时间对齐（容忍 1ms 误差）
        wrf_time_ms = n_wrf * dt_wrf_ms
        lbm_time_ms = m_lbm * dt_lbm_ms
        if abs(wrf_time_ms - lbm_time_ms) > 1:
            continue

        # 计算偏离
        deviation = abs(coupling_s - desired_interval_s)
        candidates.append((factor, coupling_s, n_wrf, m_lbm, deviation))

    if not candidates:
        # 列出所有 60 因数对应的秒数，看哪些能被 dt_wrf 整除
        feasible = []
        for f in factors_of_60:
            s = f * 60
            if (s * 1000) % dt_wrf_ms == 0:
                feasible.append(f"{f} min ({s}s)")

        if feasible:
            msg = (
                f"Cannot find coupling interval close to {desired_interval_s}s.\n"
                f"With dt_wrf={dt_wrf}s, feasible intervals are: {', '.join(feasible)}.\n"
                f"Deviation from desired {desired_interval_s}s exceeds {max_deviation_pct}%."
            )
        else:
            msg = (
                f"dt_wrf={dt_wrf}s cannot align with any 60-factor interval.\n"
                f"60-factor intervals: {[f*60 for f in factors_of_60]}s.\n"
                f"Consider adjusting dt_wrf to a value that divides these intervals."
            )
        raise ValueError(msg)

    # 选择偏离最小的候选
    best = min(candidates, key=lambda x: x[4])
    factor, coupling_s, n_wrf, m_lbm, deviation = best

    # 检查偏离是否可接受
    deviation_pct = 100 * deviation / desired_interval_s
    if deviation_pct > max_deviation_pct:
        raise ValueError(
            f"Best alignment deviates {deviation_pct:.1f}% from desired {desired_interval_s}s.\n"
            f"Best: {coupling_s}s ({factor} min). Max allowed deviation: {max_deviation_pct}%.\n"
            f"Consider increasing max_deviation_pct or adjusting desired_interval_s."
        )

    actual_coupling_minutes = float(factor)
    deviation_s = coupling_s - desired_interval_s

    return {
        'actual_coupling_s': coupling_s,
        'actual_coupling_minutes': actual_coupling_minutes,
        'n_wrf_steps': n_wrf,
        'm_lbm_steps': m_lbm,
        'deviation_s': deviation_s,
        'deviation_pct': 100 * deviation_s / desired_interval_s,
        'auxhist_interval_minutes': int(factor),
        'substeps': 60 // int(factor),
    }


def print_coupling_alignment_report(
    alignment: CouplingAlignment,
    dt_wrf: float,
    dt_lbm: float,
    desired_interval_s: float,
):
    """打印耦合对齐参数的人类可读报告"""

    print("=" * 70)
    print("WRF-LBM 耦合时间对齐参数")
    print("=" * 70)
    print()

    print("输入参数:")
    print(f"  WRF 时间步:     dt_wrf = {dt_wrf} s")
    print(f"  LBM 时间步:     dt_lbm = {dt_lbm} s")
    print(f"  期望耦合间隔:   {desired_interval_s} s ({desired_interval_s/60:.2f} 分钟)")
    print()

    print("对齐结果:")
    print(f"  实际耦合间隔:   {alignment['actual_coupling_s']} s "
          f"({alignment['actual_coupling_minutes']:.0f} 分钟)")
    print(f"  偏离期望值:     {alignment['deviation_s']:+.1f} s "
          f"({alignment['deviation_pct']:+.1f}%)")
    print()

    print("各侧步数:")
    print(f"  WRF 推进:       {alignment['n_wrf_steps']} 步 "
          f"× {dt_wrf}s = {alignment['actual_coupling_s']}s")
    print(f"  LBM 推进:       {alignment['m_lbm_steps']} 步 "
          f"× {dt_lbm}s ≈ {alignment['actual_coupling_s']}s")
    print()

    print("WRF 侧配置:")
    print(f"  auxhist interval:  {alignment['auxhist_interval_minutes']} 分钟")
    print(f"  预期 substeps:     {alignment['substeps']} 段/小时")
    print(f"  forecast_fn 调用:  每 {alignment['actual_coupling_minutes']:.0f} 分钟一次")
    print()

    print("LBM 侧配置:")
    print(f"  每次耦合运行:      {alignment['m_lbm_steps']} 步")
    print(f"  耦合区间时长:      {alignment['actual_coupling_s']} s")
    print()

    print("=" * 70)
