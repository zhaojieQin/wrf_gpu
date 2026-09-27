#!/usr/bin/env python3
"""
可视化最后3步 MPI coupling 输出 (wrfout_d03_SS)

用法:
    python plot_coupling_last3.py
    python plot_coupling_last3.py --input-dir ./received_nc_swift --level 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

T00 = 300.0  # WRF base state temperature


def find_wrfout_files(input_dir: Path, n: int = 3):
    files = sorted(input_dir.glob('wrfout_d03_*'),
                   key=lambda p: int(p.name.split('_')[-1]) if p.name.split('_')[-1].isdigit() else 0)
    if not files:
        print(f'[错误] 在 {input_dir} 下未找到 wrfout_d03_* 文件')
        sys.exit(1)
    return files[-n:]


def load_file(fpath: Path):
    """读取 NetCDF，返回 (step, fields) 字典"""
    try:
        from netCDF4 import Dataset
    except ImportError:
        print('[错误] 需要安装 netCDF4: pip install netCDF4')
        sys.exit(1)

    fields = {}
    with Dataset(str(fpath), 'r') as ds:
        step = int(getattr(ds, 'coupling_step', -1))
        for vname in ds.variables:
            fields[vname] = np.asarray(ds.variables[vname][:])

    # 后处理：T -> theta (加回 T00)
    if 'T' in fields:
        fields['theta'] = fields.pop('T') + T00

    # destagger U (west_east_stag -> west_east)
    if 'U' in fields:
        u = fields.pop('U')
        fields['u'] = 0.5 * (u[:, :, :-1] + u[:, :, 1:])

    # destagger V (south_north_stag -> south_north)
    if 'V' in fields:
        v = fields.pop('V')
        fields['v'] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])

    if 'QVAPOR' in fields:
        fields['qv'] = fields.pop('QVAPOR') * 1000.0  # kg/kg -> g/kg

    return step, fields


def plot_horizontal(ax, data, title, level, cmap, units):
    lev = min(level, data.shape[0] - 1)
    sl = data[lev]
    vmax = np.nanpercentile(np.abs(sl[np.isfinite(sl)]), 99)
    if vmax == 0:
        vmax = 1.0
    if cmap == 'RdBu_r':
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(sl, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    else:
        im = ax.imshow(sl, origin='lower', cmap=cmap, vmin=sl.min(), vmax=sl.max(), aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.85, label=units)
    ax.set_title(f'{title}\n(z={lev})', fontsize=9)
    ax.set_xlabel('west_east', fontsize=8)
    ax.set_ylabel('south_north', fontsize=8)


def plot_vertical(ax, data, title, units):
    col = data.shape[2] // 2
    sl = data[:, :, col]  # (nz, ny)
    im = ax.imshow(sl, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, shrink=0.85, label=units)
    ax.set_title(f'{title}\nvert cross (x={col})', fontsize=9)
    ax.set_xlabel('south_north', fontsize=8)
    ax.set_ylabel('bottom_top', fontsize=8)


def plot_vertical_profile(ax, data, label, color):
    cy, cx = data.shape[1] // 2, data.shape[2] // 2
    profile = data[:, cy, cx]
    levels = np.arange(len(profile))
    ax.plot(profile, levels, color=color, marker='o', markersize=3, label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='./received_nc_swift')
    parser.add_argument('--level', type=int, default=20,
                        help='水平切片垂直层索引')
    parser.add_argument('--n', type=int, default=3,
                        help='显示最后 N 步')
    parser.add_argument('--output-dir', default=None,
                        help='图片输出目录（默认与 input-dir 同目录下的 plots/）')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    plot_dir = Path(args.output_dir) if args.output_dir else input_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    files = find_wrfout_files(input_dir, args.n)
    print(f'找到 {len(files)} 个文件:')
    for f in files:
        print(f'  {f.name}')
    print(f'图片输出目录: {plot_dir}')

    # 加载所有步
    steps_data = []
    for fpath in files:
        step, fields = load_file(fpath)
        steps_data.append((step, fpath.name, fields))
        print(f'  加载 {fpath.name}: step={step}, fields={list(fields.keys())}')

    VAR_CFG = [
        ('theta', 'RdYlBu_r', 'K',     'Theta (K)'),
        ('u',     'RdBu_r',   'm/s',   'U (m/s)'),
        ('v',     'RdBu_r',   'm/s',   'V (m/s)'),
        ('qv',    'YlGnBu',   'g/kg',  'Qv (g/kg)'),
    ]

    # ── 图1: 水平切片（每步一行，每变量一列）──────────────────────────
    print('\n生成图1: 水平切片...')
    n_steps = len(steps_data)
    n_vars = len(VAR_CFG)
    fig1, axes1 = plt.subplots(n_steps, n_vars,
                                figsize=(n_vars * 4, n_steps * 3.5),
                                squeeze=False)
    fig1.suptitle(f'Coupling 最后 {n_steps} 步 — 水平切片 (z={args.level})', fontsize=13)

    for i, (step, fname, fields) in enumerate(steps_data):
        for j, (vname, cmap, units, label) in enumerate(VAR_CFG):
            ax = axes1[i][j]
            if vname not in fields:
                ax.set_title(f'Step {step}\n{vname} 缺失', fontsize=9)
                ax.axis('off')
                continue
            plot_horizontal(ax, fields[vname],
                            f'Step {step} — {label}', args.level, cmap, units)

    plt.tight_layout()
    out1 = plot_dir / 'coupling_horizontal.png'
    plt.savefig(str(out1), dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  保存: {out1}')

    # ── 图2: 垂直剖面（每步一行，每变量一列）─────────────────────────
    print('生成图2: 垂直剖面...')
    fig2, axes2 = plt.subplots(n_steps, n_vars,
                                figsize=(n_vars * 4, n_steps * 3.5),
                                squeeze=False)
    fig2.suptitle(f'Coupling 最后 {n_steps} 步 — 垂直剖面 (中间列)', fontsize=13)

    for i, (step, fname, fields) in enumerate(steps_data):
        for j, (vname, cmap, units, label) in enumerate(VAR_CFG):
            ax = axes2[i][j]
            if vname not in fields:
                ax.set_title(f'Step {step}\n{vname} 缺失', fontsize=9)
                ax.axis('off')
                continue
            plot_vertical(ax, fields[vname], f'Step {step} — {label}', units)

    plt.tight_layout()
    out2 = plot_dir / 'coupling_vertical.png'
    plt.savefig(str(out2), dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  保存: {out2}')

    # ── 图3: 垂直廓线对比（3步叠加在同一图）──────────────────────────
    print('生成图3: 垂直廓线对比...')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    fig3, axes3 = plt.subplots(1, n_vars, figsize=(n_vars * 4, 6), squeeze=False)
    fig3.suptitle(f'Coupling 最后 {n_steps} 步 — 中心点垂直廓线对比', fontsize=13)

    for j, (vname, cmap, units, label) in enumerate(VAR_CFG):
        ax = axes3[0][j]
        for i, (step, fname, fields) in enumerate(steps_data):
            if vname not in fields:
                continue
            plot_vertical_profile(ax, fields[vname],
                                  label=f'Step {step}', color=colors[i % len(colors)])
        ax.set_xlabel(f'{label} ({units})', fontsize=9)
        ax.set_ylabel('bottom_top', fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out3 = plot_dir / 'coupling_profiles.png'
    plt.savefig(str(out3), dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  保存: {out3}')

    print(f'\n完成! 共生成 3 张图:')
    print(f'  {out1}')
    print(f'  {out2}')
    print(f'  {out3}')


if __name__ == '__main__':
    main()
