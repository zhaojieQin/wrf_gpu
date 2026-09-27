#!/usr/bin/env python3
"""
可视化最后三步 wrfout_d03_SS 文件

用法：
    python plot_wrfout_last3.py --input-dir ./received_nc_swift
    python plot_wrfout_last3.py --input-dir ./received_nc_swift --level 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def find_wrfout_files(input_dir: Path, n: int = 3):
    """找到最后 n 个 wrfout_d03_SS 文件，按步数升序排列"""
    files = sorted(input_dir.glob('wrfout_d03_*'))
    if not files:
        print(f'[错误] 在 {input_dir} 下未找到 wrfout_d03_* 文件')
        sys.exit(1)
    return files[-n:]


def load_file(fpath: Path):
    """读取 NetCDF 文件，返回字段字典"""
    try:
        from netCDF4 import Dataset
    except ImportError:
        print('[错误] 需要安装 netCDF4: pip install netCDF4')
        sys.exit(1)

    fields = {}
    with Dataset(str(fpath), 'r') as ds:
        step = getattr(ds, 'coupling_step', '?')
        for vname in ds.variables:
            arr = ds.variables[vname][:]
            fields[vname] = np.asarray(arr)
    return step, fields


def plot_horizontal_slice(ax, data, title, level, cmap='RdBu_r', units=''):
    """绘制某层的水平切片"""
    if data.ndim == 3:
        lev = min(level, data.shape[0] - 1)
        sl = data[lev, :, :]
        title = f'{title} (z={lev})'
    else:
        sl = data

    vmax = np.nanpercentile(np.abs(sl), 99)
    if vmax == 0:
        vmax = 1.0

    # 对风场用发散色标，其余用线性
    if cmap == 'RdBu_r':
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(sl, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    else:
        im = ax.imshow(sl, origin='lower', cmap=cmap, aspect='auto')

    plt.colorbar(im, ax=ax, shrink=0.8, label=units)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('west_east')
    ax.set_ylabel('south_north')


def plot_vertical_slice(ax, data, title, col, units=''):
    """绘制某列的垂直剖面（z-y 截面）"""
    if data.ndim != 3:
        ax.set_visible(False)
        return
    c = min(col, data.shape[2] - 1)
    sl = data[:, :, c]  # shape=(nz, ny)

    im = ax.imshow(sl, origin='lower', aspect='auto',
                   cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8, label=units)
    ax.set_title(f'{title} vert (x={c})', fontsize=9)
    ax.set_xlabel('south_north')
    ax.set_ylabel('bottom_top')


def main():
    parser = argparse.ArgumentParser(description='可视化最后3步 wrfout_d03_SS')
    parser.add_argument('--input-dir', type=str, default='./received_nc_swift',
                        help='wrfout 文件所在目录')
    parser.add_argument('--level', type=int, default=20,
                        help='水平切片的垂直层索引（默认 20）')
    parser.add_argument('--n', type=int, default=3,
                        help='显示最后 N 步（默认 3）')
    parser.add_argument('--output', type=str, default='wrfout_last3.png',
                        help='输出图片文件名')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = find_wrfout_files(input_dir, args.n)
    print(f'找到 {len(files)} 个文件：')
    for f in files:
        print(f'  {f.name}')

    # 变量配置：(WRF变量名, colormap, units, label)
    VAR_CFG = [
        ('U',      'RdBu_r', 'm/s',   'U风'),
        ('V',      'RdBu_r', 'm/s',   'V风'),
        ('T',      'plasma', 'K',     '扰动位温T\''),
        ('QVAPOR', 'Blues',  'kg/kg', '水汽混合比'),
    ]

    n_steps = len(files)
    n_vars = len(VAR_CFG)
    # 每步：水平切片 + 垂直剖面 各一行，共 n_vars 列
    # 布局：行 = n_steps * 2，列 = n_vars
    fig, axes = plt.subplots(
        n_steps * 2, n_vars,
        figsize=(n_vars * 4, n_steps * 2 * 3),
        squeeze=False
    )
    fig.suptitle(f'WRF coupling 最后 {n_steps} 步输出（z={args.level}层）', fontsize=12)

    for i, fpath in enumerate(files):
        step, fields = load_file(fpath)
        row_h = i * 2      # 水平切片行
        row_v = i * 2 + 1  # 垂直剖面行

        for j, (vname, cmap, units, label) in enumerate(VAR_CFG):
            if vname not in fields:
                axes[row_h][j].set_title(f'{vname} 缺失', fontsize=9)
                axes[row_v][j].set_visible(False)
                continue

            data = fields[vname]
            col_mid = data.shape[2] // 2 if data.ndim == 3 else 0

            plot_horizontal_slice(
                axes[row_h][j], data,
                f'Step {step} {label}',
                args.level, cmap=cmap, units=units
            )
            plot_vertical_slice(
                axes[row_v][j], data,
                f'Step {step} {label}',
                col_mid, units=units
            )

    plt.tight_layout()
    out_path = Path(args.output)
    plt.savefig(str(out_path), dpi=120, bbox_inches='tight')
    print(f'\n图片已保存：{out_path.resolve()}')


if __name__ == '__main__':
    main()
