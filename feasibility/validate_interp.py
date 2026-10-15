"""feasibility/validate_interp.py

验证插值结果：用合成 WRF 场对 5 个边界面节点插值，检查平滑性、越界、NaN。
输出文本统计，可选输出图像（需要 matplotlib）。

运行方法（在 wrf_gpu/ 目录下）：
    python -I feasibility/validate_interp.py [--plot]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_synthetic_wrf(nz=40, ny=14, nx=20, dx_wrf=900.0, dy_wrf=900.0):
    """Synthetic WRF fields: smooth log-law wind + linear theta."""
    from lbm_adapter.vertical_interp import WRFColumnData

    G = 9.81
    dz = 15000.0 / nz

    # z_w (nz+1, ny, nx): uniform vertical grid
    k = np.arange(nz + 1)
    z_w = k[:, None, None] * dz * np.ones((nz + 1, ny, nx))
    ph_total = (z_w * G).astype(np.float32)

    z_mass = 0.5 * (z_w[:-1] + z_w[1:])

    u_star, kappa, z0 = 0.4, 0.4, 0.05
    u = (u_star / kappa * np.log(np.maximum(z_mass, z0) / z0)).astype(np.float32)
    v = np.zeros_like(u)
    w = np.zeros((nz + 1, ny, nx), dtype=np.float32)
    theta = (300.0 + 0.005 * z_mass).astype(np.float32)

    terrain = np.zeros((ny, nx), dtype=np.float64)

    return WRFColumnData(
        ph_total=ph_total.astype(np.float64),
        u=u.astype(np.float64),
        v=v.astype(np.float64),
        w=w.astype(np.float64),
        theta=theta.astype(np.float64),
        terrain_height=terrain,
        dx_wrf=dx_wrf,
        dy_wrf=dy_wrf,
    )


def make_boundary_coords(H, Lx, Ly, dx):
    """Same as precompute_bc_weights.py."""
    xs = np.arange(dx / 2, Lx, dx)
    ys = np.arange(dx / 2, Ly, dx)
    zs = np.arange(dx / 2, H, dx)
    faces = {}

    yy, zz = np.meshgrid(ys, zs, indexing='ij')
    faces['MX'] = np.column_stack([np.zeros(yy.size), yy.ravel(), zz.ravel()])
    faces['PX'] = np.column_stack([np.full(yy.size, Lx), yy.ravel(), zz.ravel()])

    xx, zz2 = np.meshgrid(xs, zs, indexing='ij')
    faces['MY'] = np.column_stack([xx.ravel(), np.zeros(xx.size), zz2.ravel()])
    faces['PY'] = np.column_stack([xx.ravel(), np.full(xx.size, Ly), zz2.ravel()])

    xx2, yy2 = np.meshgrid(xs, ys, indexing='ij')
    faces['PZ'] = np.column_stack([xx2.ravel(), yy2.ravel(), np.full(xx2.size, H)])
    return faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    from lbm_adapter.wrf_to_lbm import WRFStaticContext, interp_to_bc_nodes
    from lbm_adapter.vertical_interp import WRFColumnData

    H, Lx, Ly = 1000.0, 6000.0, 4000.0
    dx = 1000.0 / 64
    dx_wrf, dy_wrf = 900.0, 900.0
    nx_wrf, ny_wrf = 20, 14

    wrf_data = make_synthetic_wrf(ny=ny_wrf, nx=nx_wrf, dx_wrf=dx_wrf, dy_wrf=dy_wrf)
    static_ctx = WRFStaticContext(
        terrain_height=wrf_data.terrain_height,
        dx_wrf=dx_wrf, dy_wrf=dy_wrf
    )

    wrf_Lx = (nx_wrf - 1) * dx_wrf
    wrf_Ly = (ny_wrf - 1) * dy_wrf
    lbm_origin_x = (wrf_Lx - Lx) / 2
    lbm_origin_y = (wrf_Ly - Ly) / 2

    face_coords = make_boundary_coords(H, Lx, Ly, dx)

    print(f"{'Face':4s}  {'N':>7s}  {'NaN':>5s}  {'u_min':>8s}  {'u_max':>8s}  "
          f"{'theta_min':>10s}  {'theta_max':>10s}  {'smooth?':>8s}")
    print("-" * 72)

    plots_data = {}

    for face, lcoords in face_coords.items():
        # add WRF offset
        wrf_coords = lcoords.copy()
        wrf_coords[:, 0] += lbm_origin_x
        wrf_coords[:, 1] += lbm_origin_y

        fields = interp_to_bc_nodes(wrf_data, wrf_coords, z_coord_is_agl=True)

        u = fields.u
        v = fields.v
        w = fields.w
        theta = fields.theta

        n_nan = int(np.isnan(u).sum() + np.isnan(theta).sum())

        # Smoothness: max first-diff / mean
        if len(u) > 1:
            du = np.abs(np.diff(np.sort(u)))
            smooth = "ok" if du.max() < 5.0 else "WARN"
        else:
            smooth = "n/a"

        print(f"{face:4s}  {len(u):>7d}  {n_nan:>5d}  "
              f"{u.min():>8.3f}  {u.max():>8.3f}  "
              f"{theta.min():>10.3f}  {theta.max():>10.3f}  {smooth:>8s}")

        plots_data[face] = {'u': u, 'theta': theta, 'z': lcoords[:, 2]}

    print()
    print("All fields finite:", all(
        np.all(np.isfinite(d['u'])) and np.all(np.isfinite(d['theta']))
        for d in plots_data.values()
    ))
    print("u range sanity (0-20 m/s):", all(
        d['u'].min() >= 0 and d['u'].max() <= 25
        for d in plots_data.values()
    ))
    print("theta range sanity (295-320 K):", all(
        d['theta'].min() > 295 and d['theta'].max() < 320
        for d in plots_data.values()
    ))

    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 5, figsize=(16, 4))
            for ax, (face, d) in zip(axes, plots_data.items()):
                ax.scatter(d['u'], d['z'], s=1, alpha=0.3)
                ax.set_title(face)
                ax.set_xlabel('u (m/s)')
                ax.set_ylabel('z_agl (m)')
            plt.tight_layout()
            out = Path('feasibility/validate_interp.png')
            plt.savefig(out, dpi=100)
            print(f"\nPlot saved: {out}")
        except ImportError:
            print("matplotlib not available, skip plot.")


if __name__ == '__main__':
    main()
