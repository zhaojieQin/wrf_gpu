"""feasibility/precompute_bc_weights.py

原型脚本：对 LBM 的 5 个开边界面节点，预计算水平双线性插值的索引和权重，
以及垂直插值的索引（最近层），将结果序列化到 feasibility/bc_weights.npz。

不修改任何主代码。

运行方法（在 wrf_gpu/ 目录下）：
    python -I feasibility/precompute_bc_weights.py \
        --wrfinput /path/to/wrfinput_d03 \
        --lbm-H 1000 --lbm-nz 64 --lbm-Lx 6000 --lbm-Ly 4000 \
        [--out feasibility/bc_weights.npz]

如果没有真实 wrfinput_d03，脚本会用合成数据自动运行（--synthetic）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 几何辅助：生成5面边界节点坐标（LBM局部坐标系，从0,0,0起）
# ---------------------------------------------------------------------------

def make_boundary_coords(H, Lx, Ly, dx):
    """Generate node coordinates on all 5 open faces (excluding MZ=ground).

    Returns a dict face -> (N, 3) array in LBM local coords (x, y, z in m).
    Faces: MX, PX, MY, PY, PZ.
    """
    coords = {}
    xs = np.arange(dx / 2, Lx, dx)
    ys = np.arange(dx / 2, Ly, dx)
    zs = np.arange(dx / 2, H, dx)

    # MX face: x=0, y varies, z varies
    yy, zz = np.meshgrid(ys, zs, indexing='ij')
    coords['MX'] = np.column_stack([np.zeros_like(yy.ravel()), yy.ravel(), zz.ravel()])

    # PX face: x=Lx
    coords['PX'] = np.column_stack([np.full(yy.size, Lx), yy.ravel(), zz.ravel()])

    # MY face: y=0, x varies, z varies
    xx, zz2 = np.meshgrid(xs, zs, indexing='ij')
    coords['MY'] = np.column_stack([xx.ravel(), np.zeros_like(xx.ravel()), zz2.ravel()])

    # PY face: y=Ly
    coords['PY'] = np.column_stack([xx.ravel(), np.full(xx.size, Ly), zz2.ravel()])

    # PZ face: z=H (top), x varies, y varies
    xx2, yy2 = np.meshgrid(xs, ys, indexing='ij')
    coords['PZ'] = np.column_stack([xx2.ravel(), yy2.ravel(), np.full(xx2.size, H)])

    return coords


# ---------------------------------------------------------------------------
# 双线性权重预计算
# ---------------------------------------------------------------------------

def precompute_bilinear_weights(coords_xy, dx_wrf, dy_wrf, nx_wrf, ny_wrf):
    """For each node (x, y) compute bilinear stencil in WRF grid.

    Returns dict with keys: i0, j0, wx, wy (each shape (N,), int or float).
    """
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]

    xi = x / dx_wrf
    yi = y / dy_wrf

    i0 = np.clip(np.floor(xi).astype(int), 0, nx_wrf - 2)
    j0 = np.clip(np.floor(yi).astype(int), 0, ny_wrf - 2)
    wx = np.clip(xi - i0, 0.0, 1.0)
    wy = np.clip(yi - j0, 0.0, 1.0)

    return dict(i0=i0, j0=j0, wx=wx, wy=wy)


# ---------------------------------------------------------------------------
# 垂直层索引预计算（给定典型 z_mass 列，返回每个节点的最近层索引）
# ---------------------------------------------------------------------------

def precompute_vertical_index(z_nodes_msl, z_mass_col):
    """Find nearest z_mass layer index for each node (scalar column approximation)."""
    idx = np.searchsorted(z_mass_col, z_nodes_msl, side='left')
    idx = np.clip(idx, 0, len(z_mass_col) - 1)
    return idx


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Precompute LBM boundary interpolation weights')
    parser.add_argument('--wrfinput', type=str, default=None)
    parser.add_argument('--lbm-H',   type=float, default=1000.0, help='BL height (m)')
    parser.add_argument('--lbm-nz',  type=int,   default=64,     help='Nodes in z')
    parser.add_argument('--lbm-Lx',  type=float, default=6000.0, help='Domain Lx (m)')
    parser.add_argument('--lbm-Ly',  type=float, default=4000.0, help='Domain Ly (m)')
    parser.add_argument('--lbm-origin-x', type=float, default=None)
    parser.add_argument('--lbm-origin-y', type=float, default=None)
    parser.add_argument('--synthetic', action='store_true', default=False)
    parser.add_argument('--out', type=str, default='feasibility/bc_weights.npz')
    args = parser.parse_args()

    H  = args.lbm_H
    Lx = args.lbm_Lx
    Ly = args.lbm_Ly
    dx = H / args.lbm_nz

    # --- WRF grid metadata ---
    if args.wrfinput and not args.synthetic:
        import netCDF4 as nc
        with nc.Dataset(args.wrfinput, 'r') as ds:
            terrain = np.asarray(ds.variables['HGT'][0], dtype=np.float64)
            dx_wrf = float(ds.DX)
            dy_wrf = float(ds.DY)
            ny_wrf, nx_wrf = terrain.shape
        print(f"WRF d03: {ny_wrf}x{nx_wrf}, dx={dx_wrf}m, dy={dy_wrf}m")
    else:
        # Synthetic: 3km WRF grid over 18x12 km domain
        dx_wrf, dy_wrf = 900.0, 900.0
        nx_wrf, ny_wrf = 20, 14
        terrain = 50.0 * np.random.rand(ny_wrf, nx_wrf)
        if not args.synthetic:
            print("No --wrfinput given; using synthetic WRF grid.", file=sys.stderr)

    wrf_Lx = (nx_wrf - 1) * dx_wrf
    wrf_Ly = (ny_wrf - 1) * dy_wrf

    lbm_origin_x = args.lbm_origin_x if args.lbm_origin_x is not None else (wrf_Lx - Lx) / 2
    lbm_origin_y = args.lbm_origin_y if args.lbm_origin_y is not None else (wrf_Ly - Ly) / 2

    print(f"LBM domain: Lx={Lx}m Ly={Ly}m H={H}m dx={dx}m")
    print(f"LBM origin in WRF: x={lbm_origin_x:.0f}m y={lbm_origin_y:.0f}m")

    # --- Generate LBM face node coords (local) ---
    face_coords_local = make_boundary_coords(H, Lx, Ly, dx)

    results = {}
    total_nodes = 0
    memory_bytes = 0

    # Approximate typical z_mass column (flat terrain at 0m, nz=40 WRF layers)
    nz_wrf = 40
    dz_wrf_approx = 15000.0 / nz_wrf
    z_mass_col = np.arange(0.5, nz_wrf) * dz_wrf_approx

    for face, lcoords in face_coords_local.items():
        n = lcoords.shape[0]
        total_nodes += n

        # Convert LBM local coords → WRF coords
        wrf_coords = lcoords.copy()
        wrf_coords[:, 0] += lbm_origin_x
        wrf_coords[:, 1] += lbm_origin_y
        # z stays AGL; convert to MSL using nearest terrain
        j_idx = np.clip(np.round(wrf_coords[:, 1] / dy_wrf).astype(int), 0, ny_wrf - 1)
        i_idx = np.clip(np.round(wrf_coords[:, 0] / dx_wrf).astype(int), 0, nx_wrf - 1)
        terrain_at_node = terrain[j_idx, i_idx]
        z_msl = lcoords[:, 2] + terrain_at_node

        w = precompute_bilinear_weights(wrf_coords, dx_wrf, dy_wrf, nx_wrf, ny_wrf)
        v_idx = precompute_vertical_index(z_msl, z_mass_col)

        results[face] = {
            'n': n,
            'i0': w['i0'], 'j0': w['j0'],
            'wx': w['wx'], 'wy': w['wy'],
            'v_idx': v_idx,
            'z_agl': lcoords[:, 2],
            'z_msl': z_msl,
        }

        # Memory estimate: per node we store i0(4B) j0(4B) wx(8B) wy(8B) v_idx(4B) = 28 bytes
        mem_face = n * 28
        memory_bytes += mem_face

        print(f"  {face}: {n} nodes  mem={mem_face/1024:.1f} kB  "
              f"z_agl=[{lcoords[:,2].min():.0f},{lcoords[:,2].max():.0f}]m")

    print(f"\nTotal nodes (5 faces): {total_nodes}")
    print(f"Total precomputed weight memory: {memory_bytes/1024:.1f} kB  "
          f"({memory_bytes/1024**2:.3f} MB)")
    print("(Runtime cost per WRF step: bilinear lookup = O(n) numpy ops, negligible vs MPI recv)")

    # --- Save to npz ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for face, d in results.items():
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                save_dict[f"{face}_{k}"] = v
            else:
                save_dict[f"{face}_{k}"] = np.array(v)
    np.savez(out_path, **save_dict)
    print(f"\nWeights saved to {out_path}")

    return results


if __name__ == '__main__':
    main()
