"""feasibility/flux_and_flow_direction.py

分析 3 和 4 节：
- 各面法向速度 u_n 的分布（入流/出流节点统计）
- |u_n| 小于各阈值的节点比例
- 五面净通量估算（质量守恒检查）
- 最大 Ma 估算

使用合成 WRF 场（log-law 风廓线，主风向沿 +x）。

运行：
    python -I feasibility/flux_and_flow_direction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_synthetic_wrf(nz=40, ny=14, nx=20, dx_wrf=900.0, dy_wrf=900.0,
                       u_star=0.4, kappa=0.4, z0=0.05,
                       v_background=1.5):
    """Log-law u + small crosswind v."""
    from lbm_adapter.vertical_interp import WRFColumnData

    G = 9.81
    dz = 10000.0 / nz
    k = np.arange(nz + 1)
    z_w = k[:, None, None] * dz * np.ones((nz + 1, ny, nx))
    ph_total = (z_w * G).astype(np.float32)
    z_mass = 0.5 * (z_w[:-1] + z_w[1:])

    u = (u_star / kappa * np.log(np.maximum(z_mass, z0) / z0)).astype(np.float32)
    v = np.full_like(u, v_background, dtype=np.float32)
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
    xs = np.arange(dx / 2, Lx, dx)
    ys = np.arange(dx / 2, Ly, dx)
    zs = np.arange(dx / 2, H, dx)
    faces = {}
    # outward normals per face
    normals = {
        'MX': np.array([-1, 0, 0]),
        'PX': np.array([+1, 0, 0]),
        'MY': np.array([0, -1, 0]),
        'PY': np.array([0, +1, 0]),
        'PZ': np.array([0, 0, +1]),
    }

    yy, zz = np.meshgrid(ys, zs, indexing='ij')
    faces['MX'] = np.column_stack([np.zeros(yy.size), yy.ravel(), zz.ravel()])
    faces['PX'] = np.column_stack([np.full(yy.size, Lx), yy.ravel(), zz.ravel()])

    xx, zz2 = np.meshgrid(xs, zs, indexing='ij')
    faces['MY'] = np.column_stack([xx.ravel(), np.zeros(xx.size), zz2.ravel()])
    faces['PY'] = np.column_stack([xx.ravel(), np.full(xx.size, Ly), zz2.ravel()])

    xx2, yy2 = np.meshgrid(xs, ys, indexing='ij')
    faces['PZ'] = np.column_stack([xx2.ravel(), yy2.ravel(), np.full(xx2.size, H)])
    return faces, normals


def main():
    from lbm_adapter.wrf_to_lbm import WRFStaticContext, interp_to_bc_nodes

    H, Lx, Ly = 1000.0, 6000.0, 4000.0
    dx = 1000.0 / 64
    dx_wrf, dy_wrf = 900.0, 900.0
    nx_wrf, ny_wrf = 20, 14

    wrf_data = make_synthetic_wrf(ny=ny_wrf, nx=nx_wrf, dx_wrf=dx_wrf, dy_wrf=dy_wrf)
    wrf_Lx = (nx_wrf - 1) * dx_wrf
    wrf_Ly = (ny_wrf - 1) * dy_wrf
    lbm_origin_x = (wrf_Lx - Lx) / 2
    lbm_origin_y = (wrf_Ly - Ly) / 2

    face_coords_local, normals = make_boundary_coords(H, Lx, Ly, dx)

    static_ctx = WRFStaticContext(
        terrain_height=wrf_data.terrain_height,
        dx_wrf=dx_wrf, dy_wrf=dy_wrf,
    )

    # LBM params for Ma check
    u_star, kappa, z0 = 0.4, 0.4, 0.05
    mach = 0.1
    u_ref_phys = 0.5 * u_star / kappa * np.log(H / z0)
    velocity_ratio = u_ref_phys * np.sqrt(3) / mach  # = u_ref / Ma * sqrt(3)

    print(f"LBM velocity_ratio ≈ {velocity_ratio:.2f} m/s")
    print(f"LBM u_ref_phys = {u_ref_phys:.2f} m/s  Ma = {mach:.3f}")
    print()

    thresholds = [0.1, 0.5, 1.0]  # m/s
    face_fluxes = {}

    print(f"{'Face':4s}  {'N':>7s}  {'in%':>6s}  {'out%':>6s}  "
          + "  ".join(f"|u|<{t:.1f}%" for t in thresholds)
          + "  {'flux_in':>10s}  {'flux_out':>10s}  {'net':>10s}")
    print("-" * 100)

    total_inflow = 0.0
    total_flux_net = 0.0

    for face, lcoords in face_coords_local.items():
        wrf_coords = lcoords.copy()
        wrf_coords[:, 0] += lbm_origin_x
        wrf_coords[:, 1] += lbm_origin_y

        fields = interp_to_bc_nodes(wrf_data, wrf_coords, z_coord_is_agl=True)
        uvw = np.column_stack([fields.u, fields.v, fields.w])  # (N, 3)

        n_out = normals[face]  # (3,)
        u_n = uvw @ n_out  # (N,) normal component (positive = outflow)

        n_in  = (u_n < 0).sum()
        n_out_flow = (u_n >= 0).sum()
        n = len(u_n)

        # Cell area (dx^2 per node)
        A_cell = dx * dx
        flux_in  = float((-u_n[u_n < 0]).sum() * A_cell)   # m³/s, positive
        flux_out = float(u_n[u_n >= 0].sum() * A_cell)     # m³/s, positive
        face_fluxes[face] = (flux_in, flux_out)
        total_inflow += flux_in
        total_flux_net += (flux_out - flux_in)

        thresh_str = "  ".join(
            f"{100*(np.abs(u_n) < t).mean():6.1f}%" for t in thresholds
        )

        # Max Ma on this face
        u_mag_max = np.abs(u_n).max()
        ma_max = u_mag_max / velocity_ratio * np.sqrt(3)  # cs = u_ref/sqrt(3)

        print(f"{face:4s}  {n:>7d}  "
              f"{100*n_in/n:6.1f}%  {100*n_out_flow/n:6.1f}%  "
              f"{thresh_str}  "
              f"{flux_in:>10.1f}  {flux_out:>10.1f}  "
              f"{flux_out-flux_in:>+10.1f}  Ma_max={ma_max:.4f}")

    print()
    net_pct = abs(total_flux_net) / max(total_inflow, 1e-9) * 100
    print(f"Total inflow:  {total_inflow:.1f} m³/s")
    print(f"Net flux imbalance: {total_flux_net:+.1f} m³/s  "
          f"= {net_pct:.2f}% of total inflow")

    print()
    print("--- Edge/corner nodes (belong to 2+ faces) ---")
    # Count nodes on edges: simple estimate using face sizes
    # MX/PX share edges with MY, PY, PZ; MX/PX also share corner with MY/PY/PZ
    nz_lbm = int(H / dx)
    ny_lbm = int(Ly / dx)
    nx_lbm = int(Lx / dx)

    # edges on MX face
    edge_MX_MY = nz_lbm   # bottom MX-MY edge (z varies)
    edge_MX_PY = nz_lbm
    edge_MX_PZ = ny_lbm
    n_edge_on_MX = edge_MX_MY + edge_MX_PY + edge_MX_PZ
    print(f"  MX face: ~{n_edge_on_MX} edge nodes "
          f"= {100*n_edge_on_MX/(ny_lbm*nz_lbm):.1f}% of face")
    print(f"  Corner nodes (MX-MY-PZ etc): 8 corners total in full box, 4 on open faces")
    print()
    print("Priority rule recommendation: when a node is on edge of two open faces,")
    print("  use the face whose |u_n| is larger (dominant inflow wins).")
    print("  For corner nodes (3 faces): same rule, check all 3 normal components.")


if __name__ == '__main__':
    main()
