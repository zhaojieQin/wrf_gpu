#!/usr/bin/env python3
"""
LBM 侧平流计算指南

WRF 传送过来的是 staggered grid（交错网格）数据：
- u: (nz, ny, nx+1) - 定义在 x 方向面上
- v: (nz, ny+1, nx) - 定义在 y 方向面上
- w: (nz+1, ny, nx) - 定义在 z 方向面上
- theta: (nz, ny, nx) - 定义在质量点上（mass point）
- t_skin: (ny, nx) - 2D 地表场
- roughness_m: (ny, nx) - 2D 地表场

本指南展示如何：
1. 将 staggered 风场插值到质量点
2. 使用中心差分计算梯度
3. 计算动量平流：-u·∇u, -v·∇v, -w·∇w
4. 计算热量平流：-u·∇θ, -v·∇θ, -w·∇θ

物理意义：
- 动量平流：流体运动导致的动量变化（惯性项）
- 热量平流：流体运动导致的温度变化（水平/垂直输送）
"""

import numpy as np


# ============================================================
# 1. Staggered Grid 插值到质量点
# ============================================================

def unstagger_u(u_staggered):
    """u 从 x-面插值到质量点

    Args:
        u_staggered: (nz, ny, nx+1) - WRF 的 u 风场（staggered）

    Returns:
        u_mass: (nz, ny, nx) - 插值到质量点
    """
    return 0.5 * (u_staggered[:, :, :-1] + u_staggered[:, :, 1:])


def unstagger_v(v_staggered):
    """v 从 y-面插值到质量点

    Args:
        v_staggered: (nz, ny+1, nx) - WRF 的 v 风场（staggered）

    Returns:
        v_mass: (nz, ny, nx) - 插值到质量点
    """
    return 0.5 * (v_staggered[:, :-1, :] + v_staggered[:, 1:, :])


def unstagger_w(w_staggered):
    """w 从 z-面插值到质量点

    Args:
        w_staggered: (nz+1, ny, nx) - WRF 的 w 风场（staggered）

    Returns:
        w_mass: (nz, ny, nx) - 插值到质量点
    """
    return 0.5 * (w_staggered[:-1, :, :] + w_staggered[1:, :, :])


# ============================================================
# 2. 梯度计算（中心差分）
# ============================================================

def gradient_x(field, dx):
    """x 方向梯度（中心差分）

    Args:
        field: (nz, ny, nx) - 标量场（定义在质量点）
        dx: float - x 方向网格间距（米）

    Returns:
        dfield_dx: (nz, ny, nx) - ∂field/∂x
    """
    dfield_dx = np.zeros_like(field)
    dfield_dx[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2 * dx)
    dfield_dx[:, :, 0] = (field[:, :, 1] - field[:, :, 0]) / dx
    dfield_dx[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / dx
    return dfield_dx


def gradient_y(field, dy):
    """y 方向梯度（中心差分）

    Args:
        field: (nz, ny, nx) - 标量场（定义在质量点）
        dy: float - y 方向网格间距（米）

    Returns:
        dfield_dy: (nz, ny, nx) - ∂field/∂y
    """
    dfield_dy = np.zeros_like(field)
    dfield_dy[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2 * dy)
    dfield_dy[:, 0, :] = (field[:, 1, :] - field[:, 0, :]) / dy
    dfield_dy[:, -1, :] = (field[:, -1, :] - field[:, -2, :]) / dy
    return dfield_dy


def gradient_z(field, dz):
    """z 方向梯度（中心差分）

    Args:
        field: (nz, ny, nx) - 标量场（定义在质量点）
        dz: float 或 (nz,) - z 方向网格间距（米）

    Returns:
        dfield_dz: (nz, ny, nx) - ∂field/∂z
    """
    dfield_dz = np.zeros_like(field)
    if np.isscalar(dz):
        dfield_dz[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2 * dz)
        dfield_dz[0, :, :] = (field[1, :, :] - field[0, :, :]) / dz
        dfield_dz[-1, :, :] = (field[-1, :, :] - field[-2, :, :]) / dz
    else:
        for k in range(1, field.shape[0] - 1):
            dz_up = dz[k]
            dz_down = dz[k-1]
            dfield_dz[k, :, :] = (
                (field[k+1, :, :] - field[k, :, :]) / dz_up -
                (field[k, :, :] - field[k-1, :, :]) / dz_down
            ) / (0.5 * (dz_up + dz_down))
        dfield_dz[0, :, :] = (field[1, :, :] - field[0, :, :]) / dz[0]
        dfield_dz[-1, :, :] = (field[-1, :, :] - field[-2, :, :]) / dz[-2]
    return dfield_dz


# ============================================================
# 3. 平流计算
# ============================================================

def compute_momentum_advection(u_stag, v_stag, w_stag, dx, dy, dz):
    """计算动量平流：-u·∇u, -v·∇v, -w·∇w

    Args:
        u_stag: (nz, ny, nx+1) - WRF u 风场（staggered）
        v_stag: (nz, ny+1, nx) - WRF v 风场（staggered）
        w_stag: (nz+1, ny, nx) - WRF w 风场（staggered）
        dx: float - x 方向网格间距（米）
        dy: float - y 方向网格间距（米）
        dz: float 或 (nz,) - z 方向网格间距（米）

    Returns:
        adv_u: (nz, ny, nx) - u 动量平流
        adv_v: (nz, ny, nx) - v 动量平流
        adv_w: (nz, ny, nx) - w 动量平流
    """
    u = unstagger_u(u_stag)
    v = unstagger_v(v_stag)
    w = unstagger_w(w_stag)

    du_dx = gradient_x(u, dx)
    du_dy = gradient_y(u, dy)
    du_dz = gradient_z(u, dz)

    dv_dx = gradient_x(v, dx)
    dv_dy = gradient_y(v, dy)
    dv_dz = gradient_z(v, dz)

    dw_dx = gradient_x(w, dx)
    dw_dy = gradient_y(w, dy)
    dw_dz = gradient_z(w, dz)

    adv_u = -(u * du_dx + v * du_dy + w * du_dz)
    adv_v = -(u * dv_dx + v * dv_dy + w * dv_dz)
    adv_w = -(u * dw_dx + v * dw_dy + w * dw_dz)

    return adv_u, adv_v, adv_w


def compute_heat_advection(u_stag, v_stag, w_stag, theta, dx, dy, dz):
    """计算热量平流：-u·∇θ

    Args:
        u_stag: (nz, ny, nx+1) - WRF u 风场（staggered）
        v_stag: (nz, ny+1, nx) - WRF v 风场（staggered）
        w_stag: (nz+1, ny, nx) - WRF w 风场（staggered）
        theta: (nz, ny, nx) - 位温场
        dx: float - x 方向网格间距（米）
        dy: float - y 方向网格间距（米）
        dz: float 或 (nz,) - z 方向网格间距（米）

    Returns:
        adv_theta: (nz, ny, nx) - 热量平流
    """
    u = unstagger_u(u_stag)
    v = unstagger_v(v_stag)
    w = unstagger_w(w_stag)

    dtheta_dx = gradient_x(theta, dx)
    dtheta_dy = gradient_y(theta, dy)
    dtheta_dz = gradient_z(theta, dz)

    adv_theta = -(u * dtheta_dx + v * dtheta_dy + w * dtheta_dz)
    return adv_theta


# ============================================================
# 4. 完整示例：从 WRF 数据计算平流
# ============================================================

def example_compute_advection_from_wrf_subdomain(subdomain_data, grid_spacing):
    """完整示例：从 WRF 子域数据计算平流

    Args:
        subdomain_data: dict - WRF 传送的子域数据
        grid_spacing: dict - 网格间距信息

    Returns:
        advection_fields: dict - 计算出的平流场
    """
    u_stag = subdomain_data['u']
    v_stag = subdomain_data['v']
    w_stag = subdomain_data['w']
    theta = subdomain_data['theta']

    dx = grid_spacing['dx']
    dy = grid_spacing['dy']
    dz = grid_spacing['dz']

    adv_u, adv_v, adv_w = compute_momentum_advection(
        u_stag, v_stag, w_stag, dx, dy, dz
    )

    adv_theta = compute_heat_advection(
        u_stag, v_stag, w_stag, theta, dx, dy, dz
    )

    u_mass = unstagger_u(u_stag)
    v_mass = unstagger_v(v_stag)
    w_mass = unstagger_w(w_stag)

    return {
        'adv_u': adv_u,
        'adv_v': adv_v,
        'adv_w': adv_w,
        'adv_theta': adv_theta,
        'u_mass': u_mass,
        'v_mass': v_mass,
        'w_mass': w_mass,
    }


# ============================================================
# 5. 使用示例
# ============================================================

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("使用示例")
    print("="*70)

    nz, ny, nx = 30, 100, 100

    subdomain_data = {
        'u': np.random.randn(nz, ny, nx+1),
        'v': np.random.randn(nz, ny+1, nx),
        'w': np.random.randn(nz+1, ny, nx),
        'theta': 300.0 + np.random.randn(nz, ny, nx),
        't_skin': 280.0 + np.random.randn(ny, nx),
        'roughness_m': 0.1 * np.ones((ny, nx)),
    }

    grid_spacing = {
        'dx': 3000.0,
        'dy': 3000.0,
        'dz': 100.0,
    }

    print(f"\n1. 输入数据形状：")
    for key, val in subdomain_data.items():
        print(f"   {key}: {val.shape}")

    print(f"\n2. 网格间距：")
    for key, val in grid_spacing.items():
        print(f"   {key}: {val}")

    print(f"\n3. 计算平流场...")
    advection = example_compute_advection_from_wrf_subdomain(
        subdomain_data, grid_spacing
    )

    print(f"\n4. 输出平流场：")
    for key, val in advection.items():
        print(f"   {key}: shape={val.shape}, "
              f"range=[{val.min():.2e}, {val.max():.2e}]")

    print(f"\n✓ 完成！LBM 端可以直接使用这些平流场")
    print(f"\n注意事项：")
    print(f"  1. 如果 WRF 使用地形跟随坐标，dz 应该是 (nz,) 数组")
    print(f"  2. 边界处理可能需要根据 LBM 边界条件调整")
    print(f"  3. 如果 LBM 需要通量形式平流，需要乘以密度")
    print(f"  4. 这里使用笛卡尔网格假设，经纬度网格需要地图因子修正")
