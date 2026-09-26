#!/usr/bin/env python3
"""Visualize SWiFT nested WRF output files."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from pathlib import Path

# Configuration
output_dir = Path('/home/zhaqi746/code/wrf_gpu/runs/swift_coupled_dry')
plot_dir = output_dir / 'plots'
plot_dir.mkdir(exist_ok=True)

wrfout_files = {
    'd01': output_dir / 'wrfout_d01_2013-11-07_19:00:00',
    'd02': output_dir / 'wrfout_d02_2013-11-07_19:00:00',
    'd03': output_dir / 'wrfout_d03_2013-11-07_19:00:00',
}

T00 = 300.0  # Base state temperature for WRF perturbation temperature


def load_wrfout(file_path):
    """Load WRF output file and extract key variables."""
    print(f"Loading {file_path.name}...")
    nc = Dataset(file_path, 'r')

    data = {
        'time': nc.variables['Times'][0].tobytes().decode('utf-8').strip(),
        'xlat': nc.variables['XLAT'][0, :, :],
        'xlong': nc.variables['XLONG'][0, :, :],
    }

    # 2D surface fields
    if 'T2' in nc.variables:
        data['t2'] = nc.variables['T2'][0, :, :] - 273.15  # Convert to Celsius

    # 3D atmospheric fields
    if 'T' in nc.variables:
        # T is perturbation potential temperature, add T00 to get full theta
        data['theta'] = nc.variables['T'][0, :, :, :] + T00

    if 'U' in nc.variables:
        # U is on staggered grid, destagger by averaging
        u_stag = nc.variables['U'][0, :, :, :]
        data['u'] = 0.5 * (u_stag[:, :, :-1] + u_stag[:, :, 1:])

    if 'V' in nc.variables:
        # V is on staggered grid, destagger by averaging
        v_stag = nc.variables['V'][0, :, :, :]
        data['v'] = 0.5 * (v_stag[:, :-1, :] + v_stag[:, 1:, :])

    if 'QVAPOR' in nc.variables:
        data['qv'] = nc.variables['QVAPOR'][0, :, :, :] * 1000.0  # Convert to g/kg

    # Get dimensions
    data['nz'], data['ny'], data['nx'] = data['theta'].shape

    nc.close()
    print(f"  Shape: nz={data['nz']}, ny={data['ny']}, nx={data['nx']}")
    print(f"  Time: {data['time']}")

    return data


def plot_surface_fields():
    """Plot surface temperature for all three domains."""
    print("\nPlotting surface fields...")

    for domain_name, file_path in wrfout_files.items():
        if not file_path.exists():
            print(f"  Warning: {file_path} not found, skipping")
            continue

        data = load_wrfout(file_path)

        fig, ax = plt.subplots(figsize=(10, 8))

        if 't2' in data:
            im = ax.imshow(data['t2'], cmap='RdYlBu_r', origin='lower')
            plt.colorbar(im, ax=ax, label='Temperature (°C)')
            ax.set_title(f'{domain_name.upper()} Surface Temperature (T2)\n{data["time"]}')
        else:
            # Fallback to lowest theta level
            theta_sfc = data['theta'][0, :, :]
            im = ax.imshow(theta_sfc, cmap='RdYlBu_r', origin='lower')
            plt.colorbar(im, ax=ax, label='Potential Temperature (K)')
            ax.set_title(f'{domain_name.upper()} Surface Theta (lowest level)\n{data["time"]}')

        ax.set_xlabel('X index')
        ax.set_ylabel('Y index')

        out_path = plot_dir / f'swift_{domain_name}_surface.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")


def plot_vertical_profiles():
    """Plot vertical profiles of theta, u, v, qv at domain centers."""
    print("\nPlotting vertical profiles...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for domain_name, file_path in wrfout_files.items():
        if not file_path.exists():
            continue

        data = load_wrfout(file_path)

        # Extract center point
        cy, cx = data['ny'] // 2, data['nx'] // 2
        levels = np.arange(data['nz'])

        # Theta profile
        theta_profile = data['theta'][:, cy, cx]
        axes[0].plot(theta_profile, levels, marker='o', label=domain_name.upper())

        # U profile
        if 'u' in data:
            u_profile = data['u'][:, cy, cx]
            axes[1].plot(u_profile, levels, marker='o', label=domain_name.upper())

        # V profile
        if 'v' in data:
            v_profile = data['v'][:, cy, cx]
            axes[2].plot(v_profile, levels, marker='o', label=domain_name.upper())

        # Qv profile
        if 'qv' in data:
            qv_profile = data['qv'][:, cy, cx]
            axes[3].plot(qv_profile, levels, marker='o', label=domain_name.upper())

    axes[0].set_xlabel('Potential Temperature (K)')
    axes[0].set_ylabel('Model Level')
    axes[0].set_title('Theta Profile at Domain Center')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('U Wind (m/s)')
    axes[1].set_ylabel('Model Level')
    axes[1].set_title('U Profile at Domain Center')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('V Wind (m/s)')
    axes[2].set_ylabel('Model Level')
    axes[2].set_title('V Profile at Domain Center')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].set_xlabel('Water Vapor (g/kg)')
    axes[3].set_ylabel('Model Level')
    axes[3].set_title('Qv Profile at Domain Center')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = plot_dir / 'swift_profiles.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_horizontal_slices():
    """Plot horizontal slices of theta, u, v, qv at a specific level."""
    print("\nPlotting horizontal slices...")

    # Choose level (e.g., level 10 or closest to ~500m above ground)
    level_idx = 10

    fields = ['theta', 'u', 'v', 'qv']
    field_labels = {
        'theta': 'Potential Temperature (K)',
        'u': 'U Wind (m/s)',
        'v': 'V Wind (m/s)',
        'qv': 'Water Vapor (g/kg)',
    }
    field_cmaps = {
        'theta': 'RdYlBu_r',
        'u': 'RdBu_r',
        'v': 'RdBu_r',
        'qv': 'YlGnBu',
    }

    for field in fields:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (domain_name, file_path) in enumerate(wrfout_files.items()):
            if not file_path.exists():
                continue

            data = load_wrfout(file_path)

            if field not in data:
                continue

            # Extract horizontal slice
            k = min(level_idx, data['nz'] - 1)
            field_data = data[field][k, :, :]

            im = axes[idx].imshow(field_data, cmap=field_cmaps[field], origin='lower')
            axes[idx].set_title(f'{domain_name.upper()} (level {k})')
            axes[idx].set_xlabel('X index')
            axes[idx].set_ylabel('Y index')
            plt.colorbar(im, ax=axes[idx], label=field_labels[field])

        fig.suptitle(f'{field_labels[field]} - Horizontal Slice\n{data["time"]}', fontsize=14)
        plt.tight_layout()

        out_path = plot_dir / f'swift_horizontal_{field}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")


def main():
    """Main plotting routine."""
    print("=" * 60)
    print("SWiFT WRF Output Visualization")
    print("=" * 60)

    # Check if files exist
    for domain_name, file_path in wrfout_files.items():
        if file_path.exists():
            print(f"✓ Found: {file_path.name}")
        else:
            print(f"✗ Missing: {file_path.name}")

    print(f"\nPlots will be saved to: {plot_dir}")

    # Generate plots
    plot_surface_fields()
    plot_vertical_profiles()
    plot_horizontal_slices()

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
