#!/usr/bin/env python3
"""
Iteratively refine i_parent_start/j_parent_start to center d03 on SWiFT.

Uses the functions from compute_nest_params.py but applies iterative
correction to minimize center position error.
"""

import numpy as np
from math import radians, cos, sin, sqrt

# ============================================================
# Configuration (same as compute_nest_params.py)
# ============================================================

TARGET_LAT = 33.6
TARGET_LON = -102.1

D01_DX = 27000.0
D01_E_WE = 118
D01_E_SN = 100
D01_REF_LAT = 32.67
D01_REF_LON = -102.05

D02_DX = 9000.0
D02_E_WE = 154
D02_E_SN = 133
D02_PARENT_RATIO = 3

D03_DX = 3000.0
D03_E_WE = 118
D03_E_SN = 100
D03_PARENT_RATIO = 3

# Convergence threshold (degrees)
CONVERGENCE_TOL = 0.01
MAX_ITERATIONS = 10

# ============================================================
# Helper Functions (copied from compute_nest_params.py)
# ============================================================

def latlon_to_meters(lat1, lon1, lat2, lon2):
    """Convert lat/lon offset to approximate meters."""
    R_EARTH = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * np.arctan2(sqrt(a), sqrt(1-a))
    distance = R_EARTH * c

    lat_avg = radians((lat1 + lat2) / 2)
    dy = dlat * R_EARTH
    dx = dlon * R_EARTH * cos(lat_avg)

    return dx, dy


def compute_parent_start(target_lat, target_lon,
                         parent_ref_lat, parent_ref_lon, parent_dx, parent_e_we, parent_e_sn,
                         child_e_we, child_e_sn, parent_ratio):
    """Compute i_parent_start and j_parent_start for a nested domain."""
    dx_m, dy_m = latlon_to_meters(parent_ref_lat, parent_ref_lon,
                                    target_lat, target_lon)

    i_offset = dx_m / parent_dx
    j_offset = dy_m / parent_dx

    parent_center_i = (parent_e_we + 1) / 2.0
    parent_center_j = (parent_e_sn + 1) / 2.0

    child_size_i = (child_e_we - 1) / parent_ratio
    child_size_j = (child_e_sn - 1) / parent_ratio

    i_parent_start_raw = parent_center_i + i_offset - child_size_i / 2.0
    j_parent_start_raw = parent_center_j + j_offset - child_size_j / 2.0

    i_parent_start = int(round(i_parent_start_raw))
    j_parent_start = int(round(j_parent_start_raw))

    return i_parent_start, j_parent_start, child_size_i, child_size_j


def compute_domain_center(ref_lat, ref_lon, parent_dx, parent_e_we, parent_e_sn,
                          child_e_we, child_e_sn,
                          i_parent_start, j_parent_start, parent_ratio):
    """
    Compute the center lat/lon of a nested domain given its i/j_parent_start.

    Returns:
        (center_lat, center_lon)
    """
    R_EARTH = 6371000.0

    # Parent domain center
    parent_center_i = (parent_e_we + 1) / 2.0
    parent_center_j = (parent_e_sn + 1) / 2.0

    # Child domain center in parent grid coordinates
    child_size_i = (child_e_we - 1) / parent_ratio
    child_size_j = (child_e_sn - 1) / parent_ratio
    child_center_i = i_parent_start + child_size_i / 2.0
    child_center_j = j_parent_start + child_size_j / 2.0

    # Offset from parent center in meters
    di = (child_center_i - parent_center_i) * parent_dx
    dj = (child_center_j - parent_center_j) * parent_dx

    # Convert to lat/lon
    center_lat = ref_lat + (dj / R_EARTH) * (180.0 / np.pi)
    center_lon = ref_lon + (di / R_EARTH) * (180.0 / np.pi) / cos(radians(center_lat))

    return center_lat, center_lon


def validate_bounds(i_start, j_start, child_e_we, child_e_sn, parent_ratio, parent_e_we, parent_e_sn):
    """Check if nesting parameters satisfy WRF constraints."""
    if i_start < 1 or j_start < 1:
        return False

    child_size_i = (child_e_we - 1) / parent_ratio
    child_size_j = (child_e_sn - 1) / parent_ratio

    i_end = i_start + child_size_i
    j_end = j_start + child_size_j

    if i_end > parent_e_we or j_end > parent_e_sn:
        return False

    return True


# ============================================================
# Iterative Refinement
# ============================================================

def refine_d03_position(initial_i_d02, initial_j_d02, initial_i_d03, initial_j_d03):
    """
    Iteratively refine d03's i_parent_start/j_parent_start to center it on TARGET.

    Args:
        initial_i_d02, initial_j_d02: d02's position in d01
        initial_i_d03, initial_j_d03: d03's initial position in d02

    Returns:
        final_i_d03, final_j_d03, center_lat, center_lon, iterations
    """
    print("=" * 70)
    print("d03 Center Alignment - Iterative Refinement")
    print("=" * 70)
    print(f"\nTarget: {TARGET_LAT:.4f}°N, {TARGET_LON:.4f}°W (SWiFT wind farm)\n")

    # First compute d02 center (fixed)
    center_lat_d02, center_lon_d02 = compute_domain_center(
        D01_REF_LAT, D01_REF_LON, D01_DX, D01_E_WE, D01_E_SN,
        D02_E_WE, D02_E_SN,
        initial_i_d02, initial_j_d02, D02_PARENT_RATIO
    )

    print(f"d02 center (fixed): {center_lat_d02:.4f}°N, {center_lon_d02:.4f}°W")
    print(f"d02 parameters: i_parent_start={initial_i_d02}, j_parent_start={initial_j_d02}")
    print()

    # Initialize d03 position
    i_d03 = initial_i_d03
    j_d03 = initial_j_d03

    # Compute initial d03 center
    center_lat_d03, center_lon_d03 = compute_domain_center(
        center_lat_d02, center_lon_d02, D02_DX, D02_E_WE, D02_E_SN,
        D03_E_WE, D03_E_SN,
        i_d03, j_d03, D03_PARENT_RATIO
    )

    dlat = TARGET_LAT - center_lat_d03
    dlon = TARGET_LON - center_lon_d03

    print(f"Initial d03 parameters:")
    print(f"  i_parent_start = {i_d03}")
    print(f"  j_parent_start = {j_d03}")
    print(f"  Center: {center_lat_d03:.4f}°N, {center_lon_d03:.4f}°W")
    print(f"  Error: Δlat={dlat:+.4f}°, Δlon={dlon:+.4f}°")
    print()

    # Iterative refinement
    print("Iteration history:")
    print("-" * 70)

    for iteration in range(MAX_ITERATIONS):
        # Compute current center
        center_lat_d03, center_lon_d03 = compute_domain_center(
            center_lat_d02, center_lon_d02, D02_DX, D02_E_WE, D02_E_SN,
            D03_E_WE, D03_E_SN,
            i_d03, j_d03, D03_PARENT_RATIO
        )

        # Compute error
        dlat = TARGET_LAT - center_lat_d03
        dlon = TARGET_LON - center_lon_d03

        error_km = sqrt(dlat**2 * 111**2 + (dlon * cos(radians(TARGET_LAT)))**2 * 111**2)

        print(f"Iter {iteration + 1}: i={i_d03:3d}, j={j_d03:3d}  |  "
              f"Center: {center_lat_d03:.4f}°N, {center_lon_d03:.4f}°W  |  "
              f"Error: {error_km:.2f} km")

        # Check convergence
        if abs(dlat) < CONVERGENCE_TOL and abs(dlon) < CONVERGENCE_TOL:
            print(f"\n✓ Converged after {iteration + 1} iterations (error < {CONVERGENCE_TOL}°)")
            break

        # Compute grid correction
        # Convert lat/lon error to d02 grid units
        dj_grid = dlat * 111000.0 / D02_DX
        di_grid = dlon * 111000.0 * cos(radians(TARGET_LAT)) / D02_DX

        # Apply correction
        i_d03_new = i_d03 + int(round(di_grid))
        j_d03_new = j_d03 + int(round(dj_grid))

        # Validate bounds
        if not validate_bounds(i_d03_new, j_d03_new, D03_E_WE, D03_E_SN,
                               D03_PARENT_RATIO, D02_E_WE, D02_E_SN):
            print(f"\n⚠️  Correction would violate bounds: i={i_d03_new}, j={j_d03_new}")
            print("Using previous valid parameters.")
            break

        # Update
        i_d03 = i_d03_new
        j_d03 = j_d03_new

    else:
        print(f"\n⚠️  Did not converge after {MAX_ITERATIONS} iterations")

    print("-" * 70)

    # Final center
    center_lat_d03, center_lon_d03 = compute_domain_center(
        center_lat_d02, center_lon_d02, D02_DX, D02_E_WE, D02_E_SN,
        D03_E_WE, D03_E_SN,
        i_d03, j_d03, D03_PARENT_RATIO
    )

    return i_d03, j_d03, center_lat_d03, center_lon_d03, iteration + 1


# ============================================================
# Main
# ============================================================

def main():
    # Step 1: Compute initial d02 parameters (unchanged)
    print("Step 1: Computing d02 parameters...")
    i_start_d02, j_start_d02, _, _ = compute_parent_start(
        TARGET_LAT, TARGET_LON,
        D01_REF_LAT, D01_REF_LON, D01_DX, D01_E_WE, D01_E_SN,
        D02_E_WE, D02_E_SN, D02_PARENT_RATIO
    )
    print(f"  d02: i_parent_start={i_start_d02}, j_parent_start={j_start_d02}")
    print()

    # Step 2: Compute initial d03 parameters (as starting guess)
    print("Step 2: Computing initial d03 parameters...")
    center_lat_d02, center_lon_d02 = compute_domain_center(
        D01_REF_LAT, D01_REF_LON, D01_DX, D01_E_WE, D01_E_SN,
        D02_E_WE, D02_E_SN,
        i_start_d02, j_start_d02, D02_PARENT_RATIO
    )

    i_start_d03_initial, j_start_d03_initial, _, _ = compute_parent_start(
        TARGET_LAT, TARGET_LON,
        center_lat_d02, center_lon_d02, D02_DX, D02_E_WE, D02_E_SN,
        D03_E_WE, D03_E_SN, D03_PARENT_RATIO
    )
    print(f"  d03 (initial): i_parent_start={i_start_d03_initial}, j_parent_start={j_start_d03_initial}")
    print()

    # Step 3: Refine d03 position
    i_start_d03, j_start_d03, center_lat_d03, center_lon_d03, iterations = refine_d03_position(
        i_start_d02, j_start_d02,
        i_start_d03_initial, j_start_d03_initial
    )

    # Step 4: Final verification
    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)
    print()

    # Distance from target
    dx_m, dy_m = latlon_to_meters(TARGET_LAT, TARGET_LON, center_lat_d03, center_lon_d03)
    distance_km = sqrt(dx_m**2 + dy_m**2) / 1000.0

    print(f"d03 final center: {center_lat_d03:.4f}°N, {center_lon_d03:.4f}°W")
    print(f"Target (SWiFT):   {TARGET_LAT:.4f}°N, {TARGET_LON:.4f}°W")
    print(f"Distance to target: {distance_km:.2f} km")

    if distance_km < 1.0:
        print("✓ d03 center within 1 km of SWiFT")
    else:
        print(f"⚠️  d03 center is {distance_km:.2f} km from SWiFT")

    print()

    # Output namelist.wps format
    print("=" * 70)
    print("Recommended namelist.wps Parameters")
    print("=" * 70)
    print()
    print("&geogrid")
    print(f" parent_id         =   1,   1,   2,")
    print(f" parent_grid_ratio =   1,   {D02_PARENT_RATIO},   {D03_PARENT_RATIO},")
    print(f" i_parent_start    =   1, {i_start_d02:3d}, {i_start_d03:3d},")
    print(f" j_parent_start    =   1, {j_start_d02:3d}, {j_start_d03:3d},")
    print(f" e_we              = {D01_E_WE}, {D02_E_WE}, {D03_E_WE},")
    print(f" e_sn              = {D01_E_SN}, {D02_E_SN}, {D03_E_SN},")
    print(f" dx                = {D01_DX:.0f},")
    print(f" dy                = {D01_DX:.0f},")
    print("/")
    print()

    # Constraint validation
    valid = validate_bounds(i_start_d03, j_start_d03, D03_E_WE, D03_E_SN,
                           D03_PARENT_RATIO, D02_E_WE, D02_E_SN)

    if valid:
        print("✓ All WRF nesting constraints satisfied")
    else:
        print("✗ WARNING: Nesting constraints violated")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
