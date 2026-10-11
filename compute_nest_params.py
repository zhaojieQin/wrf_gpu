#!/usr/bin/env python3
"""
Calculate i_parent_start/j_parent_start for SWiFT nested domains.

Goal: Center d03 on SWiFT wind farm (33.6°N, 102.1°W) instead of current
      misaligned position (33.44°N, 96.74°W).
"""

import numpy as np
from math import radians, cos, sin, sqrt

# ============================================================
# Configuration
# ============================================================

# Target: SWiFT wind farm location
TARGET_LAT = 33.6
TARGET_LON = -102.1

# d01 configuration (outer domain)
D01_DX = 27000.0  # m
D01_E_WE = 118
D01_E_SN = 100
D01_REF_LAT = 32.67   # Reference latitude for projection
D01_REF_LON = -102.05  # Reference longitude for projection
D01_TRUELAT1 = 33.0    # Lambert standard parallel 1
D01_TRUELAT2 = 33.0    # Lambert standard parallel 2

# d02 configuration (middle domain)
D02_DX = 9000.0  # m
D02_E_WE = 154
D02_E_SN = 133
D02_PARENT_RATIO = 3

# d03 configuration (inner domain - to be centered on SWiFT)
D03_DX = 3000.0  # m
D03_E_WE = 118
D03_E_SN = 100
D03_PARENT_RATIO = 3


# ============================================================
# Helper Functions
# ============================================================

def latlon_to_meters(lat1, lon1, lat2, lon2):
    """
    Convert lat/lon offset to approximate meters using spherical geometry.

    For small distances (< 1000 km), this approximation is sufficient.
    """
    R_EARTH = 6371000.0  # Earth radius in meters

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    # Haversine formula for great circle distance
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * np.arctan2(sqrt(a), sqrt(1-a))
    distance = R_EARTH * c

    # Bearing-aware x/y components
    lat_avg = radians((lat1 + lat2) / 2)
    dy = dlat * R_EARTH  # North-south component
    dx = dlon * R_EARTH * cos(lat_avg)  # East-west component

    return dx, dy


def compute_parent_start(target_lat, target_lon,
                         parent_ref_lat, parent_ref_lon, parent_dx, parent_e_we, parent_e_sn,
                         child_e_we, child_e_sn, parent_ratio):
    """
    Compute i_parent_start and j_parent_start for a nested domain.

    Args:
        target_lat, target_lon: Desired center of child domain
        parent_ref_lat, parent_ref_lon: Parent domain reference point
        parent_dx: Parent domain grid spacing (m)
        parent_e_we, parent_e_sn: Parent domain dimensions
        child_e_we, child_e_sn: Child domain dimensions
        parent_ratio: Grid refinement ratio

    Returns:
        i_parent_start, j_parent_start (both integers)
    """
    # Convert lat/lon offset to meters
    dx_m, dy_m = latlon_to_meters(parent_ref_lat, parent_ref_lon,
                                    target_lat, target_lon)

    # Convert meters to parent grid points
    # WRF convention: i is x-direction (west-east), j is y-direction (south-north)
    i_offset = dx_m / parent_dx
    j_offset = dy_m / parent_dx

    # Parent domain center in grid coordinates (1-indexed)
    parent_center_i = (parent_e_we + 1) / 2.0
    parent_center_j = (parent_e_sn + 1) / 2.0

    # Child domain size in parent grid units
    child_size_i = (child_e_we - 1) / parent_ratio
    child_size_j = (child_e_sn - 1) / parent_ratio

    # Calculate parent_start (lower-left corner of child in parent grid)
    # Target is at child center, so subtract half child size
    i_parent_start_raw = parent_center_i + i_offset - child_size_i / 2.0
    j_parent_start_raw = parent_center_j + j_offset - child_size_j / 2.0

    # Round to nearest integer
    i_parent_start = int(round(i_parent_start_raw))
    j_parent_start = int(round(j_parent_start_raw))

    return i_parent_start, j_parent_start, child_size_i, child_size_j


def validate_nesting_constraints(i_start, j_start,
                                  child_e_we, child_e_sn, parent_ratio,
                                  parent_e_we, parent_e_sn, domain_name):
    """
    Check WRF nesting constraints and return list of violations.
    """
    violations = []

    # Constraint 1: i_parent_start, j_parent_start must be >= 1
    if i_start < 1:
        violations.append(f"{domain_name}: i_parent_start={i_start} < 1")
    if j_start < 1:
        violations.append(f"{domain_name}: j_parent_start={j_start} < 1")

    # Constraint 2: (e_we - 1) / parent_ratio must be integer
    if (child_e_we - 1) % parent_ratio != 0:
        violations.append(f"{domain_name}: (e_we-1)={child_e_we-1} not divisible by ratio={parent_ratio}")
    if (child_e_sn - 1) % parent_ratio != 0:
        violations.append(f"{domain_name}: (e_sn-1)={child_e_sn-1} not divisible by ratio={parent_ratio}")

    # Constraint 3: Child domain must not exceed parent boundaries
    child_size_i = (child_e_we - 1) / parent_ratio
    child_size_j = (child_e_sn - 1) / parent_ratio

    i_end = i_start + child_size_i
    j_end = j_start + child_size_j

    if i_end > parent_e_we:
        violations.append(f"{domain_name}: i_end={i_end:.1f} > parent_e_we={parent_e_we}")
    if j_end > parent_e_sn:
        violations.append(f"{domain_name}: j_end={j_end:.1f} > parent_e_sn={parent_e_sn}")

    return violations


def compute_domain_bounds(ref_lat, ref_lon, dx, e_we, e_sn,
                          i_parent_start, j_parent_start, parent_ratio, parent_dx):
    """
    Compute the lat/lon bounds of a nested domain.

    Args:
        ref_lat, ref_lon: Parent domain reference point
        dx: This domain's grid spacing (m)
        e_we, e_sn: This domain's dimensions
        i_parent_start, j_parent_start: Starting indices in parent grid
        parent_ratio: Grid refinement ratio
        parent_dx: Parent domain grid spacing (m)

    Returns:
        (lat_min, lat_max, lon_min, lon_max, center_lat, center_lon)
    """
    # Parent domain center
    parent_center_i = (D01_E_WE + 1) / 2.0
    parent_center_j = (D01_E_SN + 1) / 2.0

    # Child domain center in parent grid coordinates
    child_size_i = (e_we - 1) / parent_ratio
    child_size_j = (e_sn - 1) / parent_ratio
    child_center_i = i_parent_start + child_size_i / 2.0
    child_center_j = j_parent_start + child_size_j / 2.0

    # Offset from parent center in meters
    di = (child_center_i - parent_center_i) * parent_dx
    dj = (child_center_j - parent_center_j) * parent_dx

    # Convert to lat/lon (approximate)
    R_EARTH = 6371000.0
    center_lat = ref_lat + (dj / R_EARTH) * (180.0 / np.pi)
    center_lon = ref_lon + (di / R_EARTH) * (180.0 / np.pi) / cos(radians(center_lat))

    # Domain half-width in meters
    half_width_x = (e_we - 1) * dx / 2.0
    half_width_y = (e_sn - 1) * dx / 2.0

    # Convert to lat/lon extent
    dlat = (half_width_y / R_EARTH) * (180.0 / np.pi)
    dlon = (half_width_x / R_EARTH) * (180.0 / np.pi) / cos(radians(center_lat))

    lat_min = center_lat - dlat
    lat_max = center_lat + dlat
    lon_min = center_lon - dlon
    lon_max = center_lon + dlon

    return lat_min, lat_max, lon_min, lon_max, center_lat, center_lon


# ============================================================
# Main Computation
# ============================================================

def main():
    print("=" * 70)
    print("SWiFT Nested Domain Parameter Calculation")
    print("=" * 70)
    print(f"\nTarget: Center d03 on SWiFT wind farm at {TARGET_LAT}°N, {TARGET_LON}°W")
    print()

    # --------------------------------------------------------
    # Step 1: Compute d02 parameters (nested in d01)
    # --------------------------------------------------------
    print("Step 1: Computing d02 nesting parameters (d02 in d01)...")

    i_start_d02, j_start_d02, child_size_i_d02, child_size_j_d02 = compute_parent_start(
        TARGET_LAT, TARGET_LON,
        D01_REF_LAT, D01_REF_LON, D01_DX, D01_E_WE, D01_E_SN,
        D02_E_WE, D02_E_SN, D02_PARENT_RATIO
    )

    print(f"  d02 in d01: i_parent_start={i_start_d02}, j_parent_start={j_start_d02}")
    print(f"  d02 size in d01 grid: {child_size_i_d02:.1f} x {child_size_j_d02:.1f}")

    # Validate d02
    violations_d02 = validate_nesting_constraints(
        i_start_d02, j_start_d02,
        D02_E_WE, D02_E_SN, D02_PARENT_RATIO,
        D01_E_WE, D01_E_SN, "d02"
    )

    if violations_d02:
        print("\n  ⚠️  d02 CONSTRAINT VIOLATIONS:")
        for v in violations_d02:
            print(f"    - {v}")
    else:
        print("  ✓ d02 constraints satisfied")

    # --------------------------------------------------------
    # Step 2: Compute d03 parameters (nested in d02)
    # --------------------------------------------------------
    print("\nStep 2: Computing d03 nesting parameters (d03 in d02)...")

    # d03 should also be centered on TARGET, but now relative to d02
    # First compute d02's actual center
    lat_min_d02, lat_max_d02, lon_min_d02, lon_max_d02, center_lat_d02, center_lon_d02 = \
        compute_domain_bounds(
            D01_REF_LAT, D01_REF_LON, D02_DX, D02_E_WE, D02_E_SN,
            i_start_d02, j_start_d02, D02_PARENT_RATIO, D01_DX
        )

    print(f"  d02 computed center: {center_lat_d02:.4f}°N, {center_lon_d02:.4f}°W")

    # Now compute d03 relative to d02
    i_start_d03, j_start_d03, child_size_i_d03, child_size_j_d03 = compute_parent_start(
        TARGET_LAT, TARGET_LON,
        center_lat_d02, center_lon_d02, D02_DX, D02_E_WE, D02_E_SN,
        D03_E_WE, D03_E_SN, D03_PARENT_RATIO
    )

    print(f"  d03 in d02: i_parent_start={i_start_d03}, j_parent_start={j_start_d03}")
    print(f"  d03 size in d02 grid: {child_size_i_d03:.1f} x {child_size_j_d03:.1f}")

    # Validate d03
    violations_d03 = validate_nesting_constraints(
        i_start_d03, j_start_d03,
        D03_E_WE, D03_E_SN, D03_PARENT_RATIO,
        D02_E_WE, D02_E_SN, "d03"
    )

    if violations_d03:
        print("\n  ⚠️  d03 CONSTRAINT VIOLATIONS:")
        for v in violations_d03:
            print(f"    - {v}")
    else:
        print("  ✓ d03 constraints satisfied")

    # --------------------------------------------------------
    # Step 3: Compute actual domain bounds for verification
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("Verification: Computed Domain Bounds")
    print("=" * 70)

    # d01 bounds (centered at ref_lat, ref_lon)
    half_width_x_d01 = (D01_E_WE - 1) * D01_DX / 2.0
    half_width_y_d01 = (D01_E_SN - 1) * D01_DX / 2.0
    R_EARTH = 6371000.0
    dlat_d01 = (half_width_y_d01 / R_EARTH) * (180.0 / np.pi)
    dlon_d01 = (half_width_x_d01 / R_EARTH) * (180.0 / np.pi) / cos(radians(D01_REF_LAT))

    print(f"\nd01: {D01_E_WE}×{D01_E_SN}, dx={D01_DX/1000:.1f} km")
    print(f"  Center: {D01_REF_LAT:.4f}°N, {D01_REF_LON:.4f}°W")
    print(f"  Extent: lat [{D01_REF_LAT-dlat_d01:.4f}, {D01_REF_LAT+dlat_d01:.4f}], "
          f"lon [{D01_REF_LON-dlon_d01:.4f}, {D01_REF_LON+dlon_d01:.4f}]")

    # d02 bounds
    print(f"\nd02: {D02_E_WE}×{D02_E_SN}, dx={D02_DX/1000:.1f} km")
    print(f"  Center: {center_lat_d02:.4f}°N, {center_lon_d02:.4f}°W")
    print(f"  Extent: lat [{lat_min_d02:.4f}, {lat_max_d02:.4f}], "
          f"lon [{lon_min_d02:.4f}, {lon_max_d02:.4f}]")

    # d03 bounds (nested in d02, using d02 as reference)
    lat_min_d03, lat_max_d03, lon_min_d03, lon_max_d03, center_lat_d03, center_lon_d03 = \
        compute_domain_bounds(
            center_lat_d02, center_lon_d02, D03_DX, D03_E_WE, D03_E_SN,
            i_start_d03, j_start_d03, D03_PARENT_RATIO, D02_DX
        )

    print(f"\nd03: {D03_E_WE}×{D03_E_SN}, dx={D03_DX/1000:.1f} km")
    print(f"  Target center: {TARGET_LAT:.4f}°N, {TARGET_LON:.4f}°W")
    print(f"  Computed center: {center_lat_d03:.4f}°N, {center_lon_d03:.4f}°W")
    print(f"  Extent: lat [{lat_min_d03:.4f}, {lat_max_d03:.4f}], "
          f"lon [{lon_min_d03:.4f}, {lon_max_d03:.4f}]")

    # Check if SWiFT is inside d03
    swift_in_d03_lat = (lat_min_d03 <= TARGET_LAT <= lat_max_d03)
    swift_in_d03_lon = (lon_min_d03 <= TARGET_LON <= lon_max_d03)
    swift_in_d03 = swift_in_d03_lat and swift_in_d03_lon

    print(f"  SWiFT wind farm in d03: {'✓ YES' if swift_in_d03 else '✗ NO'}")

    # --------------------------------------------------------
    # Step 4: Output namelist.wps format
    # --------------------------------------------------------
    print("\n" + "=" * 70)
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

    # --------------------------------------------------------
    # Step 5: Summary
    # --------------------------------------------------------
    all_violations = violations_d02 + violations_d03

    print("=" * 70)
    print("Summary")
    print("=" * 70)

    if all_violations:
        print("\n⚠️  CONSTRAINT VIOLATIONS DETECTED:")
        for v in all_violations:
            print(f"  - {v}")
        print("\nRecommendation: Adjust e_we/e_sn to satisfy (e_we-1) % parent_ratio == 0")
    else:
        print("\n✓ All WRF nesting constraints satisfied")

    if swift_in_d03:
        print("✓ SWiFT wind farm is inside d03")
    else:
        print("✗ WARNING: SWiFT wind farm is outside d03 bounds")
        print("  This may indicate calculation error or invalid target coordinates")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
