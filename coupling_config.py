"""Shared WRF-LBM coupling configuration.

Single source of truth for coupling parameters.
"""

# Forecast duration (hours)
HOURS = 1

# d03 timestep (seconds): time_step=18 / (ratio_d02=3 * ratio_d03=3) = 2s
# Coupling fires every d03 timestep, so interval = target_dt
D03_DT_SECONDS = 2

# Coupling interval (seconds) — kept for reference, not used for alarm scheduling
INTERVAL_SECONDS = D03_DT_SECONDS

# LBM steps = HOURS * 3600 / D03_DT_SECONDS = 1 * 3600 / 2 = 1800

# LBM VTK output: write only the first few outputs, then switch output off.
# Each output is ~1-2 GB (≈21.9M nodes). 15 LBM steps ≈ one WRF frame (2 s).
# Current setting: every LBM step for steps 1..5 (VF also writes the t=0 state).
LBM_OUTPUT_EVERY_STEPS = 10
LBM_OUTPUT_COUNT = 5

# LBM lateral + roof boundary type:
#   "velocity"   — fixed (Dirichlet) WRF velocity on all five open faces
#                  (VF VelocityInterpolatedCompressible); interior evolves freely.
#                  No net-flux correction yet: watch for mean-density drift.
#   "freestream" — per-node inflow/outflow switching (FreestreamInterpolatedCompressible)
LBM_BC_MODE = "velocity"

# LBM temperature field (diurnal cycle), set up as VF apps/gpu/StratifiedBoundaryLayer:
#   - temperature as advection-diffusion scalar θ − θ_ref (F16 kernel), Pr = 0.7
#   - buoyancy from the deviation to the horizontal planar average
#   - ground: surface-layer BC driven by WRF sensible heat flux HFX (every step)
#   - five open faces: temperature Dirichlet from WRF θ (every step)
#   - SGS: AMDStratified (momentum and temperature), constant 0.2
# False → neutral setup (ground stress BC, no temperature), as before.
LBM_THERMAL = True
LBM_THETA_REF = 300.0          # K, reference potential temperature (θ − θ_ref is stored)
LBM_RHO_CP = 1.225 * 1004.5    # J m^-3 K^-1, converts HFX [W/m^2] to kinematic flux [K m/s]
