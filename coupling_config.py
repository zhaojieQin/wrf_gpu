"""Shared WRF-LBM coupling configuration.

Single source of truth for coupling parameters.
"""

# WRF spin-up (hours): WRF runs alone first, no data is sent to the LBM.
# Coupling starts at t = SPINUP_HOURS (WRF model time); the LBM is initialised
# from the first WRF frame after spin-up. Total WRF forecast = SPINUP_HOURS + HOURS.
SPINUP_HOURS = 1
COUPLING_START_SECONDS = SPINUP_HOURS * 3600

# Coupled duration (hours), counted from the end of the spin-up
HOURS = 1

# d03 timestep (seconds): time_step=18 / (ratio_d02=3 * ratio_d03=3) = 2s
# Coupling fires every d03 timestep, so interval = target_dt
D03_DT_SECONDS = 2

# Coupling interval (seconds) — kept for reference, not used for alarm scheduling
INTERVAL_SECONDS = D03_DT_SECONDS

# LBM steps = HOURS * 3600 / D03_DT_SECONDS = 1 * 3600 / 2 = 1800

# LBM full-field VTK output (~1-2 GB per output, ≈21.9M nodes). Off by default.
# When on: write every LBM_OUTPUT_EVERY_STEPS steps, LBM_OUTPUT_COUNT times, then stop.
LBM_FULL_VTK_OUTPUT = False
LBM_OUTPUT_EVERY_STEPS = 10
LBM_OUTPUT_COUNT = 5

# LBM plane probe: streamwise x–z plane at mid-y (y ≈ L_y/2, one node thick),
# instantaneous u, v, w, rho (+ temperature when LBM_THERMAL), whole run.
# ~57k nodes per output. 15 LBM steps ≈ one WRF frame (2 s). None → no probe.
LBM_PROBE_EVERY_STEPS = 15

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
# K, reference potential temperature (θ − θ_ref is stored).
# None → automatic: mean WRF θ over the LBM volume in the first coupled frame.
LBM_THETA_REF = None
LBM_RHO_CP = 1.225 * 1004.5    # J m^-3 K^-1, converts HFX [W/m^2] to kinematic flux [K m/s]
# Temperature Dirichlet scheme on the five open faces:
#   "interpolated"   → DirichletInterpolatedSlip (uses subgrid distance q, analogous to VelocityInterpolatedCompressible)
#   "antibounceback" → DirichletAntiBounceBackSlip (previous behaviour)
LBM_AD_DIRICHLET_BC = "interpolated"
