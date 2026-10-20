"""Shared WRF-LBM coupling configuration.

Single source of truth for coupling parameters.
"""

# WRF spin-up (hours): WRF runs alone first, no data is sent to the LBM.
# Coupling starts at t = SPINUP_HOURS (WRF model time); the LBM is initialised
# from the first WRF frame after spin-up. Total WRF forecast = SPINUP_HOURS + HOURS.
# Diurnal-cycle validation (SWiFT, 大气稳定度与风力机尾流日周期演变研究 §3.5):
#   WRF starts 2013-11-07 18 UTC (namelist), 12 h spin-up → coupling 2013-11-08 06 UTC
#   (00:00 LST, LST = UTC−6) to 2013-11-09 06 UTC (24:00 LST); wrfbdy_d01 ends at 11-09 06 UTC.
SPINUP_HOURS = 12
COUPLING_START_SECONDS = SPINUP_HOURS * 3600

# Coupled duration (hours), counted from the end of the spin-up
HOURS = 24

# WRF case (input dir with namelist.input, wrfinput_d01..d03, wrfbdy_d01) and run output dir.
# SWiFT_100m: d01 2.5 km / d02 500 m / d03 100 m, centred on the SWiFT site, ERA5-driven
#             (time_step=10 s, ratio 5 → d03 dt = 0.4 s).
# SWiFT     : previous 27/9/3 km case (time_step=18 s, ratio 3 → d03 dt = 2 s).
# The d03 timestep is read from the case (namelist / wrfinput_d03), nothing to keep in sync here.
WRF_CASE_DIR = "/home/zhaqi746/code/wrf_gpu/examples/SWiFT_100m"
RUN_DIR = "/home/zhaqi746/code/wrf_gpu/runs/swift100m_coupled_mpi"

# Coupling cadence: one WRF frame is sent to the LBM every COUPLING_EVERY_N_STEPS d03 timesteps.
#   frame interval = N × d03 dt  (SWiFT_100m: 5 × 0.4 s = 2 s ≈ 15 LBM steps; SWiFT 3 km: N = 1 → 2 s)
COUPLING_EVERY_N_STEPS = 5

# LBM domain (overrides abl.cfg geometry; the .cfg file itself is not changed).
#   4 × 4 × 1.6 km at dx = 10 m → 400 × 400 × 160 nodes (≈25.6M). The tower sits at the domain
#   centre, ~2 km downstream of the SSW inflow faces (CPM turbulence needs fetch).
#   None → abl.cfg values (6H × 4H × H, dx = H / NodesPerBoundaryLayerHeight).
LBM_DOMAIN_LX = 4000.0
LBM_DOMAIN_LY = 4000.0
LBM_DOMAIN_LZ = 1600.0
LBM_DX = 10.0
# LBM timestep (s). None → from abl.cfg Ma (dx = 10 m, H = 1.6 km → dt ≈ 0.119 s, dx/dt ≈ 84 m/s:
# lattice Mach √3·U·dt/dx ≈ 0.31 at 15 m/s, 0.41 at 20 m/s). The night low-level jet and the winds
# near 1.6 km can exceed 15 m/s; if [lbm_init] warns about Mach > 0.3, set e.g. 0.06 s
# (Mach ≈ 0.21 at 20 m/s, twice the LBM steps).
LBM_DT_S = None

# LBM full-field VTK output (~1-2 GB per output, ≈21.9M nodes). Off by default.
# When on: write every LBM_OUTPUT_EVERY_STEPS steps, LBM_OUTPUT_COUNT times, then stop.
LBM_FULL_VTK_OUTPUT = False
LBM_OUTPUT_EVERY_STEPS = 10
LBM_OUTPUT_COUNT = 5

# LBM plane probe: streamwise x–z plane at mid-y (y ≈ L_y/2, one node thick),
# instantaneous u, v, w, rho (+ temperature when LBM_THERMAL), whole run.
# ~64k nodes per output. 24 h run: 500 steps ≈ 60 s → ~1450 outputs (15 steps would give ~48k).
# None → no probe.
LBM_PROBE_EVERY_STEPS = 500

# Diurnal-cycle validation sampling (lbm_adapter/validation.py → RUN_DIR/lbm/val*, RUN_DIR/validation/;
# post-processing: scripts/validate_swift.py).
VALIDATION_ENABLE = True
# TTU 200 m tower (lat, lon). None → LBM domain centre (the d03 / LBM domain is centred on the
# SWiFT site 33.6105 N, 102.0505 W). Must lie inside the LBM domain.
VALIDATION_TOWER_LATLON = None
VALIDATION_TOWER_ZMAX_M = 1500.0       # tower column: every LBM node from the ground up to this height
VALIDATION_TOWER_EVERY_S = 1.0         # tower time series sampling (≈1 Hz)
VALIDATION_TOWER_WRITE_EVERY_S = 300.0 # tower file flush interval (GPU buffer = this / dt steps)
VALIDATION_PLANAR_EVERY_S = 60.0       # horizontal planar averages (profiles of means / covariances)
VALIDATION_HUB_PLANE_Z_M = 70.0        # hub-height horizontal plane (2 MW hub, §3.5)
VALIDATION_HUB_PLANE_EVERY_S = 600.0   # None → no hub plane
VALIDATION_WRF_FOOTPRINT_EVERY_S = 60.0  # WRF mean profile over the LBM footprint

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

# Cell perturbation method (CPM) for inflow turbulence (needs LBM_THERMAL).
# Muñoz-Esparza et al. 2015 (PoF), 2017 (JAMES), Muñoz-Esparza & Kosović 2018 (MWR); see lbm_adapter/cpm.py.
# θ perturbations in 2-D cells (n_c × n_c nodes, one grid level each) along the lateral faces, weighted by
# the inflow switch w_in, refreshed every t_p = Γ d_c / U_1 and seeded instantaneously into the GPU distributions.
LBM_CPM_ENABLE = True
LBM_CPM_CELL_NODES = 8           # cell side (grid nodes); 8 ≈ 83 m at dx = 10.4 m
LBM_CPM_ROWS = 3                 # rows of cells along each lateral face
LBM_CPM_OFFSET_NODES = 3         # first row starts this many nodes inside the boundary (no relaxation zone)
LBM_CPM_SKIP_LEVELS = 3          # lowest fluid levels left unperturbed (2017)
LBM_CPM_Z_MAX_M = 600.0          # m AGL, highest perturbed level (also kept 2 dx below the domain top)
LBM_CPM_ZI_FRACTION = 2.0 / 3.0  # z_CP = fraction · z_i
LBM_CPM_STABLE_DTHDZ = 0.014     # K/m, stable z_i: lowest height with dθ/dz below this (2017)
LBM_CPM_CONV_MIN_DTHDZ = 0.003   # K/m, convective z_i = height of max dθ/dz; below this → no inversion in domain
LBM_CPM_EC = 0.2                 # perturbation Eckert number, θ_pm = U_g² / (c_p Ec)
LBM_CPM_CP = 1004.5              # J kg⁻¹ K⁻¹
LBM_CPM_GAMMA = 1.0              # Γ = t_p U_1 / d_c (d_c = cell diagonal, U_1 = wind at first level)
LBM_CPM_TP_MIN_S = 5.0           # s, clamp for t_p
LBM_CPM_TP_MAX_S = 300.0         # s
LBM_CPM_U0 = 0.5                 # m/s, w_in = ½[1 − tanh(u_n / u0)]
LBM_CPM_ANGLE_MAX_DEG = None     # deg; e.g. 70 → no perturbation if wind vs inward normal exceeds it. None: off
LBM_CPM_STABLE_SCHEME = "ec"     # stable regime amplitude: "ec" or "rim" (2018, Ri_m scaling)
LBM_CPM_RIM = -1.0               # Ri_m for LBM_CPM_STABLE_SCHEME = "rim"
LBM_CPM_THETA_PM_MAX = 2.0       # K, safety cap on the perturbation amplitude
LBM_CPM_MOMENTUM = False         # also perturb u, v in the same cells (no literature optimum)
LBM_CPM_MOMENTUM_ALPHA = 0.05    # u', v' amplitude as a fraction of the local wind speed
LBM_CPM_SEED = 12345             # RNG seed (reproducible perturbations)
