#!/usr/bin/env python3
"""Minimal WRF-LBM coupling adapter (Phase 1).

Single WRF instance + single LBM instance + shared COMM_WORLD + 1:1 point-to-point.
No orchestrator, no YAML config, no heartbeat, no MPI_Comm_split.

Usage:
    srun --ntasks=2 python coupling_adapter.py
"""
import os
import sys
import traceback
from pathlib import Path

# MPI init BEFORE any CUDA import
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# GPU assignment MUST happen before any CUDA library import
os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)

def safe_abort(exc):
    """Abort with error message, MPI-aware."""
    print(f"[Rank {rank}] FATAL: {exc}", file=sys.stderr, flush=True)
    traceback.print_exc()
    if MPI.Is_initialized() and not MPI.Is_finalized():
        comm.Abort(1)
    else:
        sys.exit(1)

if size != 2:
    if rank == 0:
        print(f"[错误] 需要 2 个 MPI ranks，当前 size={size}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    if rank == 0:
        print("[Rank 0] Starting WRF GPU forecast...", flush=True)
        from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, execute_nested_pipeline
        from coupling_config import HOURS, INTERVAL_SECONDS, D03_DT_SECONDS

        config = NestedPipelineConfig(
            input_dir=Path('/home/zhaqi746/code/wrf_gpu/examples/SWiFT'),
            output_dir=Path('/home/zhaqi746/code/wrf_gpu/runs/swift_coupled_mpi'),
            proof_dir=Path('/home/zhaqi746/code/wrf_gpu/runs/swift_coupled_mpi/proofs'),
            hours=HOURS,
            max_dom=3,
            coupling_config={
                'enabled': True,
                'interval_seconds': INTERVAL_SECONDS,
                'target_domain': 'd03',
                'dry_run': False,
                'mpi_dest_rank': 1,
                'subdomain_config': {
                    'z_range': (0, -1),
                    'y_range': (0, -1),
                    'x_range': (0, -1),
                    # hfx is appended by nested_pipeline from surface diagnostics
                    'fields': ['u', 'v', 'w', 'theta', 'ph'],
                },
            },
        )
        execute_nested_pipeline(config)
        print("[Rank 0] WRF forecast completed", flush=True)

        # Send EOF signal
        EOF_TAG = 999998
        print(f"[Rank 0] Sending EOF signal (tag={EOF_TAG})", flush=True)
        comm.send({"eof": True}, dest=1, tag=EOF_TAG)

        # Wait for ACK
        ack = comm.recv(source=1, tag=EOF_TAG)
        print(f"[Rank 0] Received ACK: {ack}", flush=True)
        print("[Rank 0] Done", flush=True)

    elif rank == 1:
        # VirtualFluids 只用 MPI_COMM_SELF，不参与 WRF 在 COMM_WORLD 上的集合通信。
        # 必须在任何 pyfluids 导入 / MPICommunicator 实例化之前设置。
        os.environ['VF_MPI_USE_COMM_SELF'] = '1'
        print("[Rank 1] Starting LBM coupled simulation...", flush=True)
        from coupling_config import (
            HOURS, D03_DT_SECONDS, LBM_OUTPUT_EVERY_STEPS, LBM_OUTPUT_COUNT,
            LBM_FULL_VTK_OUTPUT, LBM_PROBE_EVERY_STEPS,
            LBM_BC_MODE, LBM_THERMAL, LBM_THETA_REF, LBM_RHO_CP, LBM_AD_DIRICHLET_BC,
        )

        # ── lbm_adapter imports ──────────────────────────────────────────
        import numpy as np
        import netCDF4 as nc
        from lbm_adapter.vf_setup import WRFLBMConfig, ThermalConfig, setup_vf_simulation
        from lbm_adapter.wrf_to_lbm import WRFStaticContext
        from lbm_adapter.unit_convert import LBMReference
        from lbm_adapter.recv_loop import recv_next_wrf_step, WRFRecvLoop, READY_TAG
        from lbm_adapter.lbm_init import (
            init_lbm_from_wrf, make_wrf_initial_condition, get_bc_coordinates,
            make_wrf_initial_temperature,
        )
        from lbm_adapter.run_coupled import run_lbm_coupled

        # ── WRF 耦合参数（与 rank 0 coupling_config 保持一致）────────────
        wrf_steps = int(HOURS * 3600 / D03_DT_SECONDS)
        print(f"[Rank 1] WRF steps={wrf_steps} "
              f"(HOURS={HOURS}, D03_DT_SECONDS={D03_DT_SECONDS})", flush=True)

        # ── VirtualFluids 域配置（从 .cfg 文件读取，单一权威来源）────────
        VF_CFG = Path('/home/zhaqi746/code/wrf_gpu/VirtualFluids/apps/gpu/AtmosphericBoundaryLayer/abl.cfg')
        lbm_cfg = WRFLBMConfig.from_cfg_file(
            config_path=VF_CFG,
            t_end_s=HOURS * 3600.0,
            output_path=Path('/home/zhaqi746/code/wrf_gpu/runs/swift_coupled_mpi/lbm'),
        )

        # ── 从 wrfinput_d03 读取静态量（terrain_height, DX/DY, z0_mean）──
        WRFINPUT_D03 = Path('/home/zhaqi746/code/wrf_gpu/examples/SWiFT/wrfinput_d03')
        with nc.Dataset(WRFINPUT_D03, 'r') as ds:
            # HGT: (1, ny, nx) m MSL → squeeze to (ny, nx)
            terrain_height = np.asarray(ds.variables['HGT'][0], dtype=np.float64)
            dx_wrf = float(ds.DX)
            dy_wrf = float(ds.DY)
            # ZNT: (1, ny, nx) roughness length (m) — land-surface mean.
            # wrfinput from real.exe may not carry ZNT (it is a model-computed
            # diagnostic); fall back to the VF ground roughness from abl.cfg,
            # which is what the LBM stress BC actually uses.
            if 'ZNT' in ds.variables:
                znt = np.asarray(ds.variables['ZNT'][0], dtype=np.float64)
                z0_mean = float(znt.mean())
                z0_source = 'wrfinput_d03/ZNT'
            else:
                z0_mean = float(lbm_cfg.z0)
                z0_source = 'abl.cfg RoughnessLength (ZNT not in wrfinput_d03)'

        print(f"[Rank 1] wrfinput_d03: terrain shape={terrain_height.shape}  "
              f"dx={dx_wrf}m  dy={dy_wrf}m  z0_mean={z0_mean:.4f}m  ({z0_source})", flush=True)

        static_ctx = WRFStaticContext(
            terrain_height=terrain_height,
            dx_wrf=dx_wrf,
            dy_wrf=dy_wrf,
            z0_mean=z0_mean,
        )

        # LBM 参考量（从 lbm_cfg 导出，与 VF para 保持一致）
        ref = LBMReference(
            dx_lbm=lbm_cfg.dx,
            dt_lbm=lbm_cfg.dt,
            theta_ref=LBM_THETA_REF,
        )
        print(f"[Rank 1] LBM: dx={lbm_cfg.dx}m  dt={lbm_cfg.dt:.6f}s  "
              f"u_ref={ref.u_ref:.2f}m/s  n_steps={lbm_cfg.n_steps}", flush=True)

        # ── LBM 域在 WRF d03 网格中的水平定位（中心对齐）────────────────
        # VF bc_coords x/y 从 (0,0) 开始（LBM 域西南角）
        # 需要加 offset 转为 WRF d03 坐标系（西南角为原点）
        ny_wrf, nx_wrf = terrain_height.shape
        wrf_Lx = (nx_wrf - 1) * dx_wrf
        wrf_Ly = (ny_wrf - 1) * dy_wrf
        lbm_origin_x = (wrf_Lx - lbm_cfg.domain_length_x) / 2.0
        lbm_origin_y = (wrf_Ly - lbm_cfg.domain_length_y) / 2.0
        if lbm_origin_x < 0 or lbm_origin_y < 0:
            raise RuntimeError(
                f"LBM domain ({lbm_cfg.domain_length_x:.0f}m × "
                f"{lbm_cfg.domain_length_y:.0f}m) is larger than "
                f"WRF d03 domain ({wrf_Lx:.0f}m × {wrf_Ly:.0f}m). "
                "Cannot center-align."
            )
        print(f"[Rank 1] LBM origin in WRF coords: "
              f"x={lbm_origin_x:.0f}m  y={lbm_origin_y:.0f}m  "
              f"(center-aligned, WRF SW corner = origin)", flush=True)

        def _add_xy_offset(coords: np.ndarray) -> np.ndarray:
            """bc/sl coords (n, 3): add (lbm_origin_x, lbm_origin_y) to x/y."""
            out = coords.copy()
            out[:, 0] += lbm_origin_x
            out[:, 1] += lbm_origin_y
            return out

        # ── 与 rank 0 握手：nested_pipeline 在积分前先 Barrier，再发 READY ──
        print("[Rank 1] Waiting at MPI Barrier for rank 0...", flush=True)
        comm.Barrier()
        ready_msg = comm.recv(source=0, tag=READY_TAG)
        print(f"[Rank 1] Ready signal received: {ready_msg}", flush=True)

        # ── 接收 step=0（需在 sim 构造前，用于 WRF 三维场初始条件）────────
        print("[Rank 1] Receiving WRF step=0...", flush=True)
        received_step0 = recv_next_wrf_step(comm, step=0, source=0)
        print(f"[Rank 1] step=0 received, t_wrf={received_step0.get('__wrf_time_s', 0):.1f}s", flush=True)

        # ── 用 WRF step=0 三维场构造 LBM 初始条件函数 ──────────────────
        ic = make_wrf_initial_condition(
            received_step0, static_ctx, ref,
            lbm_origin_x=lbm_origin_x,
            lbm_origin_y=lbm_origin_y,
            domain_length_x=lbm_cfg.domain_length_x,
            domain_length_y=lbm_cfg.domain_length_y,
            domain_length_z=lbm_cfg.domain_length_z,
        )

        # ── 温度场（日循环）：θ 初始场 + 初始局部参考温度 ─────────────────
        thermal = None
        ic_ad = ic_ref = None
        hfx_scale = None
        if LBM_THERMAL:
            thermal = ThermalConfig(theta_ref=LBM_THETA_REF, dirichlet_bc=LBM_AD_DIRICHLET_BC)
            ic_ad, ic_ref = make_wrf_initial_temperature(
                received_step0, static_ctx, ref,
                lbm_origin_x, lbm_origin_y,
                lbm_cfg.domain_length_x, lbm_cfg.domain_length_y, lbm_cfg.domain_length_z,
            )
            # HFX [W/m^2] → kinematic [K m/s] → lattice (× dt/dx), as VF stores it
            hfx_scale = lbm_cfg.dt / (lbm_cfg.dx * LBM_RHO_CP)

        # ── VirtualFluids 仿真构造（含 WRF IC）───────────────────────────
        print("[Rank 1] Setting up VirtualFluids simulation...", flush=True)
        print(f"[Rank 1] LBM open-boundary mode: {LBM_BC_MODE}  thermal: {LBM_THERMAL}", flush=True)
        sim, para, lbm_cfg = setup_vf_simulation(lbm_cfg, initial_condition_fn=ic,
                                                 bc_mode=LBM_BC_MODE,
                                                 thermal=thermal,
                                                 initial_condition_ad_fn=ic_ad,
                                                 initial_reference_temperature_fn=ic_ref,
                                                 print_files=LBM_FULL_VTK_OUTPUT,
                                                 probe_every_steps=LBM_PROBE_EVERY_STEPS)
        if LBM_FULL_VTK_OUTPUT:
            # 只输出最开始的几次全场 VTK（abl.cfg 的 tOut 远大于模拟时长，否则一次都不输出）
            para.set_timestep_out(LBM_OUTPUT_EVERY_STEPS)
            lbm_output_until_step = LBM_OUTPUT_EVERY_STEPS * LBM_OUTPUT_COUNT
            print(f"[Rank 1] LBM VTK output: every {LBM_OUTPUT_EVERY_STEPS} steps, "
                  f"{LBM_OUTPUT_COUNT} outputs, off after step {lbm_output_until_step} "
                  f"(t≈{lbm_output_until_step * lbm_cfg.dt:.1f}s) -> {lbm_cfg.output_path}", flush=True)
        else:
            # 不写全场 VTK：timestep_out=0 使 Simulation 跳过文件输出分支（只剩探针）
            para.set_timestep_out(0)
            lbm_output_until_step = None
            print("[Rank 1] LBM full-field VTK output: off (plane probe only)", flush=True)
        sim.init_timers()
        print("[Rank 1] VirtualFluids simulation initialized.", flush=True)

        # ── bc_coords / sl_coords，加 WRF 坐标系 offset ─────────────────
        bc_coords_lbm = np.asarray(
            get_bc_coordinates(sim, LBM_BC_MODE, level=0), dtype=np.float64
        )
        bc_coords_wrf = _add_xy_offset(bc_coords_lbm)

        # sl_coords 同样加 offset（用于 hfx 双线性插值）
        if hasattr(sim, "get_surface_layer_bc_coordinates"):
            sl_raw = np.asarray(
                sim.get_surface_layer_bc_coordinates(level=0), dtype=np.float64
            )
            sl_coords_wrf = _add_xy_offset(sl_raw)[:, :2]  # 只取 x/y
        else:
            sl_coords_wrf = None

        # 温度 Dirichlet 边界节点（温度场开启时；节点集合与速度边界分开取）
        ad_coords_wrf = None
        if LBM_THERMAL:
            ad_coords_wrf = _add_xy_offset(
                np.asarray(sim.get_ad_dirichlet_bc_coordinates(level=0), dtype=np.float64))
            print(f"[Rank 1] AD Dirichlet nodes: {ad_coords_wrf.shape[0]}  "
                  f"surface-layer nodes: {0 if sl_coords_wrf is None else sl_coords_wrf.shape[0]}  "
                  f"hfx_scale={hfx_scale:.3e}", flush=True)

        # ── 接收 step=1，建立双缓冲 ───────────────────────────────────
        print("[Rank 1] Receiving WRF step=1...", flush=True)
        received_step1 = recv_next_wrf_step(comm, step=1, source=0)
        print(f"[Rank 1] step=1 received, t_wrf={received_step1.get('__wrf_time_s', 0):.1f}s", flush=True)

        db = init_lbm_from_wrf(
            sim, received_step0, received_step1, static_ctx, ref,
            bc_coords_override=bc_coords_wrf,
            sl_coords_override=sl_coords_wrf,
            bc_mode=LBM_BC_MODE,
            ad_coords_override=ad_coords_wrf,
        )
        print(f"[Rank 1] Double buffer ready: t=[{db.t0:.1f}, {db.t1:.1f}]s", flush=True)

        # 诊断：首帧温度边界 vs 初始场（同一位置应几乎相同）+ 首帧地面热通量
        if LBM_THERMAL and db.buf[0].theta_ad is not None:
            from lbm_adapter.lbm_init import diagnose_theta_bc
            diagnose_theta_bc(ad_coords_wrf, db.buf[0].theta_ad,
                              label=f"init t_wrf={db.t0:.1f}s", ic_ad=ic_ad,
                              origin_xy=(lbm_origin_x, lbm_origin_y), hfx=db.buf[0].hfx)

        n_bc = bc_coords_wrf.shape[0]
        print(f"[Rank 1] BC nodes: {n_bc}", flush=True)

        # ── 主循环 ───────────────────────────────────────────────────────
        loop = WRFRecvLoop(
            comm=comm,
            static_ctx=static_ctx,
            ref=ref,
            bc_coords=bc_coords_wrf,
            next_step=2,       # step=0,1 已消耗
            source=0,
            sl_coords=sl_coords_wrf,
            ad_coords=ad_coords_wrf,
        )

        print(f"[Rank 1] Entering main loop: n_steps={lbm_cfg.n_steps}", flush=True)
        stats = run_lbm_coupled(
            sim=sim,
            db=db,
            recv_loop=loop,
            n_steps=lbm_cfg.n_steps,
            dt_lbm=lbm_cfg.dt,
            level=0,
            start_step=1,
            para=para,
            output_until_step=lbm_output_until_step,
            bc_mode=LBM_BC_MODE,
            hfx_scale=hfx_scale,
        )

        sim.finalize()
        print(f"[Rank 1] Simulation done. "
              f"steps={stats.n_steps_run}  advances={stats.n_advances}  "
              f"t_lbm_end={stats.t_lbm_end:.2f}s", flush=True)

        # ── EOF 握手 ─────────────────────────────────────────────────────
        EOF_TAG = 999998
        eof = comm.recv(source=0, tag=EOF_TAG)
        assert eof.get("eof") is True, f"Expected EOF, got {eof}"
        ack_msg = {"ack": True, "steps_run": stats.n_steps_run,
                   "advances": stats.n_advances}
        comm.send(ack_msg, dest=0, tag=EOF_TAG)
        print(f"[Rank 1] EOF ACK sent: {ack_msg}", flush=True)

        print("[Rank 1] Done", flush=True)

    else:
        raise RuntimeError(f"Unexpected rank {rank} (only 0 and 1 supported)")

except Exception as e:
    safe_abort(e)
