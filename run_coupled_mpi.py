#!/usr/bin/env python3
"""WRF-LBM coupled run with MPI rank split.

This script runs WRF and LBM in a single MPI job with proper rank assignment.
- Rank 0: WRF GPU forecast
- Rank 1: LBM MPI receiver

Usage:
    srun --ntasks=2 python run_coupled_mpi.py
"""
import os
import sys
from pathlib import Path
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"[DEBUG] rank={rank}, size={size}", flush=True)

if size != 2:
    if rank == 0:
        print(f"[错误] 需要 2 个 MPI ranks，当前 size={size}", flush=True)
    sys.exit(1)

if rank == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("[Rank 0] Starting WRF GPU forecast...", flush=True)
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, execute_nested_pipeline
    config = NestedPipelineConfig(
        input_dir=Path('/home/zhaqi746/code/wrf_gpu/examples/SWiFT'),
        output_dir=Path('/home/zhaqi746/code/wrf_gpu/runs/swift_coupled_mpi'),
        proof_dir=Path('/home/zhaqi746/code/wrf_gpu/runs/swift_coupled_mpi/proofs'),
        hours=1,
        max_dom=3,
        coupling_config={
            'enabled': True,
            'interval_seconds': 10,
            'target_domain': 'd03',
            'dry_run': False,
            'mpi_dest_rank': 1,
            'subdomain_config': {
                'z_range': (0, -1),
                'y_range': (0, -1),
                'x_range': (0, -1),
                'fields': ['u', 'v', 'theta', 'qv'],
            },
        },
    )
    execute_nested_pipeline(config)
    print("[Rank 0] WRF forecast completed", flush=True)
    comm.Barrier()
    print("[Rank 0] Barrier done", flush=True)

elif rank == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print("[Rank 1] Starting LBM MPI receiver...", flush=True)
    sys.argv = [
        'mock_receiver.py',
        '--steps', '360',
        '--verify-wrf',
        '--write-netcdf',
        '--output-dir', './received_nc_swift'
    ]
    from mock_receiver import main
    main()
    print("[Rank 1] LBM receiver completed", flush=True)
