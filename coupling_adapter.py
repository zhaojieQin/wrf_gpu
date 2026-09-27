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
        from coupling_config import HOURS, INTERVAL_SECONDS

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
                    'fields': ['u', 'v', 'theta', 'qv'],
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
        print("[Rank 1] Starting LBM MPI receiver...", flush=True)
        from coupling_config import HOURS, INTERVAL_SECONDS

        steps = int(HOURS * 3600 / INTERVAL_SECONDS)
        print(f"[Rank 1] Computed steps={steps} (HOURS={HOURS}, INTERVAL_SECONDS={INTERVAL_SECONDS})", flush=True)

        # Import and call main with computed steps
        import mock_receiver
        mock_receiver.main(steps=steps)

        print("[Rank 1] Done", flush=True)

    else:
        raise RuntimeError(f"Unexpected rank {rank} (only 0 and 1 supported)")

except Exception as e:
    safe_abort(e)
