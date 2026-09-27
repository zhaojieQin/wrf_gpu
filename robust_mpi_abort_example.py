#!/usr/bin/env python3
"""健壮的 MPI_Abort error handler 示例"""

import sys
import traceback

def safe_mpi_abort(exit_code=1, error_msg=None):
    """安全地终止 MPI 程序

    如果 MPI 已初始化，用 MPI.COMM_WORLD.Abort()
    否则用 sys.exit()
    """
    if error_msg:
        print(f"[FATAL] {error_msg}", file=sys.stderr, flush=True)

    try:
        from mpi4py import MPI
        if MPI.Is_initialized() and not MPI.Is_finalized():
            print(f"[MPI_Abort] Terminating all ranks with code {exit_code}",
                  file=sys.stderr, flush=True)
            MPI.COMM_WORLD.Abort(exit_code)
        else:
            print(f"[sys.exit] MPI not active, exiting with code {exit_code}",
                  file=sys.stderr, flush=True)
            sys.exit(exit_code)
    except ImportError:
        # mpi4py 未安装或 import 失败
        print(f"[sys.exit] MPI not available, exiting with code {exit_code}",
              file=sys.stderr, flush=True)
        sys.exit(exit_code)
    except Exception as e:
        # MPI.Is_initialized() 本身失败（极端情况）
        print(f"[sys.exit] MPI check failed ({e}), exiting with code {exit_code}",
              file=sys.stderr, flush=True)
        sys.exit(exit_code)


def main_rank_0():
    """Rank 0: WRF 发送端"""
    try:
        from gpuwrf.integration.nested_pipeline import execute_nested_pipeline
        # ... WRF 运行逻辑
        print("[Rank 0] WRF completed", flush=True)

    except KeyboardInterrupt:
        print("[Rank 0] User interrupted", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"[Rank 0 Error] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        safe_mpi_abort(1, f"Rank 0 fatal error: {e}")


def main_rank_1():
    """Rank 1: LBM 接收端"""
    try:
        # ... LBM 运行逻辑
        print("[Rank 1] LBM completed", flush=True)

    except KeyboardInterrupt:
        print("[Rank 1] User interrupted", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"[Rank 1 Error] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        safe_mpi_abort(1, f"Rank 1 fatal error: {e}")


if __name__ == '__main__':
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            main_rank_0()
        elif rank == 1:
            main_rank_1()

    except Exception as e:
        # import 或初始化阶段失败
        print(f"[Init Error] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        safe_mpi_abort(1, "Initialization failed")
