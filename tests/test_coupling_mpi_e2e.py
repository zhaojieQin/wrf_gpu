"""End-to-end MPI coupling test: WRF→LBM field transfer validation.

Run on cluster with:
    srun --ntasks=2 python -m pytest tests/test_coupling_mpi_e2e.py -v -s

Or as a standalone script:
    srun --ntasks=2 python tests/test_coupling_mpi_e2e.py

Skip conditions (auto):
  - mpi4py not installed
  - wrfinput_d03 not found at SWIFT_RUN_DIR (cluster-only fixture)
  - MPI communicator size != 2

Environment variables:
  SWIFT_RUN_DIR  Path to the SWiFT run directory containing wrfinput_d03.
                 Default: /home/zhaqi746/code/wrf_gpu/examples/SWiFT
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# MPI availability guard — must happen at import time so pytest can collect
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI as _MPI
    _HAS_MPI4PY = True
except ImportError:
    _HAS_MPI4PY = False

pytestmark = pytest.mark.skipif(
    not _HAS_MPI4PY, reason="mpi4py not available — skipping MPI e2e test"
)

# ---------------------------------------------------------------------------
# MPI rank / size (module-level, safe even without mpi4py due to guard above)
# ---------------------------------------------------------------------------
if _HAS_MPI4PY:
    _comm = _MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
    # GPU assignment before any CUDA import; CUDA_VISIBLE_DEVICES must be set
    # before cupy/jax see the device list, so do it at import time.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(_rank))
else:
    _comm = None
    _rank = 0
    _size = 1

# ---------------------------------------------------------------------------
# Fixture path configuration
# ---------------------------------------------------------------------------
_DEFAULT_SWIFT_DIR = "/home/zhaqi746/code/wrf_gpu/examples/SWiFT"
_SWIFT_RUN_DIR = os.environ.get("SWIFT_RUN_DIR", _DEFAULT_SWIFT_DIR)
_WRFINPUT_D03 = Path(_SWIFT_RUN_DIR) / "wrfinput_d03"
_SWIFT_AVAILABLE = _WRFINPUT_D03.exists()

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
_COUPLING_STEP = 0
_EOF_TAG = 999998
_FIELDS_3D = ["u", "v", "w", "theta", "ph"]
_FIELDS_2D = ["ustar", "roughness_m", "psim"]
_ALL_FIELDS = _FIELDS_3D + _FIELDS_2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_preconditions():
    """Skip with a clear message if the test cannot run in this environment."""
    if not _HAS_MPI4PY:
        pytest.skip("mpi4py not installed")
    if not _SWIFT_AVAILABLE:
        pytest.skip(
            f"wrfinput_d03 not found at {_WRFINPUT_D03} — "
            "set SWIFT_RUN_DIR to the SWiFT run directory on this cluster"
        )
    if _size != 2:
        pytest.skip(
            f"This test requires exactly 2 MPI ranks (got size={_size}). "
            "Launch with: srun --ntasks=2 python -m pytest tests/test_coupling_mpi_e2e.py"
        )


# ---------------------------------------------------------------------------
# Rank 0 logic: build carry → compute MOST → validate locally → send
# ---------------------------------------------------------------------------

def _rank0_build_and_send():
    """Load wrfinput_d03, compute surface diagnostics, send 8 fields to rank 1."""
    from gpuwrf.integration.d02_replay import build_replay_case
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.wrf_lbm_coupling.surface_diagnostics import compute_surface_diagnostics
    # WRFMPISender lives at the wrf_gpu project root (one level above tests/)
    _project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(_project_root))
    from wrf_mpi_sender import WRFMPISender

    print(f"\n[Rank 0] Loading replay case: run_dir={_SWIFT_RUN_DIR}, domain=d03", flush=True)
    case = build_replay_case(
        _SWIFT_RUN_DIR,
        domain="d03",
        load_lateral_boundaries=False,   # standalone: wrfinput IC only, no wrfbdy
    )
    carry = initial_operational_carry(case.state, base_state=case.base_state)
    grid = case.grid
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    print(f"[Rank 0] Grid: nz={nz}, ny={ny}, nx={nx}", flush=True)

    # Recompute surface-layer diagnostics from the loaded state (Path 2 MOST)
    most_fields = compute_surface_diagnostics(carry, grid)
    print(
        f"[Rank 0] MOST ustar: shape={most_fields.ustar.shape} "
        f"range=[{most_fields.ustar.min():.4f}, {most_fields.ustar.max():.4f}] m/s",
        flush=True,
    )
    print(
        f"[Rank 0] MOST roughness_m: shape={most_fields.roughness_m.shape} "
        f"range=[{most_fields.roughness_m.min():.6f}, {most_fields.roughness_m.max():.4f}] m",
        flush=True,
    )
    print(
        f"[Rank 0] MOST psim: shape={most_fields.psim.shape} "
        f"range=[{most_fields.psim.min():.4f}, {most_fields.psim.max():.4f}]",
        flush=True,
    )

    state = carry.state
    subdomain = {
        "u":           state.u,
        "v":           state.v,
        "w":           state.w,
        "theta":       state.theta,
        "ph":          state.ph,              # ph_total = PHB + PH' (m²/s²)
        "ustar":       most_fields.ustar,     # numpy (ny, nx)
        "roughness_m": most_fields.roughness_m,
        "psim":        most_fields.psim,
    }

    # Validate locally before putting anything on the wire
    _validate_rank0(subdomain, nz, ny, nx)

    sender = WRFMPISender(dest_rank=1)
    n_sent = sender.send_subdomain(subdomain, coupling_step=_COUPLING_STEP)
    sender.wait_all()
    print(f"[Rank 0] Sent {n_sent} fields via MPI", flush=True)

    # Handshake EOF so rank 1 knows the transfer is complete
    _comm.send({"eof": True}, dest=1, tag=_EOF_TAG)
    ack = _comm.recv(source=1, tag=_EOF_TAG)
    print(f"[Rank 0] EOF handshake complete: {ack}", flush=True)

    return subdomain


def _validate_rank0(subdomain, nz, ny, nx):
    """Pre-send validation: shapes, NaN, and physical-range sanity checks."""
    import jax

    expected_shapes = {
        "u":           (nz,     ny,     nx + 1),
        "v":           (nz,     ny + 1, nx    ),
        "w":           (nz + 1, ny,     nx    ),
        "theta":       (nz,     ny,     nx    ),
        "ph":          (nz + 1, ny,     nx    ),
        "ustar":       (ny,     nx            ),
        "roughness_m": (ny,     nx            ),
        "psim":        (ny,     nx            ),
    }

    for name, expected in expected_shapes.items():
        arr = subdomain[name]
        actual = tuple(arr.shape)
        assert actual == expected, (
            f"[Rank 0] Shape mismatch for '{name}': expected {expected}, got {actual}"
        )
        arr_np = np.asarray(jax.device_get(arr)) if hasattr(arr, "device") else np.asarray(arr)
        n_nan = int(np.isnan(arr_np).sum())
        assert n_nan == 0, f"[Rank 0] '{name}' contains {n_nan} NaN values before send"
        print(
            f"[Rank 0]   {name:12s}: shape={actual} "
            f"range=[{float(arr_np.min()):.4f}, {float(arr_np.max()):.4f}]",
            flush=True,
        )

    # Physical range guards
    ustar = np.asarray(jax.device_get(subdomain["ustar"]))
    assert ustar.min() > 0.0,  f"[Rank 0] ustar must be positive (min={ustar.min():.6f})"
    assert ustar.max() < 10.0, f"[Rank 0] ustar out of physical range (max={ustar.max():.4f})"

    roughness = np.asarray(jax.device_get(subdomain["roughness_m"]))
    assert roughness.min() > 0.0, f"[Rank 0] roughness_m must be positive (min={roughness.min():.8f})"

    theta = np.asarray(jax.device_get(subdomain["theta"]))
    assert theta.min() > 150.0, f"[Rank 0] theta implausibly cold (min={theta.min():.1f} K)"
    assert theta.max() < 500.0, f"[Rank 0] theta implausibly hot (max={theta.max():.1f} K)"

    ph = np.asarray(jax.device_get(subdomain["ph"]))
    # Geopotential ph = PHB + PH' is O(10^4–10^5) m²/s²; surface > 0
    assert ph.min() > 0.0, f"[Rank 0] ph_total must be positive (min={ph.min():.1f} m²/s²)"

    print("[Rank 0] Pre-send validation: PASSED", flush=True)


# ---------------------------------------------------------------------------
# Rank 1 logic: receive → validate
# ---------------------------------------------------------------------------

def _rank1_receive_and_validate():
    """Receive fields from rank 0 and validate shapes, NaN, dtype, and ranges."""
    step = _COUPLING_STEP
    meta_tag = step * 1000

    # Receive metadata dict (pickle)
    metadata = _comm.recv(source=0, tag=meta_tag)
    print(f"\n[Rank 1] Received metadata for {len(metadata)} fields: {list(metadata.keys())}", flush=True)

    # Mirror the barrier that WRFMPISender inserts after metadata send
    _comm.Barrier()

    received = {}
    for field_idx, field_name in enumerate(metadata):
        info = metadata[field_name]
        shape = tuple(info["shape"])
        dtype = np.dtype(info["dtype"])
        buf = np.empty(shape, dtype=dtype)
        data_tag = step * 1000 + field_idx + 1
        _comm.Recv(buf, source=0, tag=data_tag)
        received[field_name] = buf
        print(
            f"[Rank 1]   Received '{field_name}': shape={shape} dtype={dtype} "
            f"range=[{buf.min():.4f}, {buf.max():.4f}]",
            flush=True,
        )

    # EOF handshake
    eof_msg = _comm.recv(source=0, tag=_EOF_TAG)
    assert eof_msg.get("eof") is True, f"[Rank 1] Unexpected EOF message: {eof_msg}"
    _comm.send({"ack": True}, dest=0, tag=_EOF_TAG)
    print("[Rank 1] EOF handshake complete", flush=True)

    _validate_rank1(received)
    return received


def _validate_rank1(received: dict):
    """Post-receive assertions on all 8 fields."""
    # 1. All expected fields present
    for name in _ALL_FIELDS:
        assert name in received, f"[Rank 1] Missing field '{name}'"

    # 2. dtype: everything should arrive as float64 (sender uses cp.asnumpy on float64 arrays)
    for name, arr in received.items():
        assert arr.dtype == np.float64, (
            f"[Rank 1] '{name}' has dtype {arr.dtype}, expected float64"
        )

    # 3. NaN check
    for name, arr in received.items():
        n_nan = int(np.isnan(arr).sum())
        assert n_nan == 0, f"[Rank 1] '{name}' contains {n_nan} NaN values after receive"

    # 4. Dimensionality
    for name in _FIELDS_3D:
        assert received[name].ndim == 3, f"[Rank 1] '{name}' should be 3D, got {received[name].ndim}D"
    for name in _FIELDS_2D:
        assert received[name].ndim == 2, f"[Rank 1] '{name}' should be 2D, got {received[name].ndim}D"

    # 5. MOST 2D fields share the same (ny, nx) shape
    ustar_shape = received["ustar"].shape
    assert received["roughness_m"].shape == ustar_shape, "roughness_m / ustar shape mismatch"
    assert received["psim"].shape == ustar_shape,        "psim / ustar shape mismatch"

    # 6. Physical range checks
    ustar = received["ustar"]
    assert ustar.min() > 0.0,  f"[Rank 1] ustar must be positive (min={ustar.min():.6f})"
    assert ustar.max() < 10.0, f"[Rank 1] ustar out of physical range (max={ustar.max():.4f})"

    roughness = received["roughness_m"]
    assert roughness.min() > 0.0, f"[Rank 1] roughness_m must be positive (min={roughness.min():.8f})"

    theta = received["theta"]
    assert theta.min() > 150.0, f"[Rank 1] theta implausibly cold (min={theta.min():.1f} K)"
    assert theta.max() < 500.0, f"[Rank 1] theta implausibly hot (max={theta.max():.1f} K)"

    ph = received["ph"]
    assert ph.min() > 0.0, f"[Rank 1] ph_total must be positive (min={ph.min():.1f} m²/s²)"

    # 7. u/v/w shapes are consistent with each other (stagger check)
    u, v, w = received["u"], received["v"], received["w"]
    nz_u, ny_u, nx_u = u.shape
    nz_v, ny_v, nx_v = v.shape
    nz_w, ny_w, nx_w = w.shape
    # u: x-stagger → nx_u = nx+1
    # v: y-stagger → ny_v = ny+1
    # w: z-stagger → nz_w = nz+1
    assert nx_u == nx_v + 1,  f"[Rank 1] u/v x-stagger mismatch: u.nx={nx_u}, v.nx={nx_v}"
    assert ny_v == ny_u + 1,  f"[Rank 1] u/v y-stagger mismatch: u.ny={ny_u}, v.ny={ny_v}"
    assert nz_w == nz_u + 1,  f"[Rank 1] u/w z-stagger mismatch: u.nz={nz_u}, w.nz={nz_w}"

    print(f"\n[Rank 1] Post-receive validation: PASSED", flush=True)
    print(f"[Rank 1] Field summary (ny={ny_u}, nx={nx_v}):", flush=True)
    for name, arr in received.items():
        print(f"  {name:12s}: shape={arr.shape} range=[{arr.min():.4f}, {arr.max():.4f}]", flush=True)


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------

class TestCouplingMpiE2E:
    """End-to-end MPI coupling: WRF→LBM 8-field transfer (requires --ntasks=2)."""

    def test_mpi_field_transfer(self):
        """Send 5 WRF state fields + 3 MOST surface fields from rank 0 to rank 1."""
        _check_preconditions()

        if _rank == 0:
            subdomain = _rank0_build_and_send()
            assert len(subdomain) == len(_ALL_FIELDS), (
                f"Expected {len(_ALL_FIELDS)} fields, got {len(subdomain)}"
            )

        elif _rank == 1:
            received = _rank1_receive_and_validate()
            assert len(received) == len(_ALL_FIELDS), (
                f"Expected {len(_ALL_FIELDS)} received fields, got {len(received)}"
            )

        else:
            pytest.skip(f"Unexpected rank {_rank}")

        # Synchronize both ranks so pytest sees a clean exit on both sides
        _comm.Barrier()


# ---------------------------------------------------------------------------
# Standalone entry point (srun --ntasks=2 python tests/test_coupling_mpi_e2e.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _HAS_MPI4PY:
        print("ERROR: mpi4py not available", file=sys.stderr)
        sys.exit(1)
    if not _SWIFT_AVAILABLE:
        if _rank == 0:
            print(f"ERROR: wrfinput_d03 not found at {_WRFINPUT_D03}", file=sys.stderr)
            print(f"       Set SWIFT_RUN_DIR to the SWiFT run directory", file=sys.stderr)
        sys.exit(1)
    if _size != 2:
        if _rank == 0:
            print(f"ERROR: requires --ntasks=2 (got size={_size})", file=sys.stderr)
        sys.exit(1)

    try:
        if _rank == 0:
            _rank0_build_and_send()
            print("\n[Rank 0] SUCCESS", flush=True)
        else:
            _rank1_receive_and_validate()
            print("\n[Rank 1] SUCCESS", flush=True)
        _comm.Barrier()
        if _rank == 0:
            print("\n[MPI e2e] Both ranks passed.", flush=True)
    except Exception as exc:
        import traceback
        print(f"\n[Rank {_rank}] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        _comm.Abort(1)
