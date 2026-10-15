"""任务8单元测试：lbm_adapter/recv_loop.py

纯 numpy，无 MPI（用 Mock 替代）。

运行方法（在 wrf_gpu/ 目录下）:
    python -m pytest lbm_adapter/tests/test_recv_loop.py -v
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lbm_adapter.recv_loop import recv_next_wrf_step, WRFRecvLoop
from lbm_adapter.lbm_init import WRFDoubleBuffer, WRFFrame
from lbm_adapter.wrf_to_lbm import WRFStaticContext
from lbm_adapter.unit_convert import LBMReference

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

G = 9.81
NZ, NY, NX = 10, 4, 4
DZ = 500.0
N_BC = 5


def _make_ph(nz=NZ, ny=NY, nx=NX, dz=DZ):
    k = np.arange(nz + 1, dtype=np.float64)
    return (G * k[:, None, None] * dz * np.ones((nz + 1, ny, nx))).astype(np.float32)


def _make_metadata(t_wrf: float, nz=NZ, ny=NY, nx=NX) -> dict:
    """Minimal metadata dict matching MPI protocol."""
    return {
        "__wrf_time_s": t_wrf,
        "u":     {"shape": (nz,     ny, nx), "dtype": "float32", "size_mb": 0.1},
        "v":     {"shape": (nz,     ny, nx), "dtype": "float32", "size_mb": 0.1},
        "w":     {"shape": (nz + 1, ny, nx), "dtype": "float32", "size_mb": 0.1},
        "theta": {"shape": (nz,     ny, nx), "dtype": "float32", "size_mb": 0.1},
        "ph":    {"shape": (nz + 1, ny, nx), "dtype": "float32", "size_mb": 0.1},
    }


def _make_comm_mock(t_wrf: float, step: int, nz=NZ, ny=NY, nx=NX,
                    u_val=5.0, theta_val=310.0, hfx_val=None) -> MagicMock:
    """Mock MPI comm that returns realistic data for one step."""
    ph = _make_ph(nz, ny, nx)
    arrays = {
        "u":     u_val  * np.ones((nz,     ny, nx), dtype=np.float32),
        "v":    -2.0    * np.ones((nz,     ny, nx), dtype=np.float32),
        "w":     0.1    * np.ones((nz + 1, ny, nx), dtype=np.float32),
        "theta": theta_val * np.ones((nz,  ny, nx), dtype=np.float32),
        "ph":    ph,
    }

    metadata = _make_metadata(t_wrf, nz, ny, nx)
    if hfx_val is not None:
        arrays["hfx"] = hfx_val * np.ones((ny, nx), dtype=np.float32)
        metadata["hfx"] = {"shape": (ny, nx), "dtype": "float32", "size_mb": 0.0}

    comm = MagicMock()
    comm.recv.return_value = metadata

    # Recv fills the provided buffer in-place
    field_order = [k for k in metadata if not k.startswith("__")]

    def _Recv(buf, source, tag):
        expected_base = step * 1000
        field_idx = tag - expected_base - 1
        if 0 <= field_idx < len(field_order):
            fname = field_order[field_idx]
            buf[:] = arrays[fname]

    comm.Recv.side_effect = _Recv
    return comm


def _make_static_ctx():
    return WRFStaticContext(
        terrain_height=np.zeros((NY, NX)),
        dx_wrf=1000.0,
        dy_wrf=1000.0,
    )


def _make_ref():
    return LBMReference(dx_lbm=1.0, dt_lbm=0.01, theta_ref=300.0)


def _bc_coords(n=N_BC):
    return np.column_stack([
        np.zeros(n),
        np.zeros(n),
        np.linspace(500.0, 4000.0, n),
    ])


def _make_db(t0=0.0, t1=2.0, n=N_BC):
    f = lambda t: WRFFrame(t, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), None)
    return WRFDoubleBuffer(buf=[f(t0), f(t1)])


# ---------------------------------------------------------------------------
# TestRecvNextWrfStep
# ---------------------------------------------------------------------------

class TestRecvNextWrfStep:
    def test_returns_dict(self):
        comm = _make_comm_mock(t_wrf=0.0, step=0)
        result = recv_next_wrf_step(comm, step=0)
        assert isinstance(result, dict)

    def test_control_key_present(self):
        comm = _make_comm_mock(t_wrf=4.0, step=0)
        result = recv_next_wrf_step(comm, step=0)
        assert "__wrf_time_s" in result
        assert result["__wrf_time_s"] == pytest.approx(4.0)

    def test_array_fields_present(self):
        comm = _make_comm_mock(t_wrf=0.0, step=0)
        result = recv_next_wrf_step(comm, step=0)
        for key in ("u", "v", "w", "theta", "ph"):
            assert key in result, key

    def test_array_shapes(self):
        comm = _make_comm_mock(t_wrf=0.0, step=0)
        result = recv_next_wrf_step(comm, step=0)
        assert result["u"].shape     == (NZ, NY, NX)
        assert result["w"].shape     == (NZ + 1, NY, NX)
        assert result["ph"].shape    == (NZ + 1, NY, NX)

    def test_meta_tag_correct(self):
        """comm.recv 应以 step*1000 为 tag 调用。"""
        comm = _make_comm_mock(t_wrf=0.0, step=3)
        recv_next_wrf_step(comm, step=3)
        comm.recv.assert_called_once_with(source=0, tag=3000)

    def test_data_tags_sequential(self):
        """第一个字段 tag = step*1000+1，第二个 = step*1000+2，…"""
        comm = _make_comm_mock(t_wrf=0.0, step=2)
        recv_next_wrf_step(comm, step=2)
        # 5 个字段 → tags 2001..2005
        recv_tags = [kw["tag"] for _, kw in comm.Recv.call_args_list]
        assert recv_tags == [2001, 2002, 2003, 2004, 2005]

    def test_custom_source(self):
        comm = _make_comm_mock(t_wrf=0.0, step=0)
        recv_next_wrf_step(comm, step=0, source=2)
        comm.recv.assert_called_once_with(source=2, tag=0)

    def test_u_values_correct(self):
        comm = _make_comm_mock(t_wrf=0.0, step=0, u_val=7.0)
        result = recv_next_wrf_step(comm, step=0)
        np.testing.assert_allclose(result["u"], 7.0, rtol=1e-5)

    def test_ph_realistic_range(self):
        """ph_total 应在 O(10^4-10^5)。"""
        comm = _make_comm_mock(t_wrf=0.0, step=0)
        result = recv_next_wrf_step(comm, step=0)
        assert result["ph"].max() > 1e4


    def test_barrier_after_metadata(self):
        """默认在 metadata 之后、字段 Recv 之前调用 Barrier（与 WRFMPISender 配对）。"""
        comm = _make_comm_mock(t_wrf=0.0, step=0)
        recv_next_wrf_step(comm, step=0)
        comm.Barrier.assert_called_once()
        names = [c[0] for c in comm.mock_calls]
        assert names.index("recv") < names.index("Barrier") < names.index("Recv")

    def test_no_barrier_when_disabled(self):
        comm = _make_comm_mock(t_wrf=0.0, step=0)
        recv_next_wrf_step(comm, step=0, barrier_after_meta=False)
        comm.Barrier.assert_not_called()


# ---------------------------------------------------------------------------
# TestWRFRecvLoop
# ---------------------------------------------------------------------------

class TestWRFRecvLoop:
    def _make_loop(self, next_step=2):
        comm = MagicMock()  # will be replaced per test
        return WRFRecvLoop(
            comm=comm,
            static_ctx=_make_static_ctx(),
            ref=_make_ref(),
            bc_coords=_bc_coords(),
            next_step=next_step,
        )

    def test_no_advance_when_not_needed(self):
        loop = self._make_loop()
        db = _make_db(t0=0.0, t1=2.0)
        advanced = loop.maybe_advance(1.0, db)  # t_lbm < t1 → no advance
        assert advanced is False
        loop.comm.recv.assert_not_called()

    def test_advance_when_needed(self):
        """t_lbm >= t1 → advance one step, comm.recv called once."""
        # Build a comm that returns step=2 data
        comm = _make_comm_mock(t_wrf=4.0, step=2)
        loop = WRFRecvLoop(
            comm=comm,
            static_ctx=_make_static_ctx(),
            ref=_make_ref(),
            bc_coords=_bc_coords(),
            next_step=2,
        )
        db = _make_db(t0=0.0, t1=2.0)
        advanced = loop.maybe_advance(2.0, db)
        assert advanced is True
        comm.recv.assert_called_once()

    def test_next_step_incremented_after_advance(self):
        comm = _make_comm_mock(t_wrf=4.0, step=2)
        loop = WRFRecvLoop(
            comm=comm,
            static_ctx=_make_static_ctx(),
            ref=_make_ref(),
            bc_coords=_bc_coords(),
            next_step=2,
        )
        db = _make_db(t0=0.0, t1=2.0)
        loop.maybe_advance(2.0, db)
        assert loop.next_step == 3

    def test_db_advanced_after_advance(self):
        """After advance, db.t0 should be old t1, db.t1 should be new frame's t_wrf."""
        comm = _make_comm_mock(t_wrf=4.0, step=2)
        loop = WRFRecvLoop(
            comm=comm,
            static_ctx=_make_static_ctx(),
            ref=_make_ref(),
            bc_coords=_bc_coords(),
            next_step=2,
        )
        db = _make_db(t0=0.0, t1=2.0)
        loop.maybe_advance(2.0, db)
        assert db.t0 == pytest.approx(2.0)
        assert db.t1 == pytest.approx(4.0)

    def test_no_advance_when_eof(self):
        """eof_received=True → never advance even if t_lbm > t1."""
        loop = self._make_loop()
        loop.eof_received = True
        db = _make_db(t0=0.0, t1=2.0)
        advanced = loop.maybe_advance(99.0, db)
        assert advanced is False
        loop.comm.recv.assert_not_called()

    def test_advance_skips_out_of_order_frame(self):
        """若新帧 t_wrf <= db.t1，跳过并发出警告，不换页。"""
        # new frame timestamp (0.5) <= db.t1 (2.0) → out of order
        comm = _make_comm_mock(t_wrf=0.5, step=2)
        loop = WRFRecvLoop(
            comm=comm,
            static_ctx=_make_static_ctx(),
            ref=_make_ref(),
            bc_coords=_bc_coords(),
            next_step=2,
        )
        db = _make_db(t0=0.0, t1=2.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            advanced = loop.maybe_advance(2.0, db)
        assert advanced is False
        assert any("out-of-order" in str(wi.message) or "Skipping" in str(wi.message)
                   for wi in w)
        # step should still increment so we don't retry the same bad frame
        assert loop.next_step == 3

    def test_default_next_step_is_2(self):
        loop = WRFRecvLoop(
            comm=MagicMock(),
            static_ctx=_make_static_ctx(),
            ref=_make_ref(),
            bc_coords=_bc_coords(),
        )
        assert loop.next_step == 2

    def test_eof_initially_false(self):
        loop = self._make_loop()
        assert loop.eof_received is False

    def test_hfx_propagated_with_sl_coords(self):
        """传入 sl_coords → step≥2 的新帧 hfx 为 (n_sl,)，不再是 None。"""
        n_sl = 4
        comm = _make_comm_mock(t_wrf=4.0, step=2, hfx_val=30.0)
        loop = WRFRecvLoop(
            comm=comm,
            static_ctx=_make_static_ctx(),
            ref=_make_ref(),
            bc_coords=_bc_coords(),
            next_step=2,
            sl_coords=np.column_stack([np.zeros(n_sl), np.zeros(n_sl)]),
        )
        db = _make_db(t0=0.0, t1=2.0)
        assert loop.maybe_advance(2.0, db) is True
        assert db.buf[1].hfx is not None
        assert db.buf[1].hfx.shape == (n_sl,)
        np.testing.assert_allclose(db.buf[1].hfx, 30.0, rtol=1e-5)

    def test_barrier_flag_forwarded(self):
        comm = _make_comm_mock(t_wrf=4.0, step=2)
        loop = WRFRecvLoop(
            comm=comm,
            static_ctx=_make_static_ctx(),
            ref=_make_ref(),
            bc_coords=_bc_coords(),
            next_step=2,
            barrier_after_meta=False,
        )
        loop.maybe_advance(2.0, _make_db(t0=0.0, t1=2.0))
        comm.Barrier.assert_not_called()
