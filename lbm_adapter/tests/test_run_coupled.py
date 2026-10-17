"""任务9单元测试：lbm_adapter/run_coupled.py

纯 numpy，无 VirtualFluids，无 MPI（用 Mock 替代）。

运行方法（在 wrf_gpu/ 目录下）:
    python -m pytest lbm_adapter/tests/test_run_coupled.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lbm_adapter.run_coupled import run_lbm_coupled, CouplingStats
from lbm_adapter.lbm_init import WRFDoubleBuffer, WRFFrame
from lbm_adapter.recv_loop import WRFRecvLoop
from lbm_adapter.wrf_to_lbm import WRFStaticContext
from lbm_adapter.unit_convert import LBMReference

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_BC = 5
DT_LBM = 0.01  # s


def _frame(t, val=0.0, n=N_BC):
    arr = lambda v: np.full(n, v)
    return WRFFrame(t, arr(val), arr(val), arr(val), arr(val * 0.1), None)


def _make_db(t0=0.0, t1=2.0):
    return WRFDoubleBuffer(buf=[_frame(t0, 0.0), _frame(t1, 1.0)])


def _make_sim():
    sim = MagicMock()
    return sim


def _make_recv_loop(n_advances=0, advance_at_steps=None):
    """Mock WRFRecvLoop.maybe_advance.

    advance_at_steps: set of lbm_step values at which maybe_advance returns True.
    """
    loop = MagicMock(spec=WRFRecvLoop)
    advance_at_steps = advance_at_steps or set()

    call_count = {"n": 0}

    def _maybe_advance(t_lbm, db):
        step = round(t_lbm / DT_LBM)
        if step in advance_at_steps:
            # Simulate advancing db
            new_t = db.t1 + 2.0
            db.advance(_frame(new_t, 2.0))
            return True
        return False

    loop.maybe_advance.side_effect = _maybe_advance
    loop.eof_received = False
    return loop


# ---------------------------------------------------------------------------
# TestCouplingStats
# ---------------------------------------------------------------------------

class TestCouplingStats:
    def test_defaults(self):
        s = CouplingStats()
        assert s.n_steps_run == 0
        assert s.n_advances == 0
        assert s.t_lbm_end == 0.0


# ---------------------------------------------------------------------------
# TestRunLbmCoupled
# ---------------------------------------------------------------------------

class TestRunLbmCoupled:
    def test_returns_coupling_stats(self):
        sim = _make_sim()
        db = _make_db()
        loop = _make_recv_loop()
        stats = run_lbm_coupled(sim, db, loop, n_steps=5, dt_lbm=DT_LBM)
        assert isinstance(stats, CouplingStats)

    def test_n_steps_run(self):
        sim = _make_sim()
        db = _make_db(t0=0.0, t1=100.0)  # large t1 so no advance needed
        loop = _make_recv_loop()
        stats = run_lbm_coupled(sim, db, loop, n_steps=10, dt_lbm=DT_LBM)
        assert stats.n_steps_run == 10

    def test_calculate_timestep_called_n_times(self):
        sim = _make_sim()
        db = _make_db(t0=0.0, t1=100.0)
        loop = _make_recv_loop()
        run_lbm_coupled(sim, db, loop, n_steps=7, dt_lbm=DT_LBM)
        assert sim.calculate_timestep.call_count == 7

    def test_update_velocity_bc_called_n_times(self):
        sim = _make_sim()
        db = _make_db(t0=0.0, t1=100.0)
        loop = _make_recv_loop()
        run_lbm_coupled(sim, db, loop, n_steps=5, dt_lbm=DT_LBM)
        assert sim.update_velocity_bc.call_count == 5

    def test_calculate_timestep_step_sequence(self):
        """calculate_timestep 应按 1, 2, 3, … 顺序调用（start_step=1）。"""
        sim = _make_sim()
        db = _make_db(t0=0.0, t1=100.0)
        loop = _make_recv_loop()
        run_lbm_coupled(sim, db, loop, n_steps=3, dt_lbm=DT_LBM, start_step=1)
        calls = [c[0][0] for c in sim.calculate_timestep.call_args_list]
        assert calls == [1, 2, 3]

    def test_custom_start_step(self):
        sim = _make_sim()
        db = _make_db(t0=0.0, t1=100.0)
        loop = _make_recv_loop()
        run_lbm_coupled(sim, db, loop, n_steps=3, dt_lbm=DT_LBM, start_step=100)
        calls = [c[0][0] for c in sim.calculate_timestep.call_args_list]
        assert calls == [100, 101, 102]

    def test_t_lbm_end(self):
        """t_lbm_end = last_step * dt_lbm。"""
        sim = _make_sim()
        db = _make_db(t0=0.0, t1=100.0)
        loop = _make_recv_loop()
        n = 5
        stats = run_lbm_coupled(sim, db, loop, n_steps=n, dt_lbm=DT_LBM, start_step=1)
        assert stats.t_lbm_end == pytest.approx(n * DT_LBM)

    def test_n_advances_counted(self):
        """換页发生次数应反映在 stats.n_advances。"""
        sim = _make_sim()
        db = _make_db(t0=0.0, t1=0.02)   # t1=0.02s → step 2 crosses it
        # step 2: t_lbm=0.02 → needs_advance → mock returns True
        loop = _make_recv_loop(advance_at_steps={2})
        stats = run_lbm_coupled(sim, db, loop, n_steps=5, dt_lbm=DT_LBM, start_step=1)
        assert stats.n_advances == 1

    def test_zero_steps(self):
        sim = _make_sim()
        db = _make_db()
        loop = _make_recv_loop()
        stats = run_lbm_coupled(sim, db, loop, n_steps=0, dt_lbm=DT_LBM)
        assert stats.n_steps_run == 0
        sim.calculate_timestep.assert_not_called()
        sim.update_velocity_bc.assert_not_called()

    def test_maybe_advance_called_every_step(self):
        sim = _make_sim()
        db = _make_db(t0=0.0, t1=100.0)
        loop = _make_recv_loop()
        run_lbm_coupled(sim, db, loop, n_steps=4, dt_lbm=DT_LBM)
        assert loop.maybe_advance.call_count == 4

    def test_level_passed_to_update_velocity_bc(self):
        sim = _make_sim()
        db = _make_db(t0=0.0, t1=100.0)
        loop = _make_recv_loop()
        run_lbm_coupled(sim, db, loop, n_steps=1, dt_lbm=DT_LBM, level=2)
        _, kwargs = sim.update_velocity_bc.call_args
        assert kwargs.get("level", sim.update_velocity_bc.call_args[0][-1] if len(sim.update_velocity_bc.call_args[0]) > 3 else 0) == 2 \
            or sim.update_velocity_bc.call_args[1].get("level") == 2 \
            or sim.update_velocity_bc.call_args[0][-1] == 2

    def test_update_surface_layer_bc_called_when_hfx_present(self):
        """sim 有 update_surface_layer_heat_flux 且 hfx 存在时应被调用。"""
        sim = _make_sim()
        hfx = np.full(N_BC, 50.0)
        f0 = WRFFrame(0.0, np.zeros(N_BC), np.zeros(N_BC), np.zeros(N_BC), np.zeros(N_BC), hfx)
        f1 = WRFFrame(100.0, np.ones(N_BC), np.ones(N_BC), np.ones(N_BC), np.ones(N_BC), hfx)
        db = WRFDoubleBuffer(buf=[f0, f1])
        loop = _make_recv_loop()
        run_lbm_coupled(sim, db, loop, n_steps=3, dt_lbm=DT_LBM)
        assert sim.update_surface_layer_heat_flux.call_count == 3

    def test_update_surface_layer_bc_not_called_when_hfx_none(self):
        """hfx 为 None 时不调用 update_surface_layer_heat_flux。"""
        sim = _make_sim()
        db = _make_db(t0=0.0, t1=100.0)  # _frame uses hfx=None
        loop = _make_recv_loop()
        run_lbm_coupled(sim, db, loop, n_steps=3, dt_lbm=DT_LBM)
        sim.update_surface_layer_heat_flux.assert_not_called()

    def test_update_surface_layer_bc_not_called_when_sim_lacks_method(self):
        """sim 没有 update_surface_layer_heat_flux 时不应崩溃。"""
        sim = MagicMock(spec=["calculate_timestep", "update_velocity_bc"])
        hfx = np.full(N_BC, 50.0)
        f0 = WRFFrame(0.0, np.zeros(N_BC), np.zeros(N_BC), np.zeros(N_BC), np.zeros(N_BC), hfx)
        f1 = WRFFrame(100.0, np.ones(N_BC), np.ones(N_BC), np.ones(N_BC), np.ones(N_BC), hfx)
        db = WRFDoubleBuffer(buf=[f0, f1])
        loop = _make_recv_loop()
        # Should not raise
        stats = run_lbm_coupled(sim, db, loop, n_steps=2, dt_lbm=DT_LBM)
        assert stats.n_steps_run == 2
        """BC 速度应是插值值（不是 buf[0] 或 buf[1] 原值）。"""
        sim = _make_sim()
        # buf[0].u = 0.0, buf[1].u = 1.0; at t=0.5*dt between them alpha=0.5→u=0.5
        f0 = WRFFrame(0.0, np.zeros(N_BC), np.zeros(N_BC), np.zeros(N_BC), np.zeros(N_BC), None)
        f1 = WRFFrame(0.02, np.ones(N_BC), np.ones(N_BC), np.ones(N_BC), np.ones(N_BC), None)
        db = WRFDoubleBuffer(buf=[f0, f1])
        loop = _make_recv_loop()
        # step 1: t_lbm = 0.01 → alpha = 0.01/0.02 = 0.5 → u_interp = 0.5
        run_lbm_coupled(sim, db, loop, n_steps=1, dt_lbm=DT_LBM, start_step=1)
        u_called = sim.update_velocity_bc.call_args[0][0]
        np.testing.assert_allclose(u_called, 0.5, rtol=1e-10)


class TestOutputLimit:
    def test_output_switched_off_after_until_step(self):
        sim, para = _make_sim(), MagicMock()
        run_lbm_coupled(sim, _make_db(), _make_recv_loop(), n_steps=10, dt_lbm=0.1,
                        para=para, output_until_step=4)
        para.set_timestep_out.assert_called_once_with(0)
        # switched off right after step 4 was computed (output for step 4 still written)
        assert sim.calculate_timestep.call_args_list[3][0][0] == 4

    def test_no_output_change_by_default(self):
        para = MagicMock()
        run_lbm_coupled(_make_sim(), _make_db(), _make_recv_loop(), n_steps=5, dt_lbm=0.1)
        para.set_timestep_out.assert_not_called()
