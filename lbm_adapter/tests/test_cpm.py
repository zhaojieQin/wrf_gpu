"""CPU tests for lbm_adapter.cpm (mocked VF simulation)."""
import numpy as np
import pytest

from lbm_adapter.cpm import (
    CPMConfig, CPMPerturber, diagnose_zi, inflow_weight, theta_pm_from_ec, theta_pm_from_rim,
)

DX = 10.0
LX, LY, LZ = 800.0, 600.0, 300.0


class FakeSim:
    def __init__(self):
        self.setup = None
        self.applied = []

    def set_cell_perturbation_cells(self, table, x0, y0, z0, cdx, cdy, dz, n_cells, level=0):
        self.setup = dict(table=np.array(table), x0=x0, y0=y0, z0=z0, cdx=cdx, n=n_cells)
        return int((table >= 0).sum()) * 64

    def apply_cell_perturbation(self, theta, u=None, v=None):
        self.applied.append((np.array(theta), None if u is None else np.array(u), None if v is None else np.array(v)))


def _face_nodes():
    """Lateral face nodes at the LBM node positions (−dx/2 + i dx)."""
    xs = -0.5 * DX + DX * np.arange(int(LX / DX) + 2)
    ys = -0.5 * DX + DX * np.arange(int(LY / DX) + 2)
    zs = -0.5 * DX + DX * np.arange(int(LZ / DX) + 2)
    pts = []
    for z in zs[1:-1]:
        for y in ys[1:-1]:
            pts += [(xs[0], y, z), (xs[-1], y, z)]
        for x in xs[1:-1]:
            pts += [(x, ys[0], z), (x, ys[-1], z)]
    return np.array(pts)


def _make(cfg=None, **kw):
    sim = FakeSim()
    coords = _face_nodes()
    p = CPMPerturber(sim, cfg or CPMConfig(z_max_m=200.0, **kw), DX, LX, LY, LZ,
                     bc_coords=coords, ad_coords=coords, theta_ref=300.0, velocity_ratio=50.0, verbose=False)
    return sim, p, coords


def test_inflow_weight_limits():
    assert inflow_weight(-50.0, 0.5) == pytest.approx(1.0)
    assert inflow_weight(50.0, 0.5) == pytest.approx(0.0)
    assert inflow_weight(0.0, 0.5) == pytest.approx(0.5)


def test_theta_pm_from_ec():
    assert theta_pm_from_ec(10.0, 0.2, 1004.5) == pytest.approx(100.0 / (1004.5 * 0.2))


def test_theta_pm_from_rim_nonnegative():
    z = np.arange(10)
    tpm = theta_pm_from_rim(300.0 + 0.01 * z, 0.1 * z, 0 * z, 10.0, -1.0)
    assert np.all(tpm >= 0) and tpm[5] > 0


def test_diagnose_zi_convective_and_stable():
    z = np.arange(5.0, 1000.0, 10.0)
    th_conv = np.where(z < 500, 300.0, 300.0 + 0.05 * (z - 500))
    assert diagnose_zi(z, th_conv, "convective", 0.014, 0.003) == pytest.approx(500, abs=20)
    assert np.isinf(diagnose_zi(z, np.full_like(z, 300.0), "convective", 0.014, 0.003))
    th_st = 300.0 + 20.0 * (1 - np.exp(-z / 100.0))       # gradient decays with height
    zi = diagnose_zi(z, th_st, "stable", 0.014, 0.003)
    assert 200 < zi < 500


def test_layout_rows_offset_and_levels():
    sim, p, _ = _make()
    cfg = p.cfg
    t = sim.setup["table"]
    assert t.shape == (p.k_max + 1, p.nyc, p.nxc)
    assert np.all(t[: p.k_min] == -1)                       # skipped levels
    assert p.k_min == 1 + cfg.skip_levels
    assert p.x0 >= cfg.offset_nodes * DX                    # first cell edge inside the boundary
    interior = t[p.k_min, cfg.rows:-cfg.rows, cfg.rows:-cfg.rows]
    assert np.all(interior == -1)                           # only perimeter rows
    ids = t[t >= 0]
    assert ids.size == p.n_cells and np.unique(ids).size == p.n_cells


def test_westerly_inflow_perturbs_west_only():
    sim, p, coords = _make()
    u = np.full(len(coords), 5.0)        # wind towards +x: MX is inflow, PX outflow
    v = np.zeros(len(coords))
    theta = np.zeros(len(coords))
    assert p.maybe_refresh(0.0, u, v, theta_ad=theta, hfx=np.array([100.0]))
    th, up, vp = sim.applied[-1]
    assert up is None and vp is None
    col = p.cell_col
    west = p.col_faces["MX"][col] & ~p.col_faces["MY"][col] & ~p.col_faces["PY"][col]
    east = p.col_faces["PX"][col] & ~p.col_faces["MY"][col] & ~p.col_faces["PY"][col]
    assert np.abs(th[west]).max() > 0
    assert np.abs(th[east]).max() < 1e-3 * np.abs(th[west]).max()
    tpm = theta_pm_from_ec(5.0, p.cfg.ec, p.cfg.cp)
    assert np.abs(th).max() <= tpm + 1e-6


def test_refresh_timing_and_reproducibility():
    sim, p, coords = _make()
    u = np.full(len(coords), 4.0)
    v = np.zeros(len(coords))
    assert p.maybe_refresh(0.0, u, v)
    t_next = p.next_refresh_t
    expected = p.cfg.gamma * np.sqrt(2) * p.cell_dx / 4.0
    assert t_next == pytest.approx(np.clip(expected, p.cfg.tp_min_s, p.cfg.tp_max_s))
    assert not p.maybe_refresh(t_next * 0.5, u, v)
    assert p.maybe_refresh(t_next, u, v)
    sim2, p2, _ = _make()
    p2.maybe_refresh(0.0, u, v)
    np.testing.assert_array_equal(sim.applied[0][0], sim2.applied[0][0])


def test_levels_above_zcp_unperturbed():
    sim, p, coords = _make(CPMConfig(z_max_m=200.0))
    p.cfg.z_max_m = 100.0                 # z_CP limited by z_max at refresh time
    u = np.full(len(coords), 5.0)
    p.maybe_refresh(0.0, u, np.zeros(len(coords)))
    th = sim.applied[-1][0]
    assert np.all(th[p.cell_z > 100.0] == 0.0)


def test_momentum_option_zero_mean():
    sim, p, coords = _make(CPMConfig(z_max_m=200.0, momentum=True))
    u = np.full(len(coords), 5.0)
    p.maybe_refresh(0.0, u, np.zeros(len(coords)))
    _, up, vp = sim.applied[-1]
    assert up is not None and vp is not None
    assert abs(float(up.mean())) < 1e-6 and abs(float(vp.mean())) < 1e-6
