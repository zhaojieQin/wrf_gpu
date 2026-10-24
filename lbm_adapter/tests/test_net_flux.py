import numpy as np

from lbm_adapter.net_flux import NetFluxCorrector

DX = 10.0
LX, LY, LZ = 100.0, 80.0, 50.0


def _bc_coords():
    """Open-face nodes half a cell outside [0, L] (lateral faces + roof, no edges)."""
    xs = -0.5 * DX + DX * np.arange(1, int(LX / DX) + 1)
    ys = -0.5 * DX + DX * np.arange(1, int(LY / DX) + 1)
    zs = -0.5 * DX + DX * np.arange(1, int(LZ / DX) + 1)
    pts = []
    for x in (-0.5 * DX, LX + 0.5 * DX):
        pts += [(x, y, z) for y in ys for z in zs]
    for y in (-0.5 * DX, LY + 0.5 * DX):
        pts += [(x, y, z) for x in xs for z in zs]
    pts += [(x, y, LZ + 0.5 * DX) for x in xs for y in ys]
    return np.asarray(pts)


def test_net_flux_removed_and_tangential_kept():
    c = _bc_coords()
    corr = NetFluxCorrector(c, DX, LX, LY, LZ, diag_every_s=None)
    rng = np.random.default_rng(0)
    u = 5.0 + rng.normal(size=len(c))
    v = 2.0 + rng.normal(size=len(c))
    w = 0.3 * rng.normal(size=len(c))
    u2, v2, w2 = corr.apply(u, v, w)
    un2 = u2 * corr.nx + v2 * corr.ny + w2 * corr.nz
    assert abs(np.sum(un2)) < 1e-9 * len(c)
    # tangential components unchanged
    lateral_x = corr.face <= 1
    assert np.allclose(v2[lateral_x], v[lateral_x]) and np.allclose(w2[lateral_x], w[lateral_x])
    roof = corr.face == 4
    assert np.allclose(u2[roof], u[roof]) and np.allclose(v2[roof], v[roof])


def test_balanced_field_unchanged():
    c = _bc_coords()
    corr = NetFluxCorrector(c, DX, LX, LY, LZ, diag_every_s=None)
    u = np.full(len(c), 4.0)
    v = np.full(len(c), -1.0)
    w = np.zeros(len(c))
    u2, v2, w2 = corr.apply(u, v, w)
    assert np.allclose(u2, u) and np.allclose(v2, v) and np.allclose(w2, w)
