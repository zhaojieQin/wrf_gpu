import numpy as np

from lbm_adapter.net_flux import NetFluxCorrector

DX = 10.0
LX, LY, LZ = 100.0, 80.0, 50.0


def _bc_coords():
    """VF-style entries: fluid nodes next to each face, blocks MX, PX, MY, PY, PZ; edge and
    corner nodes appear once per face they touch."""
    xs = -0.5 * DX + DX * np.arange(1, int(LX / DX) + 1)
    ys = -0.5 * DX + DX * np.arange(1, int(LY / DX) + 1)
    zs = -0.5 * DX + DX * np.arange(1, int(LZ / DX) + 1)
    pts, labels = [], []
    for f, x in ((0, xs[0]), (1, xs[-1])):
        pts += [(x, y, z) for y in ys for z in zs]
        labels += [f] * (len(ys) * len(zs))
    for f, y in ((2, ys[0]), (3, ys[-1])):
        pts += [(x, y, z) for x in xs for z in zs]
        labels += [f] * (len(xs) * len(zs))
    pts += [(x, y, zs[-1]) for x in xs for y in ys]
    labels += [4] * (len(xs) * len(ys))
    return np.asarray(pts), np.asarray(labels)


def test_faces_assigned_per_block_with_duplicated_edges():
    c, labels = _bc_coords()
    corr = NetFluxCorrector(c, DX, LX, LY, LZ, diag_every_s=None)
    assert np.array_equal(corr.face, labels)


def test_net_flux_removed_and_tangential_kept():
    c, _ = _bc_coords()
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
    c, _ = _bc_coords()
    corr = NetFluxCorrector(c, DX, LX, LY, LZ, diag_every_s=None)
    u = np.full(len(c), 4.0)
    v = np.full(len(c), -1.0)
    w = np.zeros(len(c))
    u2, v2, w2 = corr.apply(u, v, w)
    assert np.allclose(u2, u) and np.allclose(v2, v) and np.allclose(w2, w)
