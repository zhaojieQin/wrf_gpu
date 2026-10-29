import numpy as np
import pytest

from lbm_adapter.relaxation import FACE_BITS, RelaxationConfig, RelaxationZone
from lbm_adapter.wrf_to_lbm import WRFStaticContext

DX = 10.0
LX, LY, LZ = 200.0, 160.0, 100.0
G = 9.81


class FakeSim:
    """Mimics VF BoundaryRelaxationZone node selection (fluid nodes at −dx/2 + i·dx, i = 1..n)."""

    def set_relaxation_zone(self, x_min, x_max, y_min, y_max, z_max, width, top_width, faces, level=0):
        xs = -0.5 * DX + DX * np.arange(1, int(LX / DX) + 1)
        ys = -0.5 * DX + DX * np.arange(1, int(LY / DX) + 1)
        zs = -0.5 * DX + DX * np.arange(1, int(LZ / DX) + 1)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        ramp = lambda d, W: np.where((W > 0) & (d < W), np.cos(np.pi * np.maximum(d, 0) / (2 * W)) ** 2, 0.0)  # noqa: E731
        w = np.zeros_like(X)
        if faces & FACE_BITS["MX"]:
            w = np.maximum(w, ramp(X - x_min, width))
        if faces & FACE_BITS["PX"]:
            w = np.maximum(w, ramp(x_max - X, width))
        if faces & FACE_BITS["MY"]:
            w = np.maximum(w, ramp(Y - y_min, width))
        if faces & FACE_BITS["PY"]:
            w = np.maximum(w, ramp(y_max - Y, width))
        if faces & FACE_BITS["PZ"]:
            w = np.maximum(w, ramp(z_max - Z, top_width))
        m = w > 0
        rng = np.random.default_rng(1)
        order = rng.permutation(int(m.sum()))                 # arbitrary node order, as on the GPU
        self.coords = np.stack([X[m], Y[m], Z[m]], axis=1)[order]
        self.weights = w[m][order]
        self.pushed = []
        self.applied = []
        return len(self.weights)

    def get_relaxation_zone_nodes(self):
        return self.coords, self.weights

    def push_relaxation_targets(self, vx, vy, vz, theta=None):
        self.pushed.append((np.asarray(vx), np.asarray(vy), np.asarray(vz), None if theta is None else np.asarray(theta)))

    def apply_relaxation(self, alpha, rate, relax_velocity=True, relax_temperature=True):
        self.applied.append((alpha, rate, relax_velocity, relax_temperature))


def _wrf_frame(nx=8, ny=8, nz=12, dz=20.0, u0=5.0, v0=2.0, th0=300.0, gamma=0.01):
    zw = dz * np.arange(nz + 1)
    zm = 0.5 * (zw[1:] + zw[:-1])
    rec = {
        "u": np.full((nz, ny, nx + 1), u0),
        "v": np.full((nz, ny + 1, nx), v0),
        "w": np.zeros((nz + 1, ny, nx)),
        "theta": np.broadcast_to((th0 + gamma * zm)[:, None, None], (nz, ny, nx)).copy(),
        "ph": np.broadcast_to((G * zw)[:, None, None], (nz + 1, ny, nx)).copy(),
        "__wrf_time_s": 0.0,
    }
    ctx = WRFStaticContext(terrain_height=np.zeros((ny, nx)), dx_wrf=100.0, dy_wrf=100.0, z0_mean=0.1)
    return rec, ctx


def _zone(faces=("MX", "PX", "MY", "PY"), top=0):
    rec, ctx = _wrf_frame()
    sim = FakeSim()
    z = RelaxationZone(sim, RelaxationConfig(width_nodes=3, tau_s=20.0, faces=faces, top_width_nodes=top),
                       DX, LX, LY, LZ, dt_lbm=0.08, static_ctx=ctx, origin_xy=(150.0, 150.0),
                       theta_ref=300.0, velocity_ratio=125.0)
    return z, sim, rec


def test_every_zone_node_belongs_to_one_strip():
    z, sim, _ = _zone()
    n_strip = sum(s["idx"].size for s in z.strips)
    assert n_strip == z.n
    all_idx = np.concatenate([s["idx"] for s in z.strips])
    assert np.unique(all_idx).size == z.n
    assert z.rate == pytest.approx(0.08 / 20.0)


def test_targets_match_wrf_and_units():
    z, sim, rec = _zone()
    z.push_frame(rec)
    vx, vy, vz, th = sim.pushed[-1]
    assert vx.shape == (z.n,) and th.shape == (z.n,)
    assert np.allclose(vx, 5.0 / 125.0, atol=1e-6)            # lattice units
    assert np.allclose(vy, 2.0 / 125.0, atol=1e-6)
    assert np.allclose(vz, 0.0, atol=1e-6)
    # θ − θ_ref = 0.01 z (inside the WRF column range; below the lowest mass level θ is clamped)
    zc = sim.coords[:, 2]
    ok = zc >= 10.0
    assert np.allclose(th[ok], 0.01 * zc[ok], atol=2e-3)


def test_roof_zone_and_alpha():
    z, sim, rec = _zone(faces=("MX", "PZ"), top=2)
    assert {s["name"] for s in z.strips} == {"W", "T"}
    z.push_frame(rec)
    z.push_frame(rec)

    class DB:
        t0, t1 = 100.0, 105.0
    z.apply(102.5, DB)
    alpha, rate, rv, rt = sim.applied[-1]
    assert alpha == pytest.approx(0.5) and rate == pytest.approx(0.004) and rv and rt
