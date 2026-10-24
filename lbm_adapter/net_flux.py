"""Net-flux correction of the LBM open-boundary velocities.

The WRF velocities interpolated onto the five open faces (four lateral faces + roof) are not
discretely divergence-free on the LBM grid: the volume flux through the faces does not sum to
zero, so mass accumulates in the domain (mean density drifts up) and the imbalance drives
spurious convergence / sinking near the inflow faces. Each step the outward volume flux

    Q = Σ_i (u_i · n_i) · dx²         (m³/s, > 0: net outflow)

is computed over all open-boundary nodes and removed by a uniform normal-velocity correction

    u_i ← u_i − (Q / A) · n_i,   A = n_bc · dx²,

which leaves the tangential components untouched and makes the net flux exactly zero. The
ground (surface-layer BC) is not an open face and is not touched.
"""
from __future__ import annotations

import numpy as np

_FACE_NAMES = ("MX", "PX", "MY", "PY", "PZ")
_NORMALS = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0], [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])


class NetFluxCorrector:
    """Zero-net-flux correction for the open-boundary velocity nodes.

    Parameters
    ----------
    bc_coords : (n_bc, 3) open-boundary velocity nodes in LBM coordinates (m), in the order of
                ``db.interp`` u/v/w. Boundary nodes sit half a cell outside [0, L].
    dx        : LBM grid spacing (m); every node represents a face area dx².
    lx, ly, lz: LBM domain lengths (m).
    diag_every_s : print interval of the flux diagnostics (s of LBM time); None → silent.
    """

    def __init__(self, bc_coords: np.ndarray, dx: float, lx: float, ly: float, lz: float,
                 diag_every_s: float | None = 300.0):
        c = np.asarray(bc_coords, dtype=np.float64)
        x, y, z = c[:, 0], c[:, 1], c[:, 2]
        # one face per node; lateral faces take precedence over the roof at the top edges
        face = np.select([x < 0.0, x > lx, y < 0.0, y > ly, z > lz], [0, 1, 2, 3, 4], -1)
        if np.any(face < 0):
            raise ValueError(f"NetFluxCorrector: {int(np.sum(face < 0))} BC nodes are not on an open face")
        self.face = face
        self.nx, self.ny, self.nz = (_NORMALS[face, k] for k in range(3))
        self.area = float(dx) * float(dx)
        self.a_total = self.area * face.size
        self.diag_every_s = diag_every_s
        self._next_diag = -np.inf
        counts = np.bincount(face, minlength=5)
        print("[net-flux] open-boundary nodes per face: "
              + ", ".join(f"{n}={int(k)}" for n, k in zip(_FACE_NAMES, counts))
              + f"  total area={self.a_total:.4g} m²", flush=True)

    def apply(self, u, v, w, t_lbm: float | None = None):
        """Return (u, v, w) with zero net volume flux through the open faces."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        un = u * self.nx + v * self.ny + w * self.nz
        q = self.area * float(np.sum(un))
        delta = -q / self.a_total

        if self.diag_every_s is not None and t_lbm is not None and t_lbm >= self._next_diag:
            per_face = np.bincount(self.face, weights=un, minlength=5) * self.area
            print(f"[net-flux] t={t_lbm:.1f}s  net outflow Q={q:+.4g} m³/s "
                  f"(inflow {-np.sum(np.minimum(un, 0)) * self.area:.4g}, "
                  f"outflow {np.sum(np.maximum(un, 0)) * self.area:.4g})  "
                  f"correction Δu_n={delta:+.4f} m/s  per face: "
                  + " ".join(f"{n}={f:+.3g}" for n, f in zip(_FACE_NAMES, per_face)),
                  flush=True)
            self._next_diag = t_lbm + self.diag_every_s

        return u + delta * self.nx, v + delta * self.ny, w + delta * self.nz
