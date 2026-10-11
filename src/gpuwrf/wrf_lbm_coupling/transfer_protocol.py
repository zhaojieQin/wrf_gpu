"""Transfer protocol: static + dynamic MPI packet helpers for WRF→LBM coupling.

Separates the one-time static initialisation handshake from the per-step
dynamic frame send.  Wraps the existing WRFMPISender so call-sites in
nested_pipeline.py only need to change the subdomain dict they hand in.

Protocol Specification
======================

Static Packet (sent once, tag = STATIC_TAG = 999998)
-----------------------------------------------------
Transmitted via synchronous blocking send (comm.send) at coupling init.
Contains grid metadata and base-state fields needed for LBM initialization.

Fields (all numpy arrays, serialized via LBMStaticContext.to_dict()):
  phb             : (nz+1, ny, nx)  float32  m² s⁻²   base-state geopotential
  terrain_height  : (ny, nx)        float32  m         terrain height MSL
  eta_levels      : (nz+1,)         float64  -         η-levels (ZNW: 1.0→0.0)
  dx_wrf          : scalar          float    m         WRF d03 grid spacing x
  dy_wrf          : scalar          float    m         WRF d03 grid spacing y
  nx              : scalar          int      -         horizontal grid size x
  ny              : scalar          int      -         horizontal grid size y
  nz              : scalar          int      -         vertical levels (mass)
  u_ref           : scalar          float    m s⁻¹     LBM velocity ratio
  theta_ref       : scalar          float    K         LBM reference temperature

Coordinate system: WRF Arakawa-C staggered grid on d03 domain.
  - phb on w-levels (nz+1 vertical, unstaggered horizontal)
  - terrain_height at mass points (ny, nx)
  - eta_levels descend from 1.0 (model top) to 0.0 (surface)

Dynamic Packet (sent every coupling step, dt_d03 = 5 s)
--------------------------------------------------------
Transmitted via non-blocking WRFMPISender.send_subdomain() with tag encoding.
Contains per-step velocity/theta fields interpolated to LBM inlet nodes.

Fields (all numpy arrays, float32):
  t_wrf  : (1,)     float32  s       WRF simulation time from forecast start
  u      : (N,)     float32  -       u-velocity in LBM lattice units (dimensionless)
  v      : (N,)     float32  -       v-velocity in LBM lattice units (dimensionless)
  w      : (N,)     float32  -       w-velocity in LBM lattice units (dimensionless)
  theta  : (N,)     float32  K       potential temperature deviation (θ − 300)

where N = number of LBM inlet boundary nodes.

Unit conversions applied:
  u/v/w: physical (m s⁻¹) → lattice via  u_LB = u_phys / u_ref
  theta: WRF perturbation (θ−300 K) passed through unchanged
         (equals LBM concentration when T_ref_LBM = 300 K)

Stagger status:
  All dynamic fields are interpolated to LBM node positions (arbitrary locations)
  and are thus destaggered. No Arakawa-C stagger remains in the output.

Field order:
  Dynamic fields are sent in fixed order: t_wrf, u, v, w, theta.
  Receiver must parse fields in this exact sequence for tag-based reassembly.

Usage in nested_pipeline.py
----------------------------
::

    from gpuwrf.wrf_lbm_coupling.transfer_protocol import (
        TransferProtocol,
        STATIC_TAG,
    )

    # -- once, at coupling init (before the time loop) --
    proto = TransferProtocol(mpi_sender, ctx)
    proto.send_static()

    # -- every coupling step --
    proto.send_dynamic(t_wrf, lbm_frame, coupling_step)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gpuwrf.wrf_lbm_coupling.lbm_frame import LBMFrame, LBMStaticContext

STATIC_TAG: int = 999998   # reserved tag for the static handshake packet

# Dynamic packet field order (must match receiver)
_DYNAMIC_FIELDS = ("t_wrf", "u", "v", "w", "theta")


class TransferProtocol:
    """Manages the two-phase WRF→LBM transfer protocol.

    Phase 1 – Static handshake (once):
        Sends grid metadata and base-state fields that do not change.
        Receiver can use these to pre-allocate buffers and build the
        eta→z mapping on the LBM side.

    Phase 2 – Dynamic steps (every dt_d03):
        Sends per-step velocity/theta/geopotential via the existing
        WRFMPISender (non-blocking Isend, preserves tag encoding).

    Parameters
    ----------
    mpi_sender : WRFMPISender  existing sender object (already initialised)
    ctx        : LBMStaticContext  static context produced at coupling init
    """

    def __init__(self, mpi_sender, ctx: "LBMStaticContext") -> None:
        self._sender = mpi_sender
        self._ctx = ctx
        self._static_sent = False

    def send_static(self) -> None:
        """Send the one-time static packet to the LBM receiver.

        Blocks until the receiver acknowledges (comm.send, synchronous).
        Must be called once before the first dynamic send.
        """
        if self._static_sent:
            return

        payload = self._ctx.to_dict()
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[耦合静态包] {ts} Sending static packet "
            f"(phb shape={self._ctx.phb.shape}, "
            f"eta_levels shape={self._ctx.eta_levels.shape})",
            flush=True,
        )
        self._sender.comm.send(payload, dest=self._sender.dest_rank, tag=STATIC_TAG)
        self._static_sent = True
        print(f"[耦合静态包] Static packet sent.", flush=True)

    def send_dynamic(
        self,
        t_wrf: float,
        frame: "LBMFrame",
        coupling_step: int,
    ) -> int:
        """Send one dynamic frame to the LBM receiver.

        The frame must be an LBMFrame produced by build_lbm_frame().
        Fields are sent as float32 numpy arrays via WRFMPISender.send_subdomain().

        Parameters
        ----------
        t_wrf         : float  WRF simulation time (s from forecast start)
        frame         : LBMFrame  pre-built LBM data frame
        coupling_step : int  monotonically increasing coupling counter
                        (used for MPI tag encoding inside WRFMPISender)

        Returns
        -------
        n_fields : int  number of fields sent
        """
        if not self._static_sent:
            raise RuntimeError(
                "send_dynamic() called before send_static(). "
                "Call send_static() once at the start of the coupling."
            )

        # Build the dynamic dict.  Keys must be stable (ordered) so that
        # the receiver's tag-based field reassembly is deterministic.
        dynamic = {
            "t_wrf": np.array([t_wrf], dtype=np.float32),
            "u": frame.u.astype(np.float32),
            "v": frame.v.astype(np.float32),
            "w": frame.w.astype(np.float32),
            "theta": frame.theta.astype(np.float32),
        }

        return self._sender.send_subdomain(dynamic, coupling_step)
