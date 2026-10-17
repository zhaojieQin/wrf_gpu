"""VirtualFluids 仿真初始化封装（任务10）。

封装 pyfluids GPU 仿真的构造流程，与 lbm_adapter 解耦。
coupling_adapter.py rank 1 分支直接调用：

    cfg = WRFLBMConfig.from_cfg_file(cfg_path, t_end_s=HOURS*3600,
                                     output_path=...)
    sim, para, cfg = setup_vf_simulation(cfg)
    sim.init_timers()

设计原则
--------
- 所有物理参数从 VF .cfg 文件读取，WRFLBMConfig 不硬编码物理量
- .cfg 是单一权威来源，与 VF 侧完全一致
- pyfluids 延迟导入（rank 1 才需要，rank 0 无此库）

.cfg 键名约定（与 abl.cfg / boundary_layer.cfg 一致）
------------------------------------------------------
BoundaryLayerHeight / boundaryLayerHeight  → domain_length_z (m)
NodesPerBoundaryLayerHeight / nz           → nodes_per_height
RoughnessLength / z0                       → z0 (m)
FrictionVelocity / u_star                  → u_star (m/s)
vonKarmanConstant                          → kappa
Viscosity / viscosity                      → viscosity (m²/s)
Ma                                         → mach
Path                                       → output_path (overridable)
tEnd                                       → t_end_s (overridable)
tOut                                       → t_out_s
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WRFLBMConfig:
    """LBM 域和物理参数配置，从 VF .cfg 文件读取。

    不要手动构造此类的物理参数字段。
    使用 WRFLBMConfig.from_cfg_file() 从 .cfg 文件创建。

    所有长度单位 m，时间单位 s，速度单位 m/s。
    """
    # 域尺寸（m）— 从 cfg 推导
    domain_length_x: float
    domain_length_y: float
    domain_length_z: float   # = BoundaryLayerHeight

    # 格距（m）— 从 BoundaryLayerHeight / NodesPerHeight 推导
    dx: float

    # 物理参数 — 直接从 .cfg 读
    mach: float
    u_ref_phys: float        # 0.5 × u_log(H)，与 VF 脚本约定一致
    z0: float
    u_star: float
    kappa: float
    viscosity: float

    # 时间控制（秒）
    t_end_s: float
    t_out_s: float

    # 路径
    output_path: Path
    config_path: Path        # 原始 .cfg 文件路径，传给 pyfluids Parameter

    def __post_init__(self):
        self.output_path = Path(self.output_path)
        self.config_path = Path(self.config_path)

    # ------------------------------------------------------------------
    # 工厂方法：从 VF .cfg 文件读取（权威来源）
    # ------------------------------------------------------------------

    @classmethod
    def from_cfg_file(
        cls,
        config_path: str | Path,
        t_end_s: float,
        output_path: str | Path | None = None,
        domain_length_x: float | None = None,
        domain_length_y: float | None = None,
    ) -> "WRFLBMConfig":
        """从 VirtualFluids .cfg 文件读取物理参数，构造 WRFLBMConfig。

        Parameters
        ----------
        config_path     : VF .cfg 文件路径（单一权威来源）
        t_end_s         : 总仿真时间（秒）= HOURS * 3600，由 coupling_config 提供
        output_path     : 输出目录。若为 None，使用 .cfg 中的 Path 键。
        domain_length_x : x 方向域长（m）。若为 None，用 6 × BoundaryLayerHeight。
        domain_length_y : y 方向域长（m）。若为 None，用 4 × BoundaryLayerHeight。

        从 .cfg 读取的键（兼容 abl.cfg 和 boundary_layer/configBoundaryLayer.cfg）
        -----------------------------------------------------------------------
        BoundaryLayerHeight 或 boundaryLayerHeight → domain_length_z
        NodesPerBoundaryLayerHeight 或 nz          → nodes_per_height → dx
        RoughnessLength 或 z0                      → z0
        FrictionVelocity 或 u_star                 → u_star
        vonKarmanConstant                          → kappa
        Viscosity 或 viscosity                     → viscosity
        Ma                                         → mach
        tOut                                       → t_out_s
        Path                                       → output_path（可覆盖）
        """
        from pyfluids import basics

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"VF config file not found: {config_path}")

        cfg_file = basics.ConfigurationFile()
        cfg_file.load(str(config_path))

        # --- 读取物理参数（兼容两种键名风格）---
        def _get_float(key1, key2, default=None):
            try:
                return cfg_file.get_float_value(key1)
            except Exception:
                pass
            try:
                return cfg_file.get_float_value(key2)
            except Exception:
                pass
            if default is not None:
                return default
            raise KeyError(f"Neither '{key1}' nor '{key2}' found in {config_path}")

        def _get_uint(key1, key2, default=None):
            try:
                return cfg_file.get_uint_value(key1)
            except Exception:
                pass
            try:
                return cfg_file.get_uint_value(key2)
            except Exception:
                pass
            if default is not None:
                return default
            raise KeyError(f"Neither '{key1}' nor '{key2}' found in {config_path}")

        H      = _get_float("BoundaryLayerHeight", "boundaryLayerHeight")
        nz     = _get_uint("NodesPerBoundaryLayerHeight", "nz")
        z0     = _get_float("RoughnessLength", "z0")
        u_star = _get_float("FrictionVelocity", "u_star")
        kappa  = _get_float("vonKarmanConstant", "vonKarmanConstant", default=0.4)
        visc   = _get_float("Viscosity", "viscosity", default=1.56e-5)
        mach   = _get_float("Ma", "Ma")
        t_out  = _get_float("tOut", "tOut", default=100.0)

        dx = H / nz

        # u_ref = 0.5 × log-law mean at top（与 boundary_layer.py 第94行约定一致）
        u_ref = 0.5 * u_star / kappa * math.log(H / z0)

        # 域尺寸默认值（与 boundary_layer.py: length = [6,4,1]*H 一致）
        lx = domain_length_x if domain_length_x is not None else 6.0 * H
        ly = domain_length_y if domain_length_y is not None else 4.0 * H

        # 输出路径：优先用参数传入，否则用 .cfg 里的 Path 键
        if output_path is None:
            try:
                out_str = cfg_file.get_string_value("Path")
                output_path = config_path.parent / out_str
            except Exception:
                output_path = config_path.parent / "output"
        output_path = Path(output_path)

        return cls(
            domain_length_x=lx,
            domain_length_y=ly,
            domain_length_z=H,
            dx=dx,
            mach=mach,
            u_ref_phys=u_ref,
            z0=z0,
            u_star=u_star,
            kappa=kappa,
            viscosity=visc,
            t_end_s=t_end_s,
            t_out_s=t_out,
            output_path=output_path,
            config_path=config_path,
        )

    # ------------------------------------------------------------------
    # 推导属性
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        """LBM 时间步长（秒）。dt = dx * Ma / (√3 * u_ref)。"""
        return self.dx * self.mach / (math.sqrt(3) * self.u_ref_phys)

    @property
    def velocity_ratio(self) -> float:
        """dx / dt（m/s），= para.get_velocity_ratio()。"""
        return self.dx / self.dt

    @property
    def velocity_LB(self) -> float:
        return self.u_ref_phys / self.velocity_ratio

    @property
    def viscosity_LB(self) -> float:
        return self.viscosity / (self.velocity_ratio * self.dx)

    @property
    def n_steps(self) -> int:
        return int(self.t_end_s / self.dt)

    @property
    def n_steps_out(self) -> int:
        return max(1, int(self.t_out_s / self.dt))


# ---------------------------------------------------------------------------
# VirtualFluids 仿真构造
# ---------------------------------------------------------------------------

def setup_vf_simulation(cfg: WRFLBMConfig, communicator=None,
                        initial_condition_fn=None, bc_mode: str = "freestream"):
    """构造并返回 VirtualFluids Simulation 对象。

    Parameters
    ----------
    cfg           : WRFLBMConfig（由 WRFLBMConfig.from_cfg_file() 创建）
    communicator  : pyfluids MPICommunicator（已初始化）。
                    若为 None，调用 MPICommunicator.get_instance()。

    Returns
    -------
    sim  : gpu.Simulation
    para : gpu.Parameter
    cfg  : WRFLBMConfig（透传）
    """
    from pyfluids import basics, gpu, parallel

    cfg.output_path.mkdir(parents=True, exist_ok=True)

    if communicator is None:
        communicator = parallel.MPICommunicator.get_instance()

    # --- Parameter（从 .cfg 文件加载）---
    config = basics.ConfigurationFile()
    config.load(str(cfg.config_path))

    para = gpu.Parameter(
        communicator.get_number_of_processes(),
        communicator.get_process_id(),
        config,
    )

    para.set_output_path(str(cfg.output_path))
    para.set_output_prefix("wrf_lbm_coupled")
    para.set_print_files(True)

    para.set_velocity_LB(cfg.velocity_LB)
    para.set_viscosity_LB(cfg.viscosity_LB)
    para.set_velocity_ratio(cfg.velocity_ratio)
    para.set_viscosity_ratio(cfg.dx * cfg.dx / cfg.dt)
    para.set_density_ratio(1.0)

    para.configure_main_kernel(gpu.kernel.compressible.K17CompressibleNavierStokes)

    para.set_timestep_start_out(0)
    para.set_timestep_out(cfg.n_steps_out)
    para.set_timestep_end(cfg.n_steps)
    para.set_is_body_force(False)

    # --- Grid ---
    bc_factory = gpu.BoundaryConditionFactory()
    grid_scaling_factory = gpu.GridScalingFactory()
    grid_scaling_factory.set_scaling_factory(gpu.GridScaling.ScaleCompressible)

    grid_dims = gpu.grid_generator.GridDimensions(
        0, cfg.domain_length_x,
        0, cfg.domain_length_y,
        0, cfg.domain_length_z,
        cfg.dx,
    )
    grid_builder = gpu.grid_generator.MultipleGridBuilderFacade(grid_dims)

    grid_builder.set_periodic_boundary_condition(False, False, False)
    grid_builder.create_grids(communicator.get_process_id())

    # Five lateral + roof faces → WRF-driven freestream (inflow/outflow decided per-node each step)
    grid_builder.set_velocity_boundary_condition(gpu.SideType.MX, 0.0, 0.0, 0.0)
    grid_builder.set_velocity_boundary_condition(gpu.SideType.PX, 0.0, 0.0, 0.0)
    grid_builder.set_velocity_boundary_condition(gpu.SideType.MY, 0.0, 0.0, 0.0)
    grid_builder.set_velocity_boundary_condition(gpu.SideType.PY, 0.0, 0.0, 0.0)
    grid_builder.set_velocity_boundary_condition(gpu.SideType.PZ, 0.0, 0.0, 0.0)

    # Ground: stress BC (unchanged)
    sampling_offset = 2
    grid_builder.set_stress_boundary_condition(
        gpu.SideType.MZ, 0, 0, 1, sampling_offset, cfg.z0, cfg.kappa, cfg.dx
    )

    # The five open faces are registered as velocity BC nodes above; the factory decides
    # which kernel owns them (BoundaryConditionKernelManager zeroes the other one).
    if bc_mode == "velocity":
        bc_factory.set_velocity_boundary_condition(gpu.VelocityBC.VelocityInterpolatedCompressible)
    elif bc_mode == "freestream":
        bc_factory.set_freestream_boundary_condition(gpu.FreestreamBC.FreestreamInterpolatedCompressible)
    else:
        raise ValueError(f"Unknown bc_mode {bc_mode!r} (expected 'velocity' or 'freestream')")
    bc_factory.set_stress_boundary_condition(gpu.StressBC.StressBounceBackCompressible)
    para.set_outflow_pressure_correction_factor(0.0)

    para.set_initial_condition(
        initial_condition_fn if initial_condition_fn is not None
        else gpu.initialization_functions.log_law(cfg.u_star, cfg.kappa, cfg.z0, cfg.velocity_ratio)
    )

    cuda_memory_manager = gpu.CudaMemoryManager(para)
    tm_factory = gpu.TurbulenceModelFactory(para)

    sim = gpu.Simulation(
        para,
        cuda_memory_manager,
        grid_builder.get_grid_builder(),
        bc_factory,
        tm_factory,
        grid_scaling_factory,
    )

    return sim, para, cfg
