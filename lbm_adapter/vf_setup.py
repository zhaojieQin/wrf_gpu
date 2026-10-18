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


@dataclass
class ThermalConfig:
    """温度场（AD）+ 浮力 + 地面 surface layer 的参数，取值参照 VF apps/gpu/StratifiedBoundaryLayer。

    温度在 LBM 中存为 concentration = θ − theta_ref（K）。
    """
    theta_ref: float = 300.0             # K，参考位温（= LBMReference.theta_ref）
    prandtl_number: float = 0.7          # 分子 Prandtl 数（扩散系数 = ν / Pr）
    roughness_length_temperature: float | None = None  # z0h（m）；None → 与动量 z0 相同（参考算例）
    sampling_offset: int = 3             # wall model 采样高度（格点数，参考算例 SamplingOffset = 3）
    turbulence_model: str = "AMDStratified"
    turbulence_model_ad: str = "AMDStratified"
    sgs_constant: float = 0.2
    gravity: float = 9.81


# ---------------------------------------------------------------------------
# VirtualFluids 仿真构造
# ---------------------------------------------------------------------------

def setup_vf_simulation(cfg: WRFLBMConfig, communicator=None,
                        initial_condition_fn=None, bc_mode: str = "freestream",
                        thermal: ThermalConfig | None = None,
                        initial_condition_ad_fn=None,
                        initial_reference_temperature_fn=None,
                        print_files: bool = True,
                        probe_every_steps: int | None = None):
    """构造并返回 VirtualFluids Simulation 对象。

    Parameters
    ----------
    cfg           : WRFLBMConfig（由 WRFLBMConfig.from_cfg_file() 创建）
    communicator  : pyfluids MPICommunicator（已初始化）。
                    若为 None，调用 MPICommunicator.get_instance()。
    thermal       : ThermalConfig | None。给定时启用温度场：AD kernel、浮力（planar average）、
                    地面 surface layer（热通量驱动）、五个开放面温度 Dirichlet、分层 SGS 模型。
                    None 时保持中性设置（地面 stress BC），行为与之前一致。
    initial_condition_ad_fn : ic_ad(x, y, z) -> θ − θ_ref（K），thermal 时使用
    initial_reference_temperature_fn : 同签名，初始局部参考温度（浮力/分层 SGS 用）
    print_files   : 是否写全场 VTK（False 时只保留探针输出）
    probe_every_steps : x–z 中间面（y≈L_y/2）瞬时探针的输出间隔（LBM 步）；None 不加探针

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
    para.set_print_files(print_files)

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

    if thermal is None:
        # Ground: stress BC. Binding order: (..., sampling_offset, von_karman_constant,
        # roughness_length, delta_x) — kappa before z0 (they were swapped before).
        sampling_offset = 2
        grid_builder.set_stress_boundary_condition(
            gpu.SideType.MZ, 0, 0, 1, sampling_offset, cfg.kappa, cfg.z0, cfg.dx
        )
    else:
        # Ground: surface layer BC (momentum wall model + surface heat flux), as in
        # StratifiedBoundaryLayer. The flux is replaced every step from WRF HFX via
        # sim.update_surface_layer_heat_flux; start from zero.
        z0h = thermal.roughness_length_temperature or cfg.z0
        grid_builder.set_surface_layer_boundary_condition(
            gpu.SideType.MZ, 0.0, 0.0, 1.0, thermal.sampling_offset, cfg.kappa, cfg.z0, z0h,
            0.0,   # surface_heat_flux (kinematic, K m/s) — overwritten each step
            0.0,   # surface_temperature (deviation, unused in heat-flux mode)
            0.0,   # heating_rate
            cfg.dx, cfg.dt,
        )
        # Temperature on the five open faces: Dirichlet, overwritten each step from WRF θ
        for side in (gpu.SideType.MX, gpu.SideType.PX, gpu.SideType.MY,
                     gpu.SideType.PY, gpu.SideType.PZ):
            grid_builder.set_AD_dirichlet_boundary_condition(side, 0.0, 0.0, 0.0, 0.0)

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

    if thermal is not None:
        bc_factory.set_surface_layer_boundary_condition(
            gpu.StressBC.StressBounceBackCompressible, gpu.SurfaceLayerBC.SurfaceHeatFlux)
        # Slip variant uses the local fluid velocity at the node (open boundary, not a wall)
        bc_factory.set_advection_diffusion_dirichlet_boundary_condition(
            gpu.AdvectionDiffusionDirichletBC.DirichletAntiBounceBackSlip)

        diffusivity_lb = (cfg.viscosity / thermal.prandtl_number) * cfg.dt / (cfg.dx * cfg.dx)
        gravity_lb = thermal.gravity * cfg.dt * cfg.dt / cfg.dx
        para.set_diff_on(True)
        para.set_AD_kernel(gpu.kernel.advection_diffusion.compressible.F16)
        para.set_diffusivity_LB(diffusivity_lb)
        para.set_reference_temperature(thermal.theta_ref)
        para.set_buoyancy_factor(gravity_lb / thermal.theta_ref)
        para.set_is_body_force(True)
        para.set_buoyancy_enabled(True)
        if initial_condition_ad_fn is not None:
            para.set_initial_condition_AD(initial_condition_ad_fn)
        if initial_reference_temperature_fn is not None:
            para.set_initial_local_reference_temperature(initial_reference_temperature_fn)

    para.set_initial_condition(
        initial_condition_fn if initial_condition_fn is not None
        else gpu.initialization_functions.log_law(cfg.u_star, cfg.kappa, cfg.z0, cfg.velocity_ratio)
    )

    cuda_memory_manager = gpu.CudaMemoryManager(para)
    tm_factory = gpu.TurbulenceModelFactory(para)

    if thermal is not None:
        # Buoyancy from the deviation to the horizontal planar average (StratifiedBoundaryLayer)
        para.add_interactor(gpu.BuoyancyProviderPlanarAverage(para, cuda_memory_manager))
        # SGS constant must be set first: setTurbulenceModel throws if it is still 0
        # (same order as TurbulenceModelFactory::readConfigFile)
        tm_factory.set_model_constant(thermal.sgs_constant)
        tm_factory.set_turbulence_model(thermal.turbulence_model)
        tm_factory.set_turbulence_model_advection_diffusion(thermal.turbulence_model_ad)

    if probe_every_steps is not None:
        # Streamwise x–z plane at mid-y. The probe samples every node inside the cuboid
        # [x0, x0+Lx] x [y0, y0+Ly] x [z0, z0+Lz]; take a one-node-thick slab in y around
        # the grid node closest to L_y/2 (VF nodes sit at -dx/2 + j*dx).
        dx = cfg.dx
        j_mid = int(round((0.5 * cfg.domain_length_y + 0.5 * dx) / dx))
        y_node = -0.5 * dx + j_mid * dx
        plane_probe = gpu.probes.Probe(
            para, cuda_memory_manager, str(cfg.output_path), "xzPlaneMidY",
            0, 1,                       # t_start_avg, t_avg (only instantaneous values used)
            0, int(probe_every_steps),  # t_start_out, t_out
            False, False,               # output_timeseries, average_every_timestep
            thermal is not None,        # sample_scalar → also writes temperature
        )
        plane_probe.set_probe_plane(-dx, y_node - 0.5 * dx, -dx,
                                    cfg.domain_length_x + 2 * dx, dx, cfg.domain_length_z + 2 * dx)
        plane_probe.add_statistic(gpu.probes.Probe.Statistic.Instantaneous)
        para.add_sampler(plane_probe)
        print(f"[vf_setup] x–z plane probe at y={y_node:.2f} m, every {probe_every_steps} steps "
              f"(≈{probe_every_steps * cfg.dt:.2f} s) -> {cfg.output_path}", flush=True)

    sim = gpu.Simulation(
        para,
        cuda_memory_manager,
        grid_builder.get_grid_builder(),
        bc_factory,
        tm_factory,
        grid_scaling_factory,
    )

    return sim, para, cfg
