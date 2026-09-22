"""WRF-GPU 单侧求解器封装

提供可编程接口封装 gpuwrf.integration.daily_pipeline，用于:
1. 外部 Python 进程调用 WRF-GPU 求解器
2. 未来与其他求解器耦合的接入点

设计原则:
- 不修改 gpuwrf 包内源码
- 优先复用已有 API (execute_daily_pipeline)
- 清晰的错误处理和结果返回
- 为未来耦合预留接口

作者: 自动生成
日期: 2026-09-22
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

__all__ = ["WRFGPURunner", "WRFGPUResult", "WRFGPUError"]


# ============================================================================
# 异常定义
# ============================================================================

class WRFGPUError(Exception):
    """WRF-GPU 运行错误基类"""

    def __init__(self, message: str, exit_code: int = 1, payload: dict[str, Any] | None = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.payload = payload or {}


class NamelistValidationError(WRFGPUError):
    """Namelist 验证失败"""
    pass


class GPUPreflightError(WRFGPUError):
    """GPU 预检失败"""

    def __init__(self, message: str, payload: dict[str, Any] | None = None):
        super().__init__(message, exit_code=75, payload=payload)


class PipelineBlockedError(WRFGPUError):
    """Pipeline 运行被阻塞"""
    pass


# ============================================================================
# 结果数据类
# ============================================================================

@dataclass
class WRFGPUResult:
    """WRF-GPU 运行结果

    Attributes:
        status: 运行状态 ("SUCCESS" | "FAILED" | "BLOCKED" | "DRY_RUN")
        exit_code: 退出码 (0=成功, 1=失败, 75=GPU预检失败)
        verdict: Pipeline verdict ("PIPELINE_GREEN" | "PIPELINE_PARTIAL" | "PIPELINE_BLOCKED" | None)
        wrfout_files: 生成的 wrfout NetCDF 文件路径列表
        proof_dir: Proof JSON 文件目录
        metadata: 完整 pipeline payload (JSON-serializable dict)
        wall_clock_total_s: 总墙钟时间 (秒)
        wall_clock_per_hour_s: 每小时墙钟时间列表
        init_mode: 初始化模式 ("standalone_native_init" | "cpu_wrf_replay" | None)
        error_message: 失败时的错误描述
    """
    status: str
    exit_code: int
    verdict: str | None
    wrfout_files: list[Path]
    proof_dir: Path | None
    metadata: dict[str, Any]

    wall_clock_total_s: float | None = None
    wall_clock_per_hour_s: list[float] | None = None
    init_mode: str | None = None
    error_message: str | None = None

    def is_success(self) -> bool:
        """运行是否成功 (verdict == PIPELINE_GREEN)"""
        return self.verdict == "PIPELINE_GREEN"


# ============================================================================
# 主封装类
# ============================================================================

class WRFGPURunner:
    """WRF-GPU 单侧求解器封装

    封装 gpuwrf.integration.daily_pipeline.execute_daily_pipeline()，
    提供可编程接口用于外部调用和未来耦合集成。

    当前限制:
        - execute_daily_pipeline() 是整段运行，无逐时间步回调
        - 无法在不修改 gpuwrf 内部的情况下实现实时数据交换

    未来耦合接入点 (未验证):
        方案 1: 分段运行 - 多次调用 run_forecast(hours=1)，通过 wrfout
                传递状态。需验证 gpuwrf 的 checkpoint/restart 支持。
        方案 2: 自定义 forecast_fn - 利用 execute_daily_pipeline 的
                forecast_fn 参数注入回调 (需理解 JAX pytree 结构)。
        方案 3: 上游改造 - 修改 gpuwrf 运行时添加原生步进接口。

    使用示例:
        >>> runner = WRFGPURunner()
        >>> result = runner.run_forecast(
        ...     input_dir=Path("examples/switzerland_d01"),
        ...     output_dir=Path("runs/test"),
        ...     hours=24,
        ...     domain="d01",
        ... )
        >>> if result.is_success():
        ...     print(f"成功: {len(result.wrfout_files)} 个输出文件")
    """

    def __init__(self):
        """初始化 runner (当前无需配置)"""
        pass

    def run_forecast(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        hours: int,
        domain: str = "d01",
        scratch_dir: Path | str | None = None,
        dry_run: bool = False,
        proof_dir: Path | str | None = None,
    ) -> WRFGPUResult:
        """运行一次 WRF-GPU 预报

        Args:
            input_dir: 输入目录，必须包含:
                - namelist.input
                - wrfinput_d01 (初始条件)
                - wrfbdy_d01 (边界条件)
            output_dir: 输出目录 (wrfout_* 文件写入位置)
            hours: 预报时长 (小时)，必须 > 0
            domain: WRF domain ID (默认 "d01")
            scratch_dir: 暂存目录 (用于 JAX 编译缓存等)
                - None: 默认使用 <output_dir>/.scratch，运行后自动清理
                - 显式路径: 使用指定路径，不自动清理
            dry_run: 是否 dry-run (仅检测配置，不运行预报)
            proof_dir: Proof JSON 输出目录 (默认 <output_dir>/proofs)

        Returns:
            WRFGPUResult: 运行结果，包含 wrfout 文件列表、verdict、性能信息

        Raises:
            WRFGPUError: 运行失败 (子类: NamelistValidationError, PipelineBlockedError 等)
            FileNotFoundError: 输入目录或 namelist 不存在
            ValueError: 参数无效 (如 hours <= 0)

        副作用:
            - 设置环境变量 GPUWRF_SCRATCH, GPUWRF_TMPDIR
            - 创建 output_dir, proof_dir, scratch_dir
            - 自动清理 scratch_dir (如果使用默认路径)

        内部流程:
            1. 参数校验和路径规范化
            2. Namelist fail-closed 验证
            3. 检测初始化模式 (standalone_native_init vs cpu_wrf_replay)
            4. 如果 dry_run: 返回配置信息，不运行
            5. 调用 execute_daily_pipeline() 运行预报
            6. 解析返回 payload，构造 WRFGPUResult
            7. 清理 scratch_dir (如果需要)
        """
        # ---- 1. 参数校验和规范化 ----
        input_dir = Path(input_dir).resolve()
        output_dir = Path(output_dir).resolve()

        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        namelist_path = input_dir / "namelist.input"
        if not namelist_path.is_file():
            raise FileNotFoundError(
                f"namelist.input not found in input directory: {namelist_path}"
            )

        if hours <= 0:
            raise ValueError(f"hours must be positive, got {hours}")

        # scratch_dir 逻辑: None → 默认路径 + 自动清理
        user_provided_scratch = scratch_dir is not None
        if scratch_dir is None:
            scratch_dir = output_dir / ".scratch"
        scratch_dir = Path(scratch_dir).resolve()

        if proof_dir is None:
            proof_dir = output_dir / "proofs"
        proof_dir = Path(proof_dir).resolve()

        # ---- 2. Namelist 验证 (pre-JAX, 快速失败) ----
        try:
            from gpuwrf.io.namelist_check import (
                UnsupportedSchemeError,
                validate_operational_namelist,
            )
            validate_operational_namelist(namelist_path)
        except UnsupportedSchemeError as exc:
            raise NamelistValidationError(str(exc)) from exc
        except Exception as exc:
            raise NamelistValidationError(
                f"Failed to validate namelist: {type(exc).__name__}: {exc}"
            ) from exc

        # ---- 3. 构造 DailyPipelineConfig + 检测初始化模式 ----
        try:
            from gpuwrf.integration.daily_pipeline import (
                DailyPipelineConfig,
                detect_init_mode,
                execute_daily_pipeline,
            )
        except ImportError as exc:
            raise WRFGPUError(
                f"Failed to import gpuwrf daily_pipeline: {exc}. "
                "Is the package installed and JAX available?"
            ) from exc

        config = DailyPipelineConfig(
            run_id=str(input_dir),
            run_root=input_dir.parent,
            hours=int(hours),
            output_dir=output_dir,
            proof_dir=proof_dir,
            domain=domain,
            score=False,
            restart_at_hour=None,
            repeat=False,
        )

        init_mode = detect_init_mode(config)

        # ---- 4. Dry-run: 返回配置计划 ----
        if dry_run:
            plan = {
                "schema": "WRFGPURunnerDryRun",
                "dry_run": True,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "proof_dir": str(proof_dir),
                "scratch_dir": str(scratch_dir),
                "domain": domain,
                "hours": hours,
                "init_mode": init_mode,
                "namelist_path": str(namelist_path),
            }
            return WRFGPUResult(
                status="DRY_RUN",
                exit_code=0,
                verdict=None,
                wrfout_files=[],
                proof_dir=None,
                metadata=plan,
                init_mode=init_mode,
            )

        # ---- 5. 设置 scratch 环境变量 + 创建目录 ----
        scratch_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        proof_dir.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("GPUWRF_SCRATCH", str(scratch_dir))
        os.environ["GPUWRF_TMPDIR"] = str(scratch_dir)

        # 清理条件: 用户没传 scratch_dir + 没设 GPUWRF_KEEP_SCRATCH 环境变量
        cleanup_scratch = (
            not user_provided_scratch
            and not os.environ.get("GPUWRF_KEEP_SCRATCH")
        )

        # ---- 6. 运行 pipeline ----
        try:
            payload = execute_daily_pipeline(config)
        except Exception as exc:
            # 失败时清理 scratch (如果需要)
            if cleanup_scratch:
                shutil.rmtree(scratch_dir, ignore_errors=True)

            # 捕获 PipelineBlocked 异常 (gpuwrf 内部定义)
            if type(exc).__name__ == "PipelineBlocked":
                raise PipelineBlockedError(
                    str(exc),
                    payload=getattr(exc, "payload", {}),
                ) from exc

            # 通用异常
            raise WRFGPUError(
                f"Pipeline execution failed: {type(exc).__name__}: {exc}",
                exit_code=1,
            ) from exc
        finally:
            # 无论成功失败都清理 scratch (如果需要)
            if cleanup_scratch:
                shutil.rmtree(scratch_dir, ignore_errors=True)

        # ---- 7. 解析 payload 构造结果 ----
        verdict = payload.get("verdict", "UNKNOWN")
        exit_code = 0 if verdict == "PIPELINE_GREEN" else 1

        wrfout_files = [
            Path(f) for f in payload.get("wrfout_files", [])
        ]

        result = WRFGPUResult(
            status="SUCCESS" if exit_code == 0 else "FAILED",
            exit_code=exit_code,
            verdict=verdict,
            wrfout_files=wrfout_files,
            proof_dir=proof_dir,
            metadata=payload,
            wall_clock_total_s=payload.get("wall_clock_total_s"),
            wall_clock_per_hour_s=payload.get("wall_clock_per_hour_s"),
            init_mode=init_mode,
            error_message=None if exit_code == 0 else payload.get("reason"),
        )

        return result

    def run_coupled_step(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        coupling_interval_hours: float,
        exchange_callback: Callable[[dict[str, Any]], dict[str, Any]],
        total_hours: int,
        domain: str = "d01",
        scratch_dir: Path | str | None = None,
    ) -> WRFGPUResult:
        """【未实现】按耦合间隔步进并交换数据

        未来耦合接口预留。当前 gpuwrf 的 execute_daily_pipeline() 是整段运行，
        无法在中间插入回调。可能的实现路径：

        方案 1 (最小侵入，未验证):
            分段调用 execute_daily_pipeline(hours=coupling_interval_hours)，
            每段结束后通过 exchange_callback 交换数据，将另一侧的状态写入
            下一段的 wrfinput_d01。需验证 gpuwrf 是否支持从 wrfout 重启。

        方案 2 (中等侵入):
            利用 execute_daily_pipeline(forecast_fn=...) 参数，包装
            run_forecast_operational 并在内部插入步进回调。需要理解
            JAX pytree state 结构。

        方案 3 (完全侵入):
            修改 gpuwrf/runtime/operational_mode.py 添加原生逐步回调支持。
            需要 upstream 接受此功能。

        Args:
            input_dir: 初始输入目录 (wrfinput/wrfbdy)
            output_dir: 输出目录
            coupling_interval_hours: 耦合间隔 (小时)，如 1.0
            exchange_callback: 回调函数，签名:
                input: {"step": int, "valid_time": datetime, "wrfout_path": Path, ...}
                output: {"sst": np.ndarray, "fluxes": dict, ...}  # 耦合数据
            total_hours: 总预报时长
            domain: WRF domain ID
            scratch_dir: 暂存目录

        Returns:
            WRFGPUResult: 运行结果

        Raises:
            NotImplementedError: 当前版本未实现

        TODO: 实现耦合步进逻辑 (需要先验证分段重启可行性)
        TODO: 定义 exchange_callback 的标准输入/输出 schema
        TODO: 验证 gpuwrf 的 checkpoint/restart 是否支持逐小时重启
        """
        raise NotImplementedError(
            "Coupled stepping not yet implemented. Current gpuwrf architecture "
            "runs full forecast segments without mid-run callbacks. See docstring "
            "for possible implementation paths."
        )


# ============================================================================
# 命令行入口 (测试用)
# ============================================================================

def main() -> int:
    """最小可运行示例 (手动测试用)

    用法:
        python wrf_gpu_runner.py \\
            --input-dir examples/switzerland_d01 \\
            --output-dir runs/runner_test \\
            --hours 1 \\
            --dry-run
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="WRF-GPU Runner 测试入口")
    parser.add_argument("--input-dir", type=Path, required=True, help="输入目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--hours", type=int, default=1, help="预报时长 (小时)")
    parser.add_argument("--domain", type=str, default="d01", help="Domain ID")
    parser.add_argument("--scratch-dir", type=Path, default=None, help="暂存目录")
    parser.add_argument("--dry-run", action="store_true", help="仅检测配置")

    args = parser.parse_args()

    runner = WRFGPURunner()

    try:
        result = runner.run_forecast(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            hours=args.hours,
            domain=args.domain,
            scratch_dir=args.scratch_dir,
            dry_run=args.dry_run,
        )
    except WRFGPUError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        if exc.payload:
            print(f"[PAYLOAD] {json.dumps(exc.payload, indent=2)}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        print(f"[UNEXPECTED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # 打印结果
    print(f"[STATUS] {result.status}")
    print(f"[VERDICT] {result.verdict}")
    print(f"[EXIT_CODE] {result.exit_code}")
    print(f"[INIT_MODE] {result.init_mode}")

    if result.wrfout_files:
        print(f"[WRFOUT_FILES] {len(result.wrfout_files)} 个文件:")
        for f in result.wrfout_files:
            print(f"  - {f}")

    if result.wall_clock_total_s:
        print(f"[WALL_CLOCK] {result.wall_clock_total_s:.2f} s")

    if result.error_message:
        print(f"[ERROR_MSG] {result.error_message}")

    print(f"\n[METADATA]")
    print(json.dumps(result.metadata, indent=2, default=str))

    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
