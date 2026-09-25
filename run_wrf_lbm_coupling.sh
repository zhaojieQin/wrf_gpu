#!/bin/bash
#SBATCH --job-name=wrf_lbm_coupled
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --time=01:00:00
#SBATCH --output=coupled_%j.log
#SBATCH --account=naiss2026-1-15-gpu

# ============================================================
# 环境
# ============================================================
module load GPU/buildenv-gcccuda/2026.03-cu13.0
cd $SLURM_SUBMIT_DIR
source /home/zhaqi746/code/VirtualFluids_dev/venv_vf/bin/activate

# ============================================================
# 配置
# ============================================================
SCRIPT_DIR="../wrf_gpu"
export GPUWRF_WRF_ROOT=../WRF
HOURS=${1:-1.0}                        # 预报小时数（默认 1h）
COUPLING_INTERVAL_MIN=3.0              # 耦合间隔（分钟）
COUPLING_INTERVAL_S=$(python3 -c "print($COUPLING_INTERVAL_MIN * 60)")  # 秒
DT_WRF=18.0                            # WRF 时间步（秒）
DT_LBM=0.1                             # LBM 时间步（秒）

# export GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32

# 接收步数 = 小时数 * 60 / 耦合间隔（分钟）
N_STEPS=$(python3 -c "import math; print(math.ceil($HOURS * 60 / $COUPLING_INTERVAL_MIN))")

# ============================================================
# 检查
# ============================================================
if [ ! -f "$SCRIPT_DIR/coupling_prototype_3min.py" ]; then
    echo "错误：找不到 $SCRIPT_DIR/coupling_prototype_3min.py"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/mock_receiver.py" ]; then
    echo "错误：找不到 $SCRIPT_DIR/mock_receiver.py"
    exit 1
fi

# ============================================================
# 打印配置
# ============================================================
echo "============================================================"
echo "WRF-LBM 耦合测试配置"
echo "============================================================"
echo "  预报小时数: $HOURS h"
echo "  耦合间隔: $COUPLING_INTERVAL_MIN min ($COUPLING_INTERVAL_S s)"
echo "  接收步数: $N_STEPS 步"
echo "  WRF 时间步: $DT_WRF s"
echo "  LBM 时间步: $DT_LBM s"
echo "  工作目录: $SCRIPT_DIR"
echo "============================================================"
echo ""

# ============================================================
# 启动
# ============================================================
srun --ntasks=2 bash -c "
    if [ \$SLURM_PROCID -eq 0 ]; then
        export CUDA_VISIBLE_DEVICES=0
        cd $SCRIPT_DIR
        python -u coupling_prototype_3min.py --mpi --hours $HOURS \
            --dt-wrf $DT_WRF --dt-lbm $DT_LBM \
            --desired-interval $COUPLING_INTERVAL_S
    elif [ \$SLURM_PROCID -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=1
        cd $SCRIPT_DIR
        python -u mock_receiver.py --steps $N_STEPS --verify-wrf
    fi
"

EXIT_CODE=$?

# ============================================================
# 结果
# ============================================================
echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 耦合测试完成"
else
    echo "✗ 耦合测试失败（退出码 $EXIT_CODE）"
    echo "  建议："
    echo "    1. 检查 GPU 可用性"
    echo "    2. 检查 MPI 环境配置"
    echo "    3. 减少 --hours 参数重试"
fi
echo "============================================================"

exit $EXIT_CODE