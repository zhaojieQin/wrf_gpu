#!/bin/bash
# WRF-LBM 真实耦合测试启动脚本
#
# 用途：
#   启动 WRF 发送端（rank 0, GPU 0）和 LBM 接收端（rank 1, GPU 1）
#   进行完整的真实数据耦合测试
#
# 依赖：
#   - MPICH 4.3.2+ with CUDA support
#   - mpi4py
#   - coupling_prototype_3min.py (WRF 端)
#   - mock_receiver.py (LBM 端)
#
# GPU 分配：
#   Rank 0 → GPU 0（WRF 发送端）
#   Rank 1 → GPU 1（LBM 接收端）

set -e  # 遇到错误立即退出

# ============================================================
# 配置参数
# ============================================================
HOURS=${1:-1.0}               # 预报小时数（默认 1h）
COUPLING_INTERVAL=3.0         # 耦合间隔（分钟）
DT_WRF=18.0                   # WRF 时间步（秒）
DT_LBM=0.1                    # LBM 时间步（秒）

# 自动计算接收步数
# 公式：N_STEPS = HOURS * 60 / COUPLING_INTERVAL
N_STEPS=$(python3 -c "print(int($HOURS * 60 / $COUPLING_INTERVAL))")

# ============================================================
# 环境检查
# ============================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -f "$SCRIPT_DIR/coupling_prototype_3min.py" ]; then
    echo "错误：找不到 coupling_prototype_3min.py"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/mock_receiver.py" ]; then
    echo "错误：找不到 mock_receiver.py"
    exit 1
fi

# 检查 MPI 环境
if ! command -v mpiexec.hydra &> /dev/null && ! command -v srun &> /dev/null; then
    echo "错误：未找到 MPI launcher（mpiexec.hydra 或 srun）"
    exit 1
fi

# ============================================================
# 打印配置
# ============================================================
echo "============================================================"
echo "WRF-LBM 耦合测试配置"
echo "============================================================"
echo "  预报小时数: $HOURS h"
echo "  耦合间隔: $COUPLING_INTERVAL min"
echo "  接收步数: $N_STEPS 步"
echo "  WRF 时间步: $DT_WRF s"
echo "  LBM 时间步: $DT_LBM s"
echo "  工作目录: $SCRIPT_DIR"
echo "============================================================"
echo ""

# ============================================================
# 启动 MPI 进程
# ============================================================
# 优先使用 srun（Slurm 标准）
if command -v srun &> /dev/null; then
    echo "使用 srun 启动（Slurm）..."
    srun --ntasks=2 --gpus-per-task=1 bash -c "
        if [ \$SLURM_PROCID -eq 0 ]; then
            export CUDA_VISIBLE_DEVICES=0
            cd '$SCRIPT_DIR'
            python coupling_prototype_3min.py --mpi --hours $HOURS \
                --dt-wrf $DT_WRF --dt-lbm $DT_LBM \
                --desired-interval $COUPLING_INTERVAL
        elif [ \$SLURM_PROCID -eq 1 ]; then
            export CUDA_VISIBLE_DEVICES=1
            cd '$SCRIPT_DIR'
            python mock_receiver.py --steps $N_STEPS --verify-wrf
        fi
    "
# 备选：mpiexec.hydra（MPICH 标准）
elif command -v mpiexec.hydra &> /dev/null; then
    echo "使用 mpiexec.hydra 启动（MPICH）..."
    mpiexec.hydra -n 1 bash -c "
        export CUDA_VISIBLE_DEVICES=0
        cd '$SCRIPT_DIR'
        python coupling_prototype_3min.py --mpi --hours $HOURS \
            --dt-wrf $DT_WRF --dt-lbm $DT_LBM \
            --desired-interval $COUPLING_INTERVAL
    " : -n 1 bash -c "
        export CUDA_VISIBLE_DEVICES=1
        cd '$SCRIPT_DIR'
        python mock_receiver.py --steps $N_STEPS --verify-wrf
    "
else
    echo "错误：未找到可用的 MPI launcher"
    exit 1
fi

# ============================================================
# 检查结果
# ============================================================
EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 耦合测试完成"
    echo "============================================================"
    echo "  预报小时数: $HOURS h"
    echo "  接收步数: $N_STEPS 步"
    echo "  退出码: $EXIT_CODE"
    echo "============================================================"
else
    echo "✗ 耦合测试失败"
    echo "============================================================"
    echo "  退出码: $EXIT_CODE"
    echo "  建议："
    echo "    1. 检查 GPU 可用性"
    echo "    2. 检查 MPI 环境配置"
    echo "    3. 减少 --hours 参数重试"
    echo "============================================================"
fi

exit $EXIT_CODE
