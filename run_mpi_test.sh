#!/bin/bash
# WRF-LBM MPI 通信测试启动脚本
#
# 用途：
#   启动 WRF 发送端（rank 0, GPU 0）和 LBM 接收端（rank 1, GPU 1）
#   验证 MPI 通信通道的正确性
#
# 使用前：
#   1. 确认 launcher 可用性（见下方测试命令）
#   2. 根据测试结果选择合适的 LAUNCHER
#   3. 取消注释对应的启动命令
#
# GPU 分配：
#   Rank 0 → GPU 0（WRF 发送端）
#   Rank 1 → GPU 1（LBM 接收端）

# ============================================================
# LAUNCHER 选择（待确认）
# ============================================================
# Arrhenius 推荐 mpprun，但需要验证 Python 脚本支持
# 测试命令：
#   mpprun -n 2 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()}')"
#
# 如果 mpprun 不支持，备选：
#   - srun（Slurm 标准）
#   - mpiexec.hydra（MPICH 标准）

LAUNCHER=???  # 待用户测试后填写：mpprun / srun / mpiexec.hydra

# ============================================================
# 候选方案 1: mpprun（Arrhenius 推荐）
# ============================================================
# 适用：如果 mpprun 支持 Python 脚本
# 取消注释使用：
#
# $LAUNCHER -n 2 bash -c '
#     if [ $SLURM_PROCID -eq 0 ]; then
#         export CUDA_VISIBLE_DEVICES=0
#         cd /home/qzj/code/WRFGPUTest/wrf_gpu
#         python test_wrf_sender.py
#     elif [ $SLURM_PROCID -eq 1 ]; then
#         export CUDA_VISIBLE_DEVICES=1
#         cd /home/qzj/code/WRFGPUTest/wrf_gpu
#         python mock_receiver.py --steps 3
#     fi
# '

# ============================================================
# 候选方案 2: srun（Slurm 标准）
# ============================================================
# 适用：标准 Slurm 集群
# 取消注释使用：
#
# srun --ntasks=2 bash -c '
#     if [ $SLURM_PROCID -eq 0 ]; then
#         export CUDA_VISIBLE_DEVICES=0
#         cd /home/qzj/code/WRFGPUTest/wrf_gpu
#         python test_wrf_sender.py
#     elif [ $SLURM_PROCID -eq 1 ]; then
#         export CUDA_VISIBLE_DEVICES=1
#         cd /home/qzj/code/WRFGPUTest/wrf_gpu
#         python mock_receiver.py --steps 3
#     fi
# '

# ============================================================
# 候选方案 3: mpiexec.hydra（MPICH 标准）
# ============================================================
# 适用：非 Slurm 环境或 MPICH 原生启动
# 取消注释使用：
#
# mpiexec.hydra -n 1 bash -c "
#     export CUDA_VISIBLE_DEVICES=0
#     cd /home/qzj/code/WRFGPUTest/wrf_gpu
#     python test_wrf_sender.py
# " : -n 1 bash -c "
#     export CUDA_VISIBLE_DEVICES=1
#     cd /home/qzj/code/WRFGPUTest/wrf_gpu
#     python mock_receiver.py --steps 3
# "

# ============================================================
# 使用说明
# ============================================================
# 1. 验证通道（3 步）：
#    python mock_receiver.py --steps 3
#
# 2. 验证稳定性（20 步）：
#    python mock_receiver.py --steps 20
#
# 3. GPU 数据验证（可选）：
#    python mock_receiver.py --steps 3 --verify-gpu

echo "错误：LAUNCHER 未配置"
echo ""
echo "请先测试 launcher 可用性："
echo "  mpprun -n 2 python -c \"from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()}')\""
echo ""
echo "根据测试结果，编辑本脚本设置 LAUNCHER 并取消注释对应的启动命令"
exit 1
