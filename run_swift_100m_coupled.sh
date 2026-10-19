#!/bin/bash
#SBATCH --job-name=swift100m_mpi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --time=08:00:00
#SBATCH --output=swift100m_mpi_%j.log
#SBATCH --account=naiss2026-1-15-gpu

# WRF (SWiFT 100 m case, 2.5 km / 500 m / 100 m) → LBM coupled run.
#   Rank 0: WRF GPU forecast   Rank 1: LBM (VirtualFluids)
# Case, run dir, spin-up, coupled hours, interval: coupling_config.py
#   (WRF_CASE_DIR, RUN_DIR, SPINUP_HOURS, HOURS, COUPLING_EVERY_N_STEPS = 5 → 5 × 0.4 s = 2 s).
# --time: first run also compiles the new grids; spin-up + coupled time on a 0.4 s d03 step is unknown,
#         adjust after the first run.

module load GPU/buildenv-gcccuda/2026.03-cu13.0
source /home/zhaqi746/code/VirtualFluids_dev/venv_vf/bin/activate

cd /home/zhaqi746/code/wrf_gpu
export GPUWRF_WRF_ROOT=../WRF
# 1 = per-frame coupling log + per-step [DBG-leaf] lines (large log); 0 for production
export GPUWRF_COUPLING_VERBOSE=0

RUN_DIR=$(python -c "import coupling_config as c; print(c.RUN_DIR)")
CASE_DIR=$(python -c "import coupling_config as c; print(c.WRF_CASE_DIR)")

echo "Cleaning previous output directory: ${RUN_DIR}"
rm -rf "${RUN_DIR}"
echo ""
echo "============================================================"
echo "Starting 2-rank MPI job with unified MPI_COMM_WORLD:"
echo "  Rank 0: WRF GPU forecast (CUDA_VISIBLE_DEVICES=0)"
echo "  Rank 1: LBM MPI receiver (CUDA_VISIBLE_DEVICES=1)"
echo "  Case  : ${CASE_DIR}"
echo "  Run   : ${RUN_DIR}"
python -c "import coupling_config as c; print(f'  Spin-up {c.SPINUP_HOURS} h, coupled {c.HOURS} h, frame every {c.COUPLING_EVERY_N_STEPS} d03 steps, CPM={c.LBM_CPM_ENABLE}')"
echo "  Node  : $(hostname)   Start: $(date)"
echo "============================================================"
echo ""

start_s=$(date +%s)
srun --ntasks=2 python coupling_adapter.py
exit_code=$?
wall_s=$(( $(date +%s) - start_s ))

echo ""
echo "============================================================"
echo "Job Complete   End: $(date)"
echo "Exit code: ${exit_code}"
echo "Wall time: ${wall_s} s ($(awk "BEGIN{printf \"%.1f\", ${wall_s}/60}") min)"
echo "============================================================"
if [ ${exit_code} -eq 0 ]; then
    echo "✓ Job completed successfully"
    echo "  - WRF output : ${RUN_DIR}/"
    echo "  - LBM probes : ${RUN_DIR}/lbm/"
else
    echo "✗ Job failed with exit code ${exit_code}"
    echo "Check logs above for error messages"
fi
echo "============================================================"

exit ${exit_code}
