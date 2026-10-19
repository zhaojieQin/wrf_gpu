#!/bin/bash
#SBATCH --job-name=swift100m_wrf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=swift100m_wrf_%j.log
#SBATCH --account=naiss2026-1-15-gpu

# Standalone WRF GPU test of the SWiFT 100 m case (no LBM):
#   d01 2.5 km / d02 500 m / d03 100 m, centred on the SWiFT site, ERA5-driven.
# Checks: namelist accepted + compile, stable 1 h integration, wall time per simulated hour.

module load GPU/buildenv-gcccuda/2026.03-cu13.0
source /home/zhaqi746/code/VirtualFluids_dev/venv_vf/bin/activate

cd /home/zhaqi746/code/wrf_gpu
export GPUWRF_WRF_ROOT=../WRF

CASE_DIR=/home/zhaqi746/code/wrf_gpu/examples/SWiFT_100m
OUT_DIR=/home/zhaqi746/code/wrf_gpu/runs/swift_100m_test
HOURS=1

echo "Cleaning previous output directory: ${OUT_DIR}"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

echo ""
echo "============================================================"
echo "Standalone WRF GPU test (no LBM)"
echo "  Case   : ${CASE_DIR}"
echo "  Output : ${OUT_DIR}"
echo "  Domains: 3 (2.5 km / 500 m / 100 m), ${HOURS} h"
echo "  Node   : $(hostname)   Start: $(date)"
echo "============================================================"
ls -lh "${CASE_DIR}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo ""

start_s=$(date +%s)
srun --ntasks=1 gpuwrf run \
    --input-dir "${CASE_DIR}" \
    --output-dir "${OUT_DIR}" \
    --max-dom 3 \
    --hours ${HOURS}
exit_code=$?
end_s=$(date +%s)
wall_s=$((end_s - start_s))

echo ""
echo "============================================================"
echo "Job Complete   End: $(date)"
echo "Exit code: ${exit_code}"
echo "Wall time: ${wall_s} s  ($(awk "BEGIN{printf \"%.1f\", ${wall_s}/60}") min) for ${HOURS} simulated hour(s)"
echo "  (includes the first-run compile; rerun once to get the cached-compile timing)"
echo "============================================================"
if [ ${exit_code} -eq 0 ]; then
    echo "✓ WRF run finished. History files:"
    ls -lh "${OUT_DIR}" | grep -E "wrfout" || ls -lh "${OUT_DIR}"
else
    echo "✗ WRF run failed — check the log above (namelist validation / NonFiniteStateError / CFL)"
fi
echo "============================================================"

exit ${exit_code}
