#!/bin/bash
#SBATCH --job-name=plot_swift100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=plot_swift100_%j.log
#SBATCH --account=naiss2026-1-15-gpu

# Plot the LBM x–z probe (xzPlaneMidY_bin_ID_*_t_*.vtk.pvtu) of the SWiFT 100 m coupled run.
# Output: ./runs/swift100m_coupled_mpi/lbm/plots/
# Extra arguments are passed to plotSwift100.py, e.g.
#   sbatch run_plot_swift100.sh --vars vx vz phi --every 5 --gif

echo "============================================================"
echo "SWiFT 100 m LBM probe plotting"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

module load GPU/buildenv-gcccuda/2026.03-cu13.0
source /home/zhaqi746/code/VirtualFluids_dev/venv_vf/bin/activate

cd /home/zhaqi746/code/wrf_gpu

PROBE_DIR=./runs/swift100m_coupled_mpi/lbm

echo "Python: $(which python3)"
python3 -c "import matplotlib, numpy; print('matplotlib/numpy OK')"
python3 -c "import vtk; print('vtk OK')" 2>/dev/null \
    || python3 -c "import meshio; print('meshio OK (vtk missing)')" 2>/dev/null \
    || { echo "✗ neither vtk nor meshio installed: pip install vtk"; exit 1; }
echo "Probe files: $(ls ${PROBE_DIR}/xzPlaneMidY_bin_ID_*_t_*.vtk.pvtu 2>/dev/null | wc -l) in ${PROBE_DIR}"

echo ""
echo "Running plotSwift100.py $*"
python3 plotSwift100.py --dir "${PROBE_DIR}" \
    --t0 3600 --dt 0.130594 --probe-every 15 "$@"
exit_code=$?

echo ""
echo "Exit code: $exit_code"
echo "Figures: ${PROBE_DIR}/plots/"
echo "End time: $(date)"
echo "============================================================"

exit $exit_code
