#!/bin/bash
#SBATCH --job-name=plot_swift
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=plot_swift_%j.log
#SBATCH --account=naiss2026-1-15-gpu

echo "============================================================"
echo "SWiFT WRF Output Visualization Job"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module load GPU/buildenv-gcccuda/2026.03-cu13.0

# Activate virtual environment
source /home/zhaqi746/code/VirtualFluids_dev/venv_vf/bin/activate

# Change to working directory
cd /home/zhaqi746/code/wrf_gpu

# Check Python and dependencies
echo "Python: $(which python3)"
python3 -c "import matplotlib; import netCDF4; print('Dependencies OK')"

# Run plotting script
echo ""
echo "Running plot_swift_wrfout.py..."
python3 plot_swift_wrfout.py

exit_code=$?

echo ""
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo "============================================================"

exit $exit_code
