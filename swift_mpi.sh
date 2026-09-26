#!/bin/bash
#SBATCH --job-name=swift_mpi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --time=00:30:00
#SBATCH --output=swift_mpi_%j.log
#SBATCH --account=naiss2026-1-15-gpu

echo "============================================================"
echo "SWiFT 3-Domain Nested Run with Real MPI Coupling"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

module load GPU/buildenv-gcccuda/2026.03-cu13.0
source /home/zhaqi746/code/VirtualFluids_dev/venv_vf/bin/activate

cd /home/zhaqi746/code/wrf_gpu
export GPUWRF_WRF_ROOT=/home/zhaqi746/code/WRF
export GPUWRF_COUPLING_VERBOSE=1

# Clean previous runs
echo "Cleaning previous output directories..."
rm -rf runs/swift_coupled_mpi
rm -rf received_nc_swift

echo ""
echo "============================================================"
echo "Starting 2-rank MPI job with unified MPI_COMM_WORLD:"
echo "  Rank 0: WRF GPU forecast (CUDA_VISIBLE_DEVICES=0)"
echo "  Rank 1: LBM MPI receiver (CUDA_VISIBLE_DEVICES=1)"
echo "============================================================"
echo ""

srun --ntasks=2 python run_coupled_mpi.py

exit_code=$?

echo ""
echo "============================================================"
echo "Job Complete"
echo "============================================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "✓ Job completed successfully"
    echo ""
    echo "Output locations:"
    echo "  - WRF output: runs/swift_coupled_mpi/"
    echo "  - Received data: received_nc_swift/"
    echo ""
    echo "To verify coupling:"
    echo "  ls -lh received_nc_swift/*.nc  # Should have 360 NetCDF files"
else
    echo "✗ Job failed with exit code $exit_code"
    echo "Check logs above for error messages"
fi

echo "============================================================"

exit $exit_code
