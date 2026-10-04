#!/bin/bash
#SBATCH --job-name=torchspin-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/results/cluster/%x-%j.out
# Adjust --partition/--account/module loads for the target cluster (see PERF_PLAN.md).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# module load python/3.12   # cluster-specific
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p benchmarks/results/cluster
for th in 1 4 16; do
  python benchmarks/workloads/run_workloads.py --device cpu --threads $th --repeat 2 \
    --out benchmarks/results/cluster/cpu_threads${th}_${SLURM_JOB_ID:-local}.json
done
python benchmarks/workloads/run_workloads.py --device cpu --threads ${SLURM_CPUS_PER_TASK:-16} --profile \
  --out benchmarks/results/cluster/cpu_profile_${SLURM_JOB_ID:-local}.json
