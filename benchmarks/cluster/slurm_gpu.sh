#!/bin/bash
#SBATCH --job-name=torchspin-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/results/cluster/%x-%j.out
# Adjust --partition/--account/--gres and module loads for the target cluster (see PERF_PLAN.md).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# module load cuda/12.4 python/3.12   # cluster-specific
source .venv/bin/activate 2>/dev/null || true
mkdir -p benchmarks/results/cluster
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
python -m pytest torchspin/tests/test_gpu_consistency.py -q -p no:cacheprovider
python benchmarks/workloads/run_workloads.py --device cuda --repeat 2 --out benchmarks/results/cluster/gpu_${SLURM_JOB_ID:-local}.json
python benchmarks/workloads/run_workloads.py --device cpu  --repeat 2 --out benchmarks/results/cluster/gpu-node-cpu_${SLURM_JOB_ID:-local}.json
python benchmarks/workloads/run_workloads.py --device cuda --profile --out benchmarks/results/cluster/gpu_profile_${SLURM_JOB_ID:-local}.json
