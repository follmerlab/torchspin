#!/bin/bash
# Serial benchmark campaign on a single workstation (no scheduler): MATLAB, torchspin CPU
# (thread scaling), torchspin GPU, multi-process CPU parallelism, profiles.  One job at a time.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
source ~/miniconda3/etc/profile.d/conda.sh && conda activate torchspin
ES=${EASYSPIN:-$HOME/code/EPR/EasySpin/easyspin}
# Overrides: OUT=<dir> RUN_MATLAB=0 RUN_PROFILES=0 (e.g. an "after" run of the torchspin stages only)
OUT=${OUT:-benchmarks/results/workstation_$(date +%Y%m%d)}; mkdir -p $OUT
RUN_MATLAB=${RUN_MATLAB:-1}; RUN_PROFILES=${RUN_PROFILES:-1}
unset DISPLAY   # MATLAB -batch aborts on an unreachable X display
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))" | tee $OUT/env.txt
nvidia-smi --query-gpu=name,driver_version --format=csv >> $OUT/env.txt; nproc >> $OUT/env.txt
python -m pytest torchspin/tests/test_gpu_consistency.py -q -p no:cacheprovider | tail -2 | tee $OUT/gpu_consistency.txt
for th in 1 8 32; do
  python benchmarks/workloads/run_workloads.py --device cpu --threads $th --repeat 1 --out $OUT/cpu_threads${th}.json
done
python benchmarks/workloads/run_workloads.py --device cuda --repeat 2 --out $OUT/gpu.json
for np_ in 4 16 64; do
  python benchmarks/workloads/run_workloads.py --device cpu --procs $np_ --threads 1 --out $OUT/cpu_procs${np_}.json
done
if [ "$RUN_PROFILES" = 1 ]; then
  python benchmarks/workloads/run_workloads.py --device cpu --threads 32 --profile --only pepper_cu_2n_matrix,pepper_mn_S52_matrix,cardamom_diffusion_200x1000,chili_nitroxide_2nuc,chili_powder_potential --out $OUT/cpu_profile.json
  python benchmarks/workloads/run_workloads.py --device cuda --profile --only pepper_cu_2n_matrix,pepper_mn_S52_matrix,pepper_strain_summation --out $OUT/gpu_profile.json
fi
if [ "$RUN_MATLAB" = 1 ]; then
  (cd benchmarks/workloads && matlab -batch "addpath(genpath('$ES')); matlab_workloads" | tee ../../$OUT/matlab_workloads.txt; cp matlab_workloads.json ../../$OUT/)
fi
echo "campaign done: $OUT"
