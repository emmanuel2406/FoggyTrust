#!/bin/bash
# Activate FLTrust environment (run with: source scripts/activate_env.sh)
# Loads mamba, CUDA/cuDNN (for mxnet-cu112), and activates the fltrust env

# CUDA + cuDNN 8 required by mxnet-cu112 (GPU build)
# Load cuda first; run 'module avail cuda' / 'module avail cudnn' for other versions
module load cuda/11.3.1-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01

module load Mambaforge/23.11.0-fasrc01

if [ -f "/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh" ]; then
    source "/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

export MXNET_HOME="$HOME/.mxnet"
mamba activate "$HOME/.conda/envs/fltrust"
