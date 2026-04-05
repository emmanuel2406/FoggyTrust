#!/bin/bash
# Create FLTrust conda environment using mamba (for SLURM clusters)
# All env/cache files live in home directory only; nothing created in project dir

set -e
module load Mambaforge/23.11.0-fasrc01

if [ -f "/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh" ]; then
    source "/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ENV_PREFIX="$HOME/.conda/envs/fltrust"
mkdir -p "$(dirname "$ENV_PREFIX")"

mamba create --prefix "$ENV_PREFIX" python=3.9 -y
mamba activate "$ENV_PREFIX"
# NCCL required by mxnet-cu112 (libnccl.so.2)
mamba install -c conda-forge nccl -y
pip install -r "$PROJECT_DIR/requirements.txt"

echo "Created env at $ENV_PREFIX"
echo "Run 'source scripts/activate_env.sh' to activate."

# Delete env
# conda env remove -p $HOME/.conda/envs/fltrust -y