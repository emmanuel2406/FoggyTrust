#!/bin/bash
#SBATCH --job-name=foggytrust-exp
#SBATCH --mail-user=emmanuel_rassou@college.harvard.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/n/netscratch/kdbrantley_lab/Lab/erassou/scrap/FoggyTrust/slurm_logs/%x-%j.out
#SBATCH --error=/n/netscratch/kdbrantley_lab/Lab/erassou/scrap/FoggyTrust/slurm_logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner
#SBATCH --account=kempner_undergrads

set -euo pipefail
PROJECT_DIR="/n/netscratch/kdbrantley_lab/Lab/erassou/scrap/FoggyTrust"
mkdir -p "${PROJECT_DIR}/slurm_logs"
cd "${PROJECT_DIR}"
# Activate conda env (avoid sourcing interactive bashrc under `set -u`)
source /n/home11/erassou/.conda/etc/profile.d/conda.sh
conda activate fltrust

echo "===== Slurm job context ====="
echo "whoami: $(whoami)"
echo "USER: ${USER:-unknown}"
echo "HOSTNAME: ${HOSTNAME:-unknown}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-unknown}"
echo "SLURM_JOB_ACCOUNT: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================="

# Optional sanity check
python -c "import sys, numpy; print('python:', sys.version); print('numpy:', numpy.__version__)"
nvidia-smi --query-gpu=name --format=csv,noheader
run_cmd () {
  echo "============================================================"
  echo "START: $*"
  date
  "$@"
  echo "DONE : $*"
  date
}
run_cmd python test_byz_all.py --dataset SnapshotSafari --lr 0.01 --batch_size 32 --nworkers 90 --nbyz 18 --niter 2000 --snapshot_metadata_path ../data/snapshot/snapshot_safari_2024_metadata.json --snapshot_images_root ../data/snapshot/images --snapshot_subset_projects KAR,KRU,SER --snapshot_min_category_frequency 20 --snapshot_max_train_samples 12000 --snapshot_max_test_samples 3000 --snapshot_split_seed 7 --max_workers 10 