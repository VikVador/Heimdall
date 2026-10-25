#!/bin/bash
#SBATCH --job-name=heimdall-encoder-observations
#SBATCH --account=torch_pr_347_courant
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --time=47:59:00
#SBATCH --output=cluster/logs/%x-%A_%a.out

# Datasets of the array, one GPU each
DATASETS=(sst sss-asc sss-des wind-asc wind-des chl)

# Environment overlay and container image
OVERLAY=$SCRATCH/env/heimdall.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

# Each task of the array uses its own port, since they share SLURM_JOBID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + $(echo -n $SLURM_JOBID | tail -c 4) + SLURM_ARRAY_TASK_ID))
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd experiments/forecast-latent-states

srun apptainer exec --nv --overlay $OVERLAY:ro $IMAGE bash -c "
    source /ext3/env.sh
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID python -u encoders.py \
        --config configs/encoder/encoder-${DATASETS[$SLURM_ARRAY_TASK_ID]}.yml
"
