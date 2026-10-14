#!/bin/bash
#SBATCH --job-name=heimdall-encoders
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --constraint=rtx6000
#SBATCH --cpus-per-task=14
#SBATCH --mem=1400G
#SBATCH --time=47:59:00
#SBATCH --output=cluster/logs/%x-%A_%a.out

# TODO | Modalities of the array (as in heimdall.data.MODALITIES)
MODALITIES=(glorys sst sla)

# Environment overlay and container image
OVERLAY=$SCRATCH/env/heimdall.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

# Distributed | Master on the first node, a port per task of the array (they share SLURM_JOBID)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + $(echo -n $SLURM_JOBID | tail -c 4) + SLURM_ARRAY_TASK_ID))
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd experiments/forecast-latent-states

srun apptainer exec --nv --overlay $OVERLAY:ro $IMAGE bash -c "
    source /ext3/env.sh
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID python -u encoders.py \
        --config configs/encoders.yml \
        --modality ${MODALITIES[$SLURM_ARRAY_TASK_ID]}
"
