#!/bin/bash
#
#   ~~  ~~~  ~~~   ~~  ~~~~  ~~   ~~~  ~~~   ~~  ~~~  ~~  ~~  ~~~  ~~~   ~~
#        ~~~      .-'''-.      ~~~       .-'''-.      ~~~       .-'''-.
#   _.-'''-._.-'''       '''-._.-'''-._.'       '''-._.-'''-._.'       '''-.
#
#
#                       H   E   I   M   D   A   L   L
#
#
#   ~~  ~~~  ~~~   ~~  ~~~~  ~~   ~~~  ~~~   ~~  ~~~  ~~  ~~  ~~~  ~~~   ~~
#  ~~~~   ~~   ~~~~   ~~~   ~~   ~~~~   ~~   ~~~~   ~~~  ~~ ~~~~   ~~   ~~~~
#
#SBATCH --job-name=heimdall-latents
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=cluster/logs/%x-%A_%a.out

# Datasets of the array, one RTX of the lab each, e.g. sbatch --array=1 cluster/jobs/latents.sbatch for the SST alone
DATASETS=(glorys sst sss_asc sss_des wind_asc wind_des chl)

# Environment overlay and container image
OVERLAY=$SCRATCH/env/heimdall.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

# The job that resumes this encoding if it stops before its end, the days already encoded being skipped
SCRIPT=cluster/jobs/latents.sbatch
source cluster/jobs/resubmit.sh

cd experiments/encoders

apptainer exec --nv --overlay $OVERLAY:ro $IMAGE bash -c "
    source /ext3/env.sh
    python -u latents.py --config configs/latents.yml --modality ${DATASETS[$SLURM_ARRAY_TASK_ID]}
"
