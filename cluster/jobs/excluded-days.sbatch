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
#SBATCH --job-name=heimdall-excluded-days
#SBATCH --account=torch_pr_347_general
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=cluster/logs/%x-%A_%a.out

# Observation products of the array, GLORYS having no day without values
DATASETS=(sst sss_asc sss_des wind_asc wind_des chl)

# Environment overlay and container image
OVERLAY=$SCRATCH/env/heimdall.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

apptainer exec --overlay $OVERLAY:ro $IMAGE bash -c "
    source /ext3/env.sh
    python -u experiments/data/excluded_days.py --modality ${DATASETS[$SLURM_ARRAY_TASK_ID]} --workers \$SLURM_CPUS_PER_TASK
"
