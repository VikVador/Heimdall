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
#SBATCH --job-name=heimdall-statistics
#SBATCH --account=torch_pr_347_general
#SBATCH --array=2000-2021
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=cluster/logs/%x-%A_%a.out

# One task per year of the training period, the dataset given as argument, e.g.
#   sbatch cluster/jobs/statistics.sbatch glorys
#   sbatch --cpus-per-task=4 --mem=32G cluster/jobs/statistics.sbatch sst
# then, once every task has ended, python experiments/data/statistics.py --modality glorys --reduce
MODALITY=$1

# Environment overlay and container image
OVERLAY=$SCRATCH/env/heimdall.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

# A day of GLORYS takes ~31 GB and ~55 s per process (4 processes for 160 GB, ~1.5 h per year); jobs over 6 h
# use the QoS cpu48, cpu_short capping the memory at 120 GB per user
apptainer exec --overlay $OVERLAY:ro $IMAGE bash -c "
    source /ext3/env.sh
    python -u experiments/data/statistics.py --modality $MODALITY --year \$SLURM_ARRAY_TASK_ID --workers 4
"
