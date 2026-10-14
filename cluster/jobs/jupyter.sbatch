#!/bin/bash
#SBATCH --job-name=heimdall-jupyter
#SBATCH --account=torch_pr_347_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=cluster/logs/jupyter.log

# Environment overlay and container image
OVERLAY=$SCRATCH/env/heimdall.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

apptainer exec --overlay $OVERLAY:ro $IMAGE bash -c "
    source /ext3/env.sh
    jupyter lab --ip='*' --no-browser --port=8888 --notebook-dir=$SLURM_SUBMIT_DIR
"
