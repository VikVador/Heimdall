#!/bin/bash
#SBATCH --job-name=heimdall-environment
#SBATCH --account=torch_pr_347_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/environment.log

# Environment overlay, its NYU template (15 GB, 500K files) and container image
OVERLAY=$SCRATCH/env/heimdall.ext3
TEMPLATE=/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

# Creating the overlay from the template, only once
if [ ! -f $OVERLAY ]; then
    mkdir -p $(dirname $OVERLAY)
    cp $TEMPLATE $OVERLAY.gz
    gunzip $OVERLAY.gz
fi

# Installing the environment, the overlay being locked while mounted read-write (no other job must use it)
apptainer exec --fakeroot --overlay $OVERLAY:rw $IMAGE bash cluster/environment/install.sh

# Checking the environment, mounted read-only as in the other jobs
apptainer exec --overlay $OVERLAY:ro $IMAGE bash -c "
    source /ext3/env.sh
    python -c 'import albus, azula, heimdall, shaggy, torch; print(torch.__version__, torch.version.cuda)'
"
