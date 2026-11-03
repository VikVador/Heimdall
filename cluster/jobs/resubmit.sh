#!/bin/bash
#
# Sourced by the jobs of a training, with SCRIPT set to their own path. Submits the job that resumes this training if
# this one ends without success (time limit, crash, lost node). It is dropped by Slurm if this one completes.

export RUN_ID=${RUN_ID:-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}}
RESTARTS=${RESTARTS:-0}

if ((RESTARTS < ${MAX_RESTARTS:-8})); then
    sbatch --dependency=afternotok:$SLURM_JOB_ID --kill-on-invalid-dep=yes \
        --export=ALL,RUN_ID=$RUN_ID,RESTARTS=$((RESTARTS + 1)) \
        ${SLURM_ARRAY_TASK_ID:+--array=$SLURM_ARRAY_TASK_ID} "$SCRIPT"
fi
