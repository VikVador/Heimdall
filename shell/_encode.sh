# Moving to the root of the repository, whatever the current directory
cd "$(dirname "$0")/.."

# Launching the trainings of the encoders, GLORYS on 8 GPUs and each observation dataset on 1 GPU (14 of the 16 GPUs
# allowed per user)
sbatch cluster/jobs/encoder-glorys.sbatch
sbatch cluster/jobs/encoder-observations.sbatch

# Checking the jobs once (Torch asks not to poll Slurm with watch)
squeue --me
