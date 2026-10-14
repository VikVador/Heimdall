# Moving to the root of the repository, whatever the current directory
cd "$(dirname "$0")/.."

# Launching a jupyter notebook
sbatch cluster/jobs/jupyter.sbatch

# Checking the job once (Torch asks not to poll Slurm with watch)
squeue --me
