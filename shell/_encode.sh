# Moving to the root of the repository, whatever the current directory
cd "$(dirname "$0")/.."

# Launching the trainings of the encoders, one per modality
sbatch cluster/jobs/encoders.sbatch

# Checking the job once (Torch asks not to poll Slurm with watch)
squeue --me
