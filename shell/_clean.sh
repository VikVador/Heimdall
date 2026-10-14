# Moving to the root of the repository, whatever the current directory
cd "$(dirname "$0")/.."

# Removes the WandB files and the logs of the jobs
rm -rf wandb/
rm -rf experiments/forecast-latent-states/wandb/
rm -f cluster/logs/*.out cluster/logs/*.log
