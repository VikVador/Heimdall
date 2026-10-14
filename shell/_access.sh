# Moving to the root of the repository, whatever the current directory
cd "$(dirname "$0")/.."

# Retrieving the node and the link of the jupyter notebook (its port changes if 8888 is already taken on the node)
node=$(squeue --me --name=heimdall-jupyter --states=RUNNING --noheader --format=%N | head -n 1)
link=$(grep -oE "http://127.0.0.1:[0-9]+/lab\?token=[0-9a-f]+" cluster/logs/jupyter.log | tail -n 1)
port=$(echo "$link" | grep -oE ":[0-9]+/" | tr -d ":/")

# Prints the link to use in VSCode, connected to the login node (Select Kernel > Existing Jupyter Server)
echo "${link/127.0.0.1/$node}"

# Prints the ssh command and the link to use from your computer (Host torch in ~/.ssh/config)
printf "ssh -NL %s:%s:%s torch\n" "$port" "$node" "$port"
echo "${link/127.0.0.1/localhost}"
