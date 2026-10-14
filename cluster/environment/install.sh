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
# This script installs the Python environment of Heimdall, i.e. conda, the `heimdall` environment and all
# the packages, into a single file, the overlay, that every job then uses. You can run it again at any
# time, e.g. after changing the dependencies, as it only installs what is missing.

set -e

# Installing Miniforge
if [ ! -d /ext3/miniforge3 ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p /ext3/miniforge3
    rm /tmp/miniforge.sh
fi

# Creating the wrapper used to activate the environment
cat > /ext3/env.sh <<'EOT'
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
export PYTHONNOUSERSITE=True
conda activate heimdall
EOT

# Creating the conda environment
source /ext3/miniforge3/etc/profile.d/conda.sh
conda env list | grep -q "^heimdall " || conda create -y -n heimdall python=3.11 git
conda activate heimdall

# Installing heimdall as an editable package with its dependencies
pip install --editable .[all] --extra-index-url https://download.pytorch.org/whl/cu130

# Cleaning caches to save space and inodes
conda clean --all --yes
pip cache purge
