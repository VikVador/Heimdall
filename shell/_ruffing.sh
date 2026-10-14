# Moving to the root of the repository, whatever the current directory
cd "$(dirname "$0")/.."

# Linting and formatting with Ruff
ruff check --fix
ruff format
