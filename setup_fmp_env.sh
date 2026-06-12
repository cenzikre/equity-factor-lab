#!/usr/bin/env bash
set -e

ENV_NAME="fmp_data"
PYTHON_VERSION="3.10"

echo "Initializing conda..."
source /opt/conda/etc/profile.d/conda.sh

if conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    echo "Creating conda env '${ENV_NAME}'..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

echo "Activating env '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

echo "Installing Python packages..."
pip install --upgrade pip
pip install \
    requests \
    aiohttp \
    aiolimiter \
    tenacity \
    tqdm \
    python-dotenv \
    numpy \
    pandas \
    pyarrow \
    matplotlib \
    pytest \
    ipykernel

echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo "Environment '${ENV_NAME}' is ready."
