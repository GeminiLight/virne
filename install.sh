#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[Step] $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for CUDA GPU
check_cuda_gpu() {
    if command_exists nvidia-smi; then
        return 0
    else
        return 1
    fi
}

get_nvcc_version() {
    if command_exists nvcc; then
        nvcc --version | grep "release" | awk '{print $6}' | cut -c2- | cut -d. -f1-2
    else
        echo "N/A"
    fi
}

# Check nvcc version
check_nvcc_version() {
    if command_exists nvcc; then
        nvcc_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        nvcc_major=$(echo $nvcc_version | cut -d. -f1)
        nvcc_minor=$(echo $nvcc_version | cut -d. -f2)
        if [[ "$nvcc_major" -gt 12 || ( "$nvcc_major" -eq 12 && "$nvcc_minor" -ge 4 ) ]]; then
            return 0
        else
            return 1
        fi
    else
        return 1
    fi
}



# Supported CUDA versions (use "cpu" for no CUDA)
supported_cuda_versions=("cpu" "10.2" "11.3" "11.6" "11.7" "12.0" "12.1" "12.2" "12.3" "12.4")
cuda=""

# Parse options
while getopts ":c:" opt; do
    case $opt in
    c)
        if [[ " ${supported_cuda_versions[*]} " == *" ${OPTARG} "* ]]; then
            cuda="${OPTARG}"
        else
            echo "Warning: Unsupported CUDA version '${OPTARG}'. Falling back to auto-detect."
            cuda=""
        fi
        ;;
    :)
        echo "Option -$OPTARG requires an argument."
        exit 1
        ;;
    \?)
        echo "Invalid option: -$OPTARG."
        exit 1
        ;;
    esac
done

print_step "Checking prerequisites..."
if ! command_exists conda; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi
if ! command_exists pip; then
    echo "pip is not installed. Please install pip first."
    exit 1
fi

print_step "Upgrading pip..."
pip install --upgrade pip

print_step "Installing basic dependencies..."
pip install numpy pandas matplotlib networkx
pip install pyyaml tqdm colorama hydra-core colorlog wandb
pip install ortools scikit-learn

# Auto-detect CUDA if not specified
if [[ -z "$cuda" ]]; then
    if check_cuda_gpu; then
        print_step "NVIDIA GPU detected."
        if check_nvcc_version; then
            print_step "CUDA >= 12.4 detected."
            nvcc_version=$(get_nvcc_version)
            cuda=12.4
            echo "INFO: CUDA $nvcc_version detected."
        else
            print_step "CUDA < 12.4 or nvcc not found. Installing CUDA toolkit 12.4 via conda."
            conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
            export CUDA_HOME=$CONDA_PREFIX
            echo "INFO: CUDA 12.4 installed."
        fi
    else
        print_step "No NVIDIA GPU detected. Installing CPU version."
        cuda="cpu"
    fi
fi

# always exit here

print_step "Installing PyTorch..."
pip install tensorboard higher
if [[ "${cuda}" == "cpu" ]]; then
    echo "Installing PyTorch (CPU version)..."
    pip install torch==2.6.0  --index-url https://download.pytorch.org/whl/cpu --force-reinstall
else
    echo "Installing PyTorch with CUDA ${cuda}..."
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
fi

print_step "Installing additional packages..."
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install gym==0.22.0
pip install --force-reinstall scipy

echo -e "${GREEN}Installation complete.${NC}"
