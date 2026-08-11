#!/usr/bin/env bash
set -euo pipefail

readonly TORCH_VERSION="2.11.0"
readonly PYG_VERSION="2.8.0.post1"
readonly SUPPORTED_ACCELERATORS="cpu 12.6 12.8 13.0"

accelerator="cpu"
virne_python="${VIRNE_PYTHON:-python3}"

usage() {
    echo "Usage: $0 [-c cpu|12.6|12.8|13.0]"
    echo "Set VIRNE_PYTHON to choose the Python executable (default: python3)."
}

while getopts ":c:h" opt; do
    case "${opt}" in
    c)
        accelerator="${OPTARG}"
        ;;
    h)
        usage
        exit 0
        ;;
    :)
        echo "Option -${OPTARG} requires an argument." >&2
        usage >&2
        exit 1
        ;;
    \?)
        echo "Invalid option: -${OPTARG}." >&2
        usage >&2
        exit 1
        ;;
    esac
done

if [[ " ${SUPPORTED_ACCELERATORS} " != *" ${accelerator} "* ]]; then
    echo "Unsupported accelerator '${accelerator}'. Supported values: ${SUPPORTED_ACCELERATORS}." >&2
    exit 1
fi

if ! command -v "${virne_python}" >/dev/null 2>&1; then
    echo "Python executable '${virne_python}' was not found." >&2
    exit 1
fi

platform="$(uname -s)"
if [[ "${platform}" == "Darwin" && "${accelerator}" != "cpu" ]]; then
    echo "CUDA builds are not available on macOS. Use '-c cpu'." >&2
    exit 1
fi

case "${accelerator}" in
cpu)
    torch_index="https://download.pytorch.org/whl/cpu"
    pyg_wheel_tag="cpu"
    ;;
12.6)
    torch_index="https://download.pytorch.org/whl/cu126"
    pyg_wheel_tag="cu126"
    ;;
12.8)
    torch_index="https://download.pytorch.org/whl/cu128"
    pyg_wheel_tag="cu128"
    ;;
13.0)
    torch_index="https://download.pytorch.org/whl/cu130"
    pyg_wheel_tag="cu130"
    ;;
esac

echo "Installing Virne with PyTorch ${TORCH_VERSION}, PyG ${PYG_VERSION}, accelerator ${accelerator}."
"${virne_python}" -m pip install --upgrade pip

if [[ "${platform}" == "Darwin" ]]; then
    "${virne_python}" -m pip install "torch==${TORCH_VERSION}"
else
    "${virne_python}" -m pip install "torch==${TORCH_VERSION}" --index-url "${torch_index}"
fi

# pyproject.toml is the source of truth for PyG and all remaining dependencies.
"${virne_python}" -m pip install --editable .

# PyG can run without compiled extensions, but Virne installs the supported
# acceleration and SparseTensor packages. Remove old wheels first because their
# version numbers do not encode the PyTorch ABI they were compiled against.
"${virne_python}" -m pip uninstall --yes \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
"${virne_python}" -m pip install pyg_lib torch_scatter torch_sparse \
    --find-links "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${pyg_wheel_tag}.html"

"${virne_python}" -c \
    "import gymnasium, torch, torch_geometric, virne; print(f'Virne {virne.__version__}; Gymnasium {gymnasium.__version__}; PyTorch {torch.__version__}; PyG {torch_geometric.__version__}')"
