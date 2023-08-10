#!/bin/bash

supported_cuda_versions=("false" "0" "10.2" "11.3")
cuda=""

while getopts ":c:" opt; do
    case $opt in
    c)
        if [[ " ${supported_cuda_versions[*]} " == *" ${OPTARG} "* ]]; then
            cuda="${OPTARG}"
        fi
        ;;
    :)
        echo "Install pytorch with cpu version."
        ;;
    \?)
        echo "Invalid option: -$OPTARG."
        ;;
    esac
done

if [[ " ${supported_cuda_versions[*]} " == *" ${cuda} "* ]]; then
    echo "Install pytorch with cuda ${cuda} version."
else
    echo "Install pytorch with cpu version."
fi

# Basic dependencies
pip install numpy pandas matplotlib networkx pyyaml tqdm ortools colorama

# PyTorch installation
pytorch_install_command="pytorch==1.11.0"
if [[ "${cuda}" != "" ]]; then
    pytorch_install_command+=" cudatoolkit=${cuda}"
fi
echo -e "y" | conda install ${pytorch_install_command} -c pytorch


# Additional packages
pip install tensorboard
echo -e "y" | conda install pyg -c pyg -c conda-forge
pip install gym==0.21.0
pip install --force-reinstall scipy