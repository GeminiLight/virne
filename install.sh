supported_cuda_versions=(false, 0, 10.2, 11.3)

while getopts ":c:" opt; do
    case $opt in
    c)
        if [[ "${supported_cuda_versions[@]}" =~ "${OPTARG}" ]] ; then
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

if [[ "${supported_cuda_versions[@]}" =~ "${cuda}" ]] ; then
    echo "Install pytorch with cuda ${cuda} version.\n"
else
    echo "Install pytorch with cpu version.\n"
fi

# basic
pip install numpy pandas matplotlib networkx pyyaml tqdm ortools

# for DL
if [[ "${supported_cuda_versions[@]}" =~ "${cuda}" ]] ; then
    echo -e "y" | conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0  cudatoolkit=${cuda} -c pytorch
else
    echo -e "y" | conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
fi

pip install tensorboard
# for GNN 
echo -e "y" | conda install pyg -c pyg -c conda-forge
# for RL
pip install gym==0.19.0 stable_baselines3 sb3_contrib

pip install --force-reinstall scipy