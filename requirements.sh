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
pip install numpy pandas matplotlib networkx

# for DL
if [[ "${supported_cuda_versions[@]}" =~ "${cuda}" ]] ; then
    conda install pytorch torchvision torchaudio cudatoolkit=${cuda} -c pytorch
else
    conda install pytorch torchvision torchaudio -c pytorch
fi

pip install tensorboard
# for GNN 
conda install pyg -c pyg -c conda-forge
# for RL
pip install gym stable_baselines3 sb3_contrib