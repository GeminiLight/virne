# Basic Usage

## Installation

### Complete installation

```shell
# only cpu
bash requirements.sh -c 0

# use cuda (optional version: 10.2, 11.3)
bash requirements.sh -c 11.3
```

### Selective installation

Necessary

```shell
pip install networkx numpy pandas ortools matplotlib pyyaml
```

Expansion

```shell
# for deep learning
# use cuda
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# only cpu
conda install pytorch torchvision torchaudio -c pytorch

# for reinforcement learning
pip install gym stable_baselines3 sb3_contrib

# for graph neural network
conda install pyg -c pyg -c conda-forge
```


## Minimal Example

```Python
# get config
config = get_config()

# generate p_net and v_net dataset
Generator.generate_dataset(config)

# create scenario with Env and Solver
scenario = create_scenario(config)

# use Solver in Env
scenario.run()
```