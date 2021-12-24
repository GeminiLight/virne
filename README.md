# Virne

> Developing ...

**Virne** is a framework for Virtual Network Embedding (VNE) problem with the following characteristics:

- **Lightweight**: Environments and algorithms are implemented concisely, using three necessary dependencies (networkx, numpy, pandas).
- **Develop efficiently**: General operation methods are implemented uniformly and several environments for RL are supplied in gym.Env-style.
- **Rich implementations**: Various algorithms are preset here and unified interfaces for calling are provided.

Supported features

- Pre-implemented attributes: node-resource, edge-resource, node-extrema, edge-extrema, node-position
- Customize node/ edge/ graph attributes for resources/ constraints
- Provide serval Environment class for Reinforcement Learning in gym.Env-style
- Reject Early some not cost-effective virtual networks (Admission control)

## Implemented Algorithms

**Virne** has implemented the following heuristic-based and learning-based algorithms:

### Learning-based

| Name                           | Command                | Type         | Mapping  | Title                                                        | Publication    | Year |
| ------------------------------ | ---------------------- | ------------ | ------------------------------------------------------------ | -------------- | ---- | ---- |
| A3C-Seq2Seq*       | `a3c_seq2seq` | `learning`   | `joint_pr` | [DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9500964) | ICC      | 2021 |
| PG-Seq2Seq      | `pg_seq2seq` | `learning`   | `joint_pr` | [A Continuous-Decision Virtual Network Embedding Scheme Relying on Reinforcement Learning](https://ieeexplore.ieee.org/document/8982091) | TNSM   | 2020 |
| SRL-Seq2Seq*             | `srl_seq2seq` | `learning`   | `joint_pr` | [Virtual Network Function Placement Optimization With Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/8945291/) | JSAC           | 2020 |
| A3C-GCN                        | `a3c_gcn`              | `learning`   | `joint_pr` | [Automatic Virtual Network Embedding: A Deep Reinforcement Learning Approach With Graph Convolutional Networks](https://ieeexplore.ieee.org/document/9060910) | JSAC           | 2020 |
| Graph-ViNE                     | `graph_vine`           | `learning`   | `bfs_trials` | [Accelerating Virtual Network Embedding with Graph Neural Networks](https://ieeexplore.ieee.org/document/9269128) | CNSM           | 2020 |
| PG-MLP                | `pg_mlp`   | `learning`   | `joint_pr` | [NFVdeep: adaptive online service function chain deployment with deep reinforcement learning](http://ieeexplore.ieee.org/document/9068634/). | IWQOS          | 2019 |
| ReINFORCE-Conv | `reinforce_conv`            | `learning`   | `two-stage` | [A Novel Reinforcement Learning Algorithm for Virtual Network Embedding](https://bura.brunel.ac.uk/bitstream/2438/17673/1/FullText.pdf) | Neurocomputing | 2018 |
| MCTS-ViNE                   | `mcts_vine`              | `learning`   | `two-stage` | [Virtual Network Embedding via Monte Carlo Tree Search](https://www.researchgate.net/profile/Ljiljana-Trajkovic/publication/313873926_Virtual_Network_Embedding_via_Monte_Carlo_Tree_Search/links/5ac0386945851584fa7404f4/Virtual-Network-Embedding-via-Monte-Carlo-Tree-Search.pdf?_sg%5B0%5D=IbJ7vUDENmXiBbfMTzU7pe38Z0gve9tpmZe8Z0178rNWQVa5y6AFGJksV2UA1gPa2Fiohm7X1HzI-1rdAPT5Jg.Edi8Rb3R7d-SAgZ4Jl6Z-AnccOosuWHRn2EFIt8dcGLqnDdaw8vBfh1mKV-HieWT8lpuArIMwCjnyAg4CflgVw.cWgci1nNGkvx6bRqmirSaRRk-bi80Q0gMjvmyL49gbkiYRuKU6Zu1Aswe4xTxC99BNyBH7dYbFH3YyQTzUJczg&_sg%5B1%5D=XE66L-R7TPh36UxeMPExdBq5KyXxwAikDWvZbhvLjlAdwbBQ3MNiZbmBZzwQ0L1ntkXedGL1rZZYqX6LhuHdgQbg5Xi8I7phGNSAPGvh1OJv.Edi8Rb3R7d-SAgZ4Jl6Z-AnccOosuWHRn2EFIt8dcGLqnDdaw8vBfh1mKV-HieWT8lpuArIMwCjnyAg4CflgVw.cWgci1nNGkvx6bRqmirSaRRk-bi80Q0gMjvmyL49gbkiYRuKU6Zu1Aswe4xTxC99BNyBH7dYbFH3YyQTzUJczg&_iepl=) | TCYB           | 2017 |

> `*` means that the algorithm only supports chain-shape virtual networks embedding

### Heuristics-based

| Name                           | Command       | Type         | Mapping      | Title                                                        | Publication | Year |
| ------------------------------ | ------------- | ------------ | ------------ | ------------------------------------------------------------ | ----------- | ---- |
| NRM                            | `nrm_rank`    | `heuristics` | `two-stage`  | [Virtual Network Embedding Based on Computing, Network, and Storage Resource Constraints](https://ieeexplore.ieee.org/document/7976281) | IoTJ        | 2018 |
| GRC (Global resource capacity) | `grc_rank`    | `heuristics` | `two-stage`  | [Toward Profit-Seeking Virtual Network Embedding Algorithm via Global Resource Capacity](https://ieeexplore.ieee.org/document/6847918) | INFOCOM     | 2014 |
| RW-MaxMatch (NodeRank)         | `nr_rank`     | `heuristics` | `two-stage`  | [Virtual Network Embedding Through Topology-Aware Node Ranking](https://dl.acm.org/doi/10.1145/1971162.1971168) | SIGCOMM     | 2011 |
| RW-BFS (NodeRank)              | `nr_rank_bfs` | `heuristics` | `bfs_trials` | [Virtual Network Embedding Through Topology-Aware Node Ranking](https://dl.acm.org/doi/10.1145/1971162.1971168) | SIGCOMM     | 2011 |

### Simple Baselines

| Name                                          | Command             | Mapping      |
| --------------------------------------------- | ------------------- | ------------ |
| Random Rank                                   | `random_rank`       | `two-stage`  |
| Random Joint Place and Route                  | `random_joint_pr`   | `joint_pr`   |
| Random Rank Breath First Search               | `random_bfs_trials` | `bfs_trials` |
| Order Rank                                    | `order_rank`        | `two-stage`  |
| Order Joint Place and Route                   | `order_joint_pr`    | `joint_pr`   |
| Order Rank Breath First Search                | `order_bfs_trials`  | `bfs_trials` |
| First Fit Decreasing Rank                     | `ffd_rank`          | `two-stage`  |
| First Fit Decreasing Joint Place and Route    | `ffd_joint_pr`      | `joint_pr`   |
| First Fit Decreasing Rank Breath First Search | `ffd_bfs_trials`    | `bfs_trials` |

## File Structure

```plain text
virne
│
├─algo
│  ├─base
│  ├─learning
│  ├─heuritics
│  └─rank
│
├─data
│  ├─attribute.py
│  ├─generator.py
│  ├─network.py
│  ├─physical_network.py
│  ├─virtual_network.py
│  ├─vn_simulator.py
│  └─utils.py
│
├─dataset
│  ├─pn
│  └─vns
│
├─records
│  ├─..
│  └─global_summary.csv
│
├─settings
│  ├─vns_setting.json
│  └─pn_setting.json
│
├─tester
│  └─tester.py
│
├─save
├─logs
│
├─config.py
├─utils.py
└─main.py
```

## Quick Start

The structure of this framework are still optimized steadily. We will construct the first version of the document as soon as possible until stability is ensured.

### Requirements

#### Complete installation

```powershell
pip install -r requirements.txt
```

#### Selective installation

Necessary

```powershell
pip install networkx, numpy, pandas
```

Expansion

```shell
# for data analysis
pip install matplotlib

# for deep learning
pip install torch

# for reinforcement learning
pip install gym                # rl environment
pip install stable-baselines3  # rl algorithms, some drl-based algos use it

# for graph neural network
pip install torch_geometric
```

### Simple Example

```Python
# get config
config = get_config()

# generate dataset
Generator.generate_dataset(config)

# load environment and algorithm
env, agent = load_algo(config)

# run with algorithm
agent.run(env)
```

## Document

Under constructing ...

## To-do List

### Environment Modeling

- [ ] `ADD` `Environment` Latency Constraint
- [ ] `ADD` `Controller` Check graph constraints
- [ ] `ADD` `Controller` Multi-commodity flow
- [ ] `ADD` `Recorder` Count partial solutions' information
- [x] `ADD` `Enironment` Early rejection (Admission control)
- [x] `ADD` `Environment` Multi-Resources Attributes
- [x] `ADD` `Environment` Position Constraint
- [x] `ADD` `Recorder` Count Running physical network nodes

### Algorithm Implementations

| Name            | Command          | Type         | Mapping     | Title                                                        | Publication |      |
| --------------- | ---------------- | ------------ | ----------- | ------------------------------------------------------------ | ----------- | ---- |
| DDPG-Attention* | `ddpg_attention` | `learning`   | `joint`     | [A-DDPG: Attention Mechanism-based Deep Reinforcement Learning for NFV](https://research.tudelft.nl/en/publications/a-ddpg-attention-mechanism-based-deep-reinforcement-learning-for-) | IWQOS       | 2021 |
| MUVINE          | `mu_vne`        | `learning`   | `joint`     | [MUVINE: Multi-stage Virtual Network Embedding in Cloud Data Centers using Reinforcement Learning based Predictions](https://arxiv.org/pdf/2111.02737.pdf) | JSAC        | 2020 |
| TD-VNE          | `td_vne`         | `learning`   | `two-stage` | [VNE-TD: A virtual network embedding algorithm based on temporal-difference learning](https://arxiv.org/pdf/2111.02737.pdf) | CN          | 2019 |
| D-ViNE/ R-ViNE  | `d_vne`          | `heuristics` | `joint`     | [ViNEYard: Virtual Network Embedding Algorithms With Coordinated Node and Link Mapping](https://ieeexplore.ieee.org/document/5951812?arnumber=5951812) | TON         | 2012 |

### Data Visualization

- [ ] use Neo4j Database

### Module Testing

#### config

- [x] `config`

#### data

- [ ] `data.attribute`
- [x] `data.network`
- [x] `data.physical_network`
- [x] `data.virtual_network`
- [ ] `data.vn_simulator`

#### algo

- [x] `algo.base.recorder`
- [x] `algo.base.controller`
- [x] `algo.base.enviroment`
- [x] `algo.base.agent`
- [x] `algo.rank.node_rank`
- [x] `algo.rank.edge_rank`
- [x] `algo.heuristics`
- [x] `algo.learning.mcts`
- [ ] ...
