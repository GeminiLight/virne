# Virne

**Virne** is a framework for Virtual Network Embedding (VNE) problem with the following characteristics:

- **Lightweight**: Environments and algorithms are implemented concisely, using three necessary dependencies (networkx, numpy, pandas).
- **Develop efficiently**: General operation methods are implemented uniformly and several environments for RL are supplied in gym.Env-style.
- **Rich implementations**: Various algorithms are preset here and unified interfaces for calling are provided.

Supported features

- **Diverse Network Topologies**
  - Physical network: Waxman graph
  - Virtual network: Path graph, Edge probabilistic connection graph
- **Graph/ Node / Link-level Attributes**: 
  - For resources/ constraints/ QoS
  - Graph Level: e.g. the global requirements of virtual network
  - Node level: e.g. Node resource, node position
  - Link level: e.g. Link resource, link latency
- **Multiple RL Environments**
  - Provide serval RL Environments in gym.Env-style
- **Various Simulation Scenarios**
  - Admission control: Reject Early some not cost-effective virtual networks
  - Time window: Developping

## VNE Problem

### Brife Defination

![](resource/figures/vne-example.png)

### Main Objectives

- Acceptance rate

$$
\text{Acceptance Rate} = \cfrac{\sum_{t=0}^{t=T} \text{Number}(VNR_{accept})}{\sum_{t=0}^{t=T} \text{Number}(VNR_{reject})}
$$

- Long-term revenue

$$
\text{Long-term Revenue} = \sum_{n_v \in N_v}{Revenue(n_v)} + \sum_{e_v \in E_v}{Revenue(e_v)}
$$

- Revenue-to-cost Ratio

$$
\text{Long-term Revenue} = \cfrac{\text{Long-term Revenue}}{\text{Long-term Cost}}
$$

- Running Time

$$
\text{Running Time}
$$


### QoS Awarenesses (Additional Constraints/ Objectives)

- [x] Position (Node level)
- [x] Latency (Graph, Node and Link level)
- [x] Security (Graph, Node and Link level)
- [ ] Congestion (Graph, Node and Link level)
- [ ] Energy (Graph, Node and Link level)
- [ ] Reliability (Graph, Node and Link level)
- [ ] Dynamic (Graph, Node and Link level)
- [ ] Parallelization
- [ ] Privacy

### Mapping Strategy

- Two-State
  - In this fromework, the VNE solving process are composed of Node mapping and Edge Mapping.
  - Firstly, the node mapping solution is 
- Joint Place and Route
- BFS Trails


## Implemented Algorithms

**Virne** has implemented the following heuristic-based and learning-based algorithms:

### Learning-based Solvers

| Name                           | Command                | Type         | Mapping  | Title                                                        | Publication    | Year | Note |
| ------------------------------ | ---------------------- | ------------ | ------------------------------------------------------------ | -------------- | ---- | ---- | ------------------------------ |
| PG-CNN2 | `pg_cnn2` | `learning`   | `two-stage` | [A Virtual Network EmbeddingAlgorithm Based On Double-LayerReinforcement Learning](https://ieeexplore.ieee.org/document/9500964) | The Computer Journal | 2022 |  |
| A3C-Seq2Seq*       | `a3c_seq2seq` | `learning`   | `joint_pr` | [DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9500964) | ICC      | 2021 |  |
| PG-CNN-QoS | `pg_cnn_qos` | `learning`   | `two-stage` | [Resource Management and Security Scheme of ICPSs and IoT Based on VNE Algorithm](https://arxiv.org/pdf/2202.01375.pdf) | IoTJ | 2021 |  |
| PG-Seq2Seq      | `pg_seq2seq` | `learning`   | `joint_pr` | [A Continuous-Decision Virtual Network Embedding Scheme Relying on Reinforcement Learning](https://ieeexplore.ieee.org/document/8982091) | TNSM   | 2020 |  |
| SRL-Seq2Seq*             | `srl_seq2seq` | `learning`   | `joint_pr` | [Virtual Network Function Placement Optimization With Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/8945291/) | JSAC           | 2020 |  |
| A3C-GCN                        | `a3c_gcn`              | `learning`   | `joint_pr` | [Automatic Virtual Network Embedding: A Deep Reinforcement Learning Approach With Graph Convolutional Networks](https://ieeexplore.ieee.org/document/9060910) | JSAC           | 2020 |  |
| GAE-VNE                    | `gae_vne`          | `learning`   | `bfs_trials` | [Accelerating Virtual Network Embedding with Graph Neural Networks](https://ieeexplore.ieee.org/document/9269128) | CNSM           | 2020 | Clustering |
| PG-MLP                | `pg_mlp`   | `learning`   | `joint_pr` | [NFVdeep: adaptive online service function chain deployment with deep reinforcement learning](http://ieeexplore.ieee.org/document/9068634/). | IWQOS          | 2019 |  |
| Neuro-VNE          | `neuro_vne` | `learning`   | `two-stage` | [NeuroViNE: A Neural Preprocessor for Your Virtual Network Embedding Algorithm](https://mediatum.ub.tum.de/doc/1449121/document.pdf) | INFOCOM   | 2018 | Subgraph Extraction |
| PG-CNN | `pg_cnn`          | `learning`   | `two-stage` | [A Novel Reinforcement Learning Algorithm for Virtual Network Embedding](https://bura.brunel.ac.uk/bitstream/2438/17673/1/FullText.pdf) | Neurocomputing | 2018 |  |
| MCTS-VNE                   | `mcts_vne`              | `learning`   | `two-stage` | [Virtual Network Embedding via Monte Carlo Tree Search](https://www.researchgate.net/profile/Ljiljana-Trajkovic/publication/313873926_Virtual_Network_Embedding_via_Monte_Carlo_Tree_Search/links/5ac0386945851584fa7404f4/Virtual-Network-Embedding-via-Monte-Carlo-Tree-Search.pdf?_sg%5B0%5D=IbJ7vUDENmXiBbfMTzU7pe38Z0gve9tpmZe8Z0178rNWQVa5y6AFGJksV2UA1gPa2Fiohm7X1HzI-1rdAPT5Jg.Edi8Rb3R7d-SAgZ4Jl6Z-AnccOosuWHRn2EFIt8dcGLqnDdaw8vBfh1mKV-HieWT8lpuArIMwCjnyAg4CflgVw.cWgci1nNGkvx6bRqmirSaRRk-bi80Q0gMjvmyL49gbkiYRuKU6Zu1Aswe4xTxC99BNyBH7dYbFH3YyQTzUJczg&_sg%5B1%5D=XE66L-R7TPh36UxeMPExdBq5KyXxwAikDWvZbhvLjlAdwbBQ3MNiZbmBZzwQ0L1ntkXedGL1rZZYqX6LhuHdgQbg5Xi8I7phGNSAPGvh1OJv.Edi8Rb3R7d-SAgZ4Jl6Z-AnccOosuWHRn2EFIt8dcGLqnDdaw8vBfh1mKV-HieWT8lpuArIMwCjnyAg4CflgVw.cWgci1nNGkvx6bRqmirSaRRk-bi80Q0gMjvmyL49gbkiYRuKU6Zu1Aswe4xTxC99BNyBH7dYbFH3YyQTzUJczg&_iepl=) | TCYB           | 2018 | MultiThreading Support |

> `*` means that the algorithm only supports chain-shape virtual networks embedding

### Meta-heuristics Solvers

| Name                           | Command       | Type         | Mapping      | Title                                                        | Publication | Year | Note |
| ------------------------------ | ------------- | ------------ | ------------ | ------------------------------------------------------------ | ----------- | ---- | ------------------------------ |
| PSO-VNE          | `pso_vne`         | `meta-heuristics`   | `two-stage` | [Energy-Aware Virtual Network Embedding](https://ieeexplore.ieee.org/document/6709811) | TON         | 2014 | MultiThreading Support |
| ACO-VNE  | `aco_vne`          | `meta-heuristics` | `joint`     | [VNE-AC: Virtual Network Embedding Algorithm Based on Ant Colony Metaheuristic](https://www.gta.ufrj.br/ensino/cpe717-2011/VNE-ICC-1.pdf) | ICC         | 2011 | MultiThreading Support |

### Heuristics-based Solvers

| Name                           | Command       | Type         | Mapping      | Title                                                        | Publication | Year | Note |
| ------------------------------ | ------------- | ------------ | ------------ | ------------------------------------------------------------ | ----------- | ---- | ---- |
| Ego-Network                    | `ego_vne`     | `heuristics` | `two-stage`  | [Ego Network-based Virtual Network Embedding Scheme for Revenue Maximization](https://ieeexplore.ieee.org/document/9415185) | ICAIIC      | 2021 |      |
| NRM (Node Resource Management) | `nrm_rank`    | `heuristics` | `two-stage`  | [Virtual Network Embedding Based on Computing, Network, and Storage Resource Constraints](https://ieeexplore.ieee.org/document/7976281) | IoTJ        | 2018 |      |
| GRC (Global resource capacity) | `grc_rank`    | `heuristics` | `two-stage`  | [Toward Profit-Seeking Virtual Network Embedding Algorithm via Global Resource Capacity](https://ieeexplore.ieee.org/document/6847918) | INFOCOM     | 2014 |      |
| RW-MaxMatch (NodeRank)         | `rw_rank`     | `heuristics` | `two-stage`  | [Virtual Network Embedding Through Topology-Aware Node Ranking](https://dl.acm.org/doi/10.1145/1971162.1971168) | SIGCOMM     | 2011 |      |
| RW-BFS (NodeRank)              | `rw_rank_bfs` | `heuristics` | `bfs_trials` | [Virtual Network Embedding Through Topology-Aware Node Ranking](https://dl.acm.org/doi/10.1145/1971162.1971168) | SIGCOMM     | 2011 |      |

### Exact Solvers

| Name                                 | Command | Type    | Mapping | Title                                                        | Publication | Year | Note |
| ------------------------------------ | ------- | ------- | ------- | ------------------------------------------------------------ | ----------- | ---- | ---- |
| D-VNE (Deterministic Rounding Based) | `d_bfs` | `exact` | `joint` | [ViNEYard: Virtual Network Embedding Algorithms With Coordinated Node and Link Mapping](https://ieeexplore.ieee.org/document/5951812?arnumber=5951812) | TON         | 2012 |      |
| R-VNE (Random Rounding Based)        | `d_bfs` | `exact` | `joint` | [ViNEYard: Virtual Network Embedding Algorithms With Coordinated Node and Link Mapping](https://ieeexplore.ieee.org/document/5951812?arnumber=5951812) | TON         | 2012 |      |

### Simple Baseline Solvers

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
├─base
│  ├─recorder
│  ├─controller
│  └─enviroment
│
├─solver
│  ├─exact
│  ├─learning
│  ├─heuritics
│  └─meta_heuritics
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

```shell
# only cpu
sh requirements.sh -c 0

# use cuda (optional version: 10.2, 11.3)
sh requirements.sh -c 11.3
```

#### Selective installation

Necessary

```shell
pip install networkx, numpy, pandas, matplotlib
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

### Minimal Example

```Python
# get config
config = get_config()

# generate pn and vn dataset
Generator.generate_dataset(config)

# create scenario with Env and Solver
scenario = create_scenario(config)

# use Solver in Env
scenario.run()
```

## Document

Under constructing ...

## To-do List

### Environment Modeling

- [ ] `ADD` `Scenario` Window Batch Processing
- [ ] `ADD` `Environment` Check Attributes of pn and vn
- [ ] `ADD` `Environment` Latency Constraint
- [ ] `ADD` `Controller` Check graph constraints
- [ ] `ADD` `Controller` Multi-commodity flow
- [x] `ADD` `Environment` QoS level Constraints
- [x] `ADD` `Recorder` Count partial solutions' information
- [x] `ADD` `Enironment` Early rejection (Admission control)
- [x] `ADD` `Environment` Multi-Resources Attributes
- [x] `ADD` `Environment` Position Constraint
- [x] `ADD` `Recorder` Count Running physical network nodes

### Algorithm Implementations

| Name            | Command          | Type         | Mapping     | Title                                                        | Publication | Year | Note |
| --------------- | ---------------- | ------------ | ----------- | ------------------------------------------------------------ | ----------- | --------------- | --------------- |
| PG-Conv-QoS-Security | `pg_cnn_qs` | `learning`   | `joint`     | [VNE Solution for Network Differentiated QoS and Security Requirements: From the Perspective of Deep Reinforcement Learning](https://arxiv.org/pdf/2202.01362.pdf) | arXiv        |         | Security |
| DDPG-Attention* | `ddpg_attention` | `learning`   | `joint`     | [A-DDPG: Attention Mechanism-based Deep Reinforcement Learning for NFV](https://research.tudelft.nl/en/publications/a-ddpg-attention-mechanism-based-deep-reinforcement-learning-for-) | IWQOS       | 2021   |  |
| MUVINE          | `mu_vne`        | `learning`   | `joint`     | [MUVINE: Multi-stage Virtual Network Embedding in Cloud Data Centers using Reinforcement Learning based Predictions](https://arxiv.org/pdf/2111.02737.pdf) | JSAC        | 2020    | Admission Control |
| TD-VNE      | `td_vne`      | `learning`   | `two-stage` | [VNE-TD: A virtual network embedding algorithm based on temporal-difference learning](https://www.sciencedirect.com/science/article/pii/S138912861830584X?via%3Dihub) | CN          | 2019      |  |
| RNN-VNE          | `rnn_vne`         | `learning`   | `two-stage` | [Boost Online Virtual Network Embedding: Using Neural Networks for Admission Control](https://mediatum.ub.tum.de/doc/1346092/1346092.pdf) | CNSM        | 2016    | Admission Control |

### Module Testing

#### config

- [x] `config`

#### data

- [ ] `data.attribute`
- [x] `data.network`
- [x] `data.physical_network`
- [x] `data.virtual_network`
- [ ] `data.vn_simulator`

#### solver

- [x] `base.recorder`
- [x] `base.controller`
- [x] `base.enviroment`
- [x] `solver.rank.node_rank`
- [x] `solver.rank.edge_rank`
- [x] `solver.heuristics`
- [x] `solver.learning.mcts`
- [ ] ...
