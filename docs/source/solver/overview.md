# Overview

**Virne** has implemented the following heuristic-based and learning-based algorithms:

> Mapping Strategies
> - Two-Stage
>   - In this fromework, the VNE solving process are composed of Node mapping and Edge Mapping.
>   - Firstly, the node mapping solution is generate with node mapping algorithm, i.e., Node Ranking
>   - Secondly, the BFS algorithm is employed to route the physical link pairs obtained from the node mapping solution. 
> - Joint Place and Route
>   - The solution of node mapping consists of a sequential placement decision.
>   - Simultaneously, the available physical link pairs are routed by BFS algorithm.
> - BFS Trails
>   - Based on breadth-first search, it expands the search space by exploiting the awareness of restarts.

## Simple Baseline Solvers

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

## Learning-based Solvers

| Name                           | Command                | Type         | Mapping  | Title                                                        | Publication    | Year | Note |
| ------------------------------ | ---------------------- | ------------ | ------------------------------------------------------------ | -------------- | ---- | ---- | ------------------------------ |
| PG-CNN2 | `pg_cnn2` | `learning`   | `two-stage` | [A Virtual Network EmbeddingAlgorithm Based On Double-LayerReinforcement Learning](https://ieeexplore.ieee.org/document/9500964) | The Computer Journal | 2022 |  |
| A3C-G3C-Seq2Seq* | `a3c_gcn_seq2seq` | `learning` | `joint_pr`   | [DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9500964)     | ICC         | 2021 |             |
| PG-CNN-QoS | `pg_cnn_qos` | `learning`   | `two-stage` | [Resource Management and Security Scheme of ICPSs and IoT Based on VNE Algorithm](https://arxiv.org/pdf/2202.01375.pdf) | IoTJ | 2021 |  |
| PG-Seq2Seq      | `pg_seq2seq` | `learning`   | `joint_pr` | [A Continuous-Decision Virtual Network Embedding Scheme Relying on Reinforcement Learning](https://ieeexplore.ieee.org/document/8982091) | TNSM   | 2020 |  |
| GAE-Clustering                    | `gae_clustering`          | `learning`   | `bfs_trials` | [Accelerating Virtual Network Embedding with Graph Neural Networks](https://ieeexplore.ieee.org/document/9269128) | CNSM           | 2020 | Clustering |
| PG-MLP                | `pg_mlp`   | `learning`   | `joint_pr` | [NFVdeep: adaptive online service function chain deployment with deep reinforcement learning](http://ieeexplore.ieee.org/document/9068634/). | IWQOS          | 2019 |  |
| Hopfield-Network          | `hopfield_network` | `learning`   | `two-stage` | [NeuroViNE: A Neural Preprocessor for Your Virtual Network Embedding Algorithm](https://mediatum.ub.tum.de/doc/1449121/document.pdf) | INFOCOM   | 2018 | Subgraph Extraction |
| PG-CNN | `pg_cnn`          | `learning`   | `two-stage` | [A Novel Reinforcement Learning Algorithm for Virtual Network Embedding](https://bura.brunel.ac.uk/bitstream/2438/17673/1/FullText.pdf) | Neurocomputing | 2018 |  |
| MCTS                   | `mcts`              | `learning`   | `two-stage` | [Virtual Network Embedding via Monte Carlo Tree Search](https://www.researchgate.net/profile/Ljiljana-Trajkovic/publication/313873926_Virtual_Network_Embedding_via_Monte_Carlo_Tree_Search/links/5ac0386945851584fa7404f4/Virtual-Network-Embedding-via-Monte-Carlo-Tree-Search.pdf?_sg%5B0%5D=IbJ7vUDENmXiBbfMTzU7pe38Z0gve9tpmZe8Z0178rNWQVa5y6AFGJksV2UA1gPa2Fiohm7X1HzI-1rdAPT5Jg.Edi8Rb3R7d-SAgZ4Jl6Z-AnccOosuWHRn2EFIt8dcGLqnDdaw8vBfh1mKV-HieWT8lpuArIMwCjnyAg4CflgVw.cWgci1nNGkvx6bRqmirSaRRk-bi80Q0gMjvmyL49gbkiYRuKU6Zu1Aswe4xTxC99BNyBH7dYbFH3YyQTzUJczg&_sg%5B1%5D=XE66L-R7TPh36UxeMPExdBq5KyXxwAikDWvZbhvLjlAdwbBQ3MNiZbmBZzwQ0L1ntkXedGL1rZZYqX6LhuHdgQbg5Xi8I7phGNSAPGvh1OJv.Edi8Rb3R7d-SAgZ4Jl6Z-AnccOosuWHRn2EFIt8dcGLqnDdaw8vBfh1mKV-HieWT8lpuArIMwCjnyAg4CflgVw.cWgci1nNGkvx6bRqmirSaRRk-bi80Q0gMjvmyL49gbkiYRuKU6Zu1Aswe4xTxC99BNyBH7dYbFH3YyQTzUJczg&_iepl=) | TCYB           | 2018 | MultiThreading Support |

> `*` means that the algorithm only supports chain-shape virtual networks embedding


## Meta-heuristics Solvers

| Name                           | Command       | Type         | Mapping      | Title                                                        | Publication | Year | Note |
| ------------------------------ | ------------- | ------------ | ------------ | ------------------------------------------------------------ | ----------- | ---- | ------------------------------ |
| NodeRanking-MetaHeuristic          | `**_**`         | `meta-heuristics`   | `joint` | [Virtual network embedding through topology awareness and optimization](https://www.sciencedirect.com/science/article/abs/pii/S1389128612000461) | CN         | 2012 | MultiThreading Support |
| Genetic-Algorithm          | `ga`         | `meta-heuristics`   | `two-stage` | [Virtual network embedding based on modified genetic algorithm](https://link.springer.com/article/10.1007/s12083-017-0609-x#:~:text=Virtual%20network%20embedding%20is%20a,nodes%2C%20the%20goal%20of%20link) | Peer-to-Peer Networking and Applications         | 2019 | MultiThreading Support |
| Tabu-Search          | `ts`         | `meta-heuristics`   | `joint` | [Virtual network forwarding graph embedding based on Tabu Search](https://ieeexplore.ieee.org/document/8171072) | WCSP         | 2017 | MultiThreading Support |
| ParticleSwarmOptimization          | `pso`         | `meta-heuristics`   | `two-stage` | [Energy-Aware Virtual Network Embedding](https://ieeexplore.ieee.org/document/6709811) | TON         | 2014 | MultiThreading Support |
| Ant-Colony-Optimization  | `aco`          | `meta-heuristics` | `joint`     | [Link mapping-oriented ant colony system for virtual network embedding](https://ieeexplore.ieee.org/document/7969445) | CEC         | 2017 | MultiThreading Support |
| AntColony-Optimization  | `aco`          | `meta-heuristics` | `joint`     | [VNE-AC: Virtual Network Embedding Algorithm Based on Ant Colony Metaheuristic](https://www.gta.ufrj.br/ensino/cpe717-2011/VNE-ICC-1.pdf) | ICC         | 2011 | MultiThreading Support |
| Simulated-Annealing  | `sa`          | `meta-heuristics` | `two-stage`     | [FELL: A Flexible Virtual Network Embedding Algorithm with Guaranteed Load Balancing](https://ieeexplore.ieee.org/abstract/document/5962960) | ICC         | 2011 | MultiThreading Support |

**Other Related Papers**
- Particle Swarm Optimization 
  - Xiang Cheng et al. "Virtual network embedding through topology awareness and optimization". CN, 2012.
  - An Song et al. "A Constructive Particle Swarm Optimizer for Virtual Network Embedding". TNSE, 2020.
- Genetic Algorithm
  - Liu Boyang et al. "Virtual Network Embedding Based on Hybrid Adaptive Genetic Algorithm" In ICCC, 2019.
  - Khoa T.D. Nguyen et al. "An Intelligent Parallel Algorithm for Online Virtual Network Embedding". In CITS, 2019.
  - Khoa Nguyen et al. "Efficient Virtual Network Embedding with Node Ranking and Intelligent Link Mapping". In CloudNet, 2020.
  - Khoa Nguyen et al. "Joint Node-Link Algorithm for Embedding Virtual Networks with Conciliation Strategy". In GLOBECOM, 2021.
- Ant Colony Optimization
  - N/A

## Heuristics-based Solvers

| Name                           | Command       | Type         | Mapping      | Title                                                        | Publication | Year | Note |
| ------------------------------ | ------------- | ------------ | ------------ | ------------------------------------------------------------ | ----------- | ---- | ---- |
| PL (Priority of Location)      | `pl_rank`     | `heuristics` | `two-stage`  | [Efficient Virtual Network Embedding of Cloud-Based Data Center Networks into Optical Networks](https://ieeexplore.ieee.org/document/9415134) | TPDS        | 2021 |      |
| NRM (Node Resource Management) | `nrm_rank`    | `heuristics` | `two-stage`  | [Virtual Network Embedding Based on Computing, Network, and Storage Resource Constraints](https://ieeexplore.ieee.org/document/7976281) | IoTJ        | 2018 |      |
| GRC (Global resource capacity) | `grc_rank`    | `heuristics` | `two-stage`  | [Toward Profit-Seeking Virtual Network Embedding Algorithm via Global Resource Capacity](https://ieeexplore.ieee.org/document/6847918) | INFOCOM     | 2014 |      |
| RW-MaxMatch (NodeRank)         | `rw_rank`     | `heuristics` | `two-stage`  | [Virtual Network Embedding Through Topology-Aware Node Ranking](https://dl.acm.org/doi/10.1145/1971162.1971168) | ACM SIGCOMM Computer Communication Review     | 2011 |      |
| RW-BFS (NodeRank)              | `rw_rank_bfs` | `heuristics` | `bfs_trials` | [Virtual Network Embedding Through Topology-Aware Node Ranking](https://dl.acm.org/doi/10.1145/1971162.1971168) | ACM SIGCOMM Computer Communication Review     | 2011 |      |

## Exact Solvers

| Name                                 | Command     | Type      | Mapping   | Title                                                                                                                                               | Publication | Year | Note |
| ------------------------------------ | ----------- | --------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---- | ---- |
| MIP (Mixed-Integer Programming)  | `mip` | `exact` | `joint` | [ViNEYard: Virtual Network Embedding Algorithms With Coordinated Node and Link Mapping](https://ieeexplore.ieee.org/document/5951812?arnumber=5951812) | TON         | 2012 |      |
| D-Rounding (Deterministic Rounding) | `d_rounding`   | `exact` | `joint` | [ViNEYard: Virtual Network Embedding Algorithms With Coordinated Node and Link Mapping](https://ieeexplore.ieee.org/document/5951812?arnumber=5951812) | TON         | 2012 |      |
| R-Rounding (Random Rounding)        | `r_rounding`   | `exact` | `joint` | [ViNEYard: Virtual Network Embedding Algorithms With Coordinated Node and Link Mapping](https://ieeexplore.ieee.org/document/5951812?arnumber=5951812) | TON         | 2012 |      |
