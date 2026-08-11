<div align="center">
  <img src="resources/figures/virne-logo-text.png" width="200" alt="Virne logo">
  <h2>A Comprehensive Simulator & Benchmark for NFV-RA</h2>
  <p>
    <a href="https://deepwiki.com/GeminiLight/virne"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
    <a href="https://github.com/GeminiLight/virne/releases"><img src="https://img.shields.io/badge/version-1.0.0-blue" alt="Virne version 1.0.0"></a>
    <a href="https://pypi.org/project/virne/"><img src="https://img.shields.io/pypi/v/virne?label=pypi" alt="Virne on PyPI"></a>
    <a href="https://github.com/GeminiLight/virne/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="Apache 2.0 license"></a>
  </p>
  <p>
    <a href="https://arxiv.org/abs/2507.19234">✨ Benchmark Paper</a> &nbsp;&nbsp;•&nbsp;&nbsp;
    <a href="https://virne.readthedocs.io">Documentation</a> &nbsp;&nbsp;•&nbsp;&nbsp;
    <a href="https://github.com/GeminiLight/virne?tab=readme-ov-file#citations">Citations</a> &nbsp;&nbsp;•&nbsp;&nbsp;
    <a href="https://github.com/GeminiLight/sdn-nfv-papers">SDN-NFV Papers</a>
  </p>
</div>

--------------------------------------------------------------------------------

**Virne** is a simulator and benchmark for **resource allocation (RA) in Network Functions Virtualisation (NFV)**, with unified support for traditional and **reinforcement learning (RL)**-based algorithms.

> In the literature, RA in NFV is often termed Virtual Network Embedding (VNE), Virtual Network Function (VNF) placement, service function chain (SFC) deployment, or network slicing in 5G.

Virne offers a unified and comprehensive framework for NFV-RA, with the following key features:

* 1️⃣ **Highly Customizable Simulations**: Simulates diverse network environments (e.g., cloud, edge, 5G), with user-defined topologies, resources, and service requirements.
* 2️⃣ **Extensive Algorithm Suite**: Registers exact, heuristic, meta-heuristic, and learning-based solvers behind a common interface.
* 3️⃣ **Reinforcement Learning Support**: Provides standardized RL pipelines and Gymnasium-compatible environments for rapid development and benchmarking of RL-based solutions.
* 4️⃣ **In-depth Evaluation Aspects**: Enables insightful analysis beyond effectiveness, covering multiple practicality perspectives (e.g., solvability, generalization, and scalability).

> [!IMPORTANT]
> 🎉 The [Virne benchmark paper](https://arxiv.org/abs/2507.19234) has been accepted at ICLR 2026. Welcome to check it out!
>
> ✨ If you have any questions, please open a new issue or contact me via email (wtfly2018@gmail.com)

![Virne architecture: simulation, solver, environment, and evaluation components](resources/figures/virne-architecture.png)

### Citations

> ❤️ If you find Virne helpful to your research, please feel free to cite our related papers.

#### Benchmark Paper

**[ICLR, 2026] Virne** ([paper](https://arxiv.org/abs/2507.19234))

```bibtex
@inproceedings{tfwang-2026-virne,
  title={Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV},
  author={Wang, Tianfu and Deng, Liwei and Chen, Xi and Wang, Junyang and He, Huiguo and Hu, Zhengyu and Wu, Wei and Ding, Leilei and Fan, Qilin and Xiong, Hui},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
}
```

#### Algorithmic Papers

**[IJCAI, 2024] FlagVNE** ([paper](https://arxiv.org/pdf/2404.12633) & [code](https://github.com/GeminiLight/flag-vne))

```bibtex
@INPROCEEDINGS{ijcai-2024-flagvne,
  title={FlagVNE: A Flexible and Generalizable Reinforcement Learning Framework for Network Resource Allocation},
  author={Wang, Tianfu and Fan, Qilin and Wang, Chao and Ding, Leilei and Yuan, Nicholas Jing and Xiong, Hui},
  booktitle={Proceedings of the 33rd International Joint Conference on Artificial Intelligence},
  year={2024},
}
```

**[TSC, 2023] HRL-ACRA** ([paper](https://arxiv.org/pdf/2406.17334) & [code](https://github.com/GeminiLight/hrl-acra))

```bibtex
@ARTICLE{tsc-2023-hrl-acra,
  author={Wang, Tianfu and Shen, Li and Fan, Qilin and Xu, Tong and Liu, Tongliang and Xiong, Hui},
  journal={IEEE Transactions on Services Computing},
  title={Joint Admission Control and Resource Allocation of Virtual Network Embedding Via Hierarchical Deep Reinforcement Learning},
  volume={17},
  number={03},
  pages={1001--1015},
  year={2024},
}
```

**[ICC, 2021] DRL-SFCP** ([paper](https://ieeexplore.ieee.org/document/9500964) & [code](https://github.com/GeminiLight/drl-sfcp))

```bibtex
@INPROCEEDINGS{icc-2021-drl-sfcp,
  author={Wang, Tianfu and Fan, Qilin and Li, Xiuhua and Zhang, Xu and Xiong, Qingyu and Fu, Shu and Gao, Min},
  booktitle={ICC 2021 - IEEE International Conference on Communications}, 
  title={DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning}, 
  year={2021},
  pages={1-6},
}
```

## Quickstart

### Installation

The installation script supports CPU environments on Linux and macOS, plus
CUDA 12.6, 12.8, and 13.0 on Linux. Clone the repository, then create and
activate a Python 3.10 or 3.11 environment:

```bash
git clone https://github.com/GeminiLight/virne.git
cd virne
python3 -m venv .venv
source .venv/bin/activate
```

Install either the CPU build or a supported CUDA build. The script defaults to
CPU and installs PyTorch 2.11.0 with PyG 2.8.0.post1. It also installs the
matching optional PyG acceleration wheels without hard-coding the operating
system or Python ABI:

```bash
# CPU-only PyTorch and PyG
bash install.sh -c cpu

# CUDA 12.6 (use 12.8 or 13.0 when appropriate for your driver and GPU)
bash install.sh -c 12.6
```

The script installs Virne in editable mode and prints all three versions. You
can verify them again with:

```bash
python -c "import torch, torch_geometric, virne; print(virne.__version__, torch.__version__, torch_geometric.__version__)"
```

### Run a Small Experiment

Use a fast heuristic and ten VN requests for the first run. Calling
`python main.py` without overrides starts the larger default RL experiment.

```bash
python main.py \
  solver.solver_name=nrm_rank \
  v_sim_setting.num_v_nets=10 \
  training.use_cuda=false \
  'logger.backends=[console]'
```

The run finishes with `Complete` and writes its resolved configuration,
summary, and per-event records under:

```text
results/virne/nrm_rank/<run-id>/
```

See the [Quickstart](https://virne.readthedocs.io/en/latest/start/running.html)
for Hydra overrides and output details, and the
[solver registry](https://virne.readthedocs.io/en/latest/solver/overview.html)
for every valid solver command.

## Choose a Solver

Start with a solver that matches the purpose of your experiment:

| Command | Use it for | Important note |
| --- | --- | --- |
| `nrm_rank` | A fast, deterministic heuristic baseline | Recommended for the first end-to-end run |
| `random_rank` | A simple randomized baseline or sanity check | Set `experiment.seed` when comparing runs |
| `mip` | An exact-method baseline on small instances | Requires OR-Tools and can be substantially slower |
| `ppo_dual_gat+` | An example of the RL training and inference pipeline | Requires the learning dependencies and training |

This table is intentionally limited to useful starting points. See the
[generated solver registry](https://virne.readthedocs.io/en/latest/solver/overview.html)
for every command registered by the current code, or follow the
[RL Pipeline](https://virne.readthedocs.io/en/latest/intro/rl-support.html) for
the learning-based workflow.
