Virne: An NFV-RA Benchmark
==========================

**Virne** is a simulator and benchmark for **resource allocation (RA) in
Network Functions Virtualisation (NFV)**, with unified support for traditional
and reinforcement learning (RL)-based algorithms.

.. note::

  In the literature, RA in NFV is often termed Virtual Network Embedding (VNE), Virtual Network Function (VNF) placement, service function chain (SFC) deployment, or network slicing in 5G.

Start Here
----------

* **New to Virne?** Follow the :doc:`installation <start/installation>` and
  run the :doc:`5-minute quickstart <start/running>`.
* **Studying or extending the benchmark?** Read the :doc:`problem formulation
  <intro/formulation>`, :doc:`architecture <intro/framework>`, and
  :doc:`solver registry <solver/overview>`.

Virne provides the following core capabilities:

.. grid:: 2 2 2 4
   :gutter: 3

   .. grid-item-card::
      :class-item: sd-font-weight-bold
      :class-header: sd-bg-info sd-text-white sd-font-weight-bold
      :class-card: sd-outline-info sd-rounded-1

      Highly Customizable Simulations
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      Simulate diverse network environments (e.g., cloud, edge, 5G) with user-defined topologies, resources, and service requirements.

   .. grid-item-card::
      :class-item: sd-font-weight-bold
      :class-header: sd-bg-success sd-text-white sd-font-weight-bold
      :class-card: sd-outline-success sd-rounded-1

      Extensive Algorithm Library
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
      Registers exact, heuristic, meta-heuristic, and learning-based solvers behind a common interface.

   .. grid-item-card::
      :class-item: sd-font-weight-bold
      :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
      :class-card: sd-outline-primary sd-rounded-1

      Reinforcement Learning Support
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      Provides standardized RL pipelines and Gym-style environments for rapid development and benchmarking of RL-based solutions.

   .. grid-item-card::
      :class-item: sd-font-weight-bold
      :class-header: sd-bg-warning sd-text-white sd-font-weight-bold
      :class-card: sd-outline-warning sd-rounded-1

      In-depth Evaluation Aspects
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
      Enables insightful analysis beyond effectiveness, covering practicality perspectives such as solvability, generalization, and scalability.

The overall architecture of Virne is illustrated below:

.. image:: _static/virne-architecture.png
  :width: 1000
  :alt: Overall Architecture of Virne

.. note::
  A Virne experiment has four steps: configure the simulation, launch the
  event-driven system, process service requests, and record results.

Virne also provides a unified environment and training pipeline for deep RL
algorithms.

.. image:: _static/virne-rl-support.png
  :width: 1000
  :alt: Unified Gym-style Environment and RL Pipeline in Virne 

Citations
---------

❤️ If you find Virne helpful to your research, please feel free to cite our related papers.


Benchmark Paper
~~~~~~~~~~~~~~~

**[ICLR, 2026] Virne Benchmark** (`paper <https://openreview.net/forum?id=jngvm9MGyv>`__ & `arXiv <https://arxiv.org/abs/2507.19234>`__ & `code <https://github.com/GeminiLight/virne>`__)

..  code-block:: bib

    @inproceedings{tfwang-2026-virne,
      title={Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV},
      author={Wang, Tianfu and Deng, Liwei and Chen, Xi and Wang, Junyang and He, Huiguo and Hu, Zhengyu and Wu, Wei and Ding, Leilei and Fan, Qilin and Xiong, Hui},
      booktitle={The Fourteenth International Conference on Learning Representations},
      year={2026},
      url={https://openreview.net/forum?id=jngvm9MGyv},
    }

Algorithmic Papers
~~~~~~~~~~~~~~~~~~

**[IJCAI-2024] FlagVNE** (`paper <https://arxiv.org/abs/2404.12633>`__ & `code <https://github.com/GeminiLight/flag-vne>`__)

..  code-block:: bib

    @INPROCEEDINGS{tfwang-ijcai-2024-flagvne,
      title={FlagVNE: A Flexible and Generalizable Reinforcement Learning Framework for Network Resource Allocation},
      author={Wang, Tianfu and Fan, Qilin and Wang, Chao and Ding, Leilei and Yuan, Nicholas Jing and Xiong, Hui},
      booktitle={Proceedings of the 33rd International Joint Conference on Artificial Intelligence},
      year={2024},
    }

**[TSC-2023] HRL-ACRA** (`paper <https://ieeexplore.ieee.org/document/10291038>`__ & `code <https://github.com/GeminiLight/hrl-acra>`__)

..  code-block:: bib

    @ARTICLE{tfwang-tsc-2023-hrl-acra,
      author={Wang, Tianfu and Shen, Li and Fan, Qilin and Xu, Tong and Liu, Tongliang and Xiong, Hui},
      journal={IEEE Transactions on Services Computing},
      title={Joint Admission Control and Resource Allocation of Virtual Network Embedding Via Hierarchical Deep Reinforcement Learning},
      volume={17},
      number={03},
      pages={1001--1015},
      year={2024},
      doi={10.1109/TSC.2023.3326539}
    }

**[ICC-2021] DRL-SFCP** (`paper <https://ieeexplore.ieee.org/document/9500964>`__ & `code <https://github.com/GeminiLight/drl-sfcp>`__)

..  code-block:: bib

    @INPROCEEDINGS{tfwang-icc-2021-drl-sfcp,
      author={Wang, Tianfu and Fan, Qilin and Li, Xiuhua and Zhang, Xu and Xiong, Qingyu and Fu, Shu and Gao, Min},
      booktitle={ICC 2021 - IEEE International Conference on Communications}, 
      title={DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning}, 
      year={2021},
      volume={},
      number={},
      pages={1-6},
      doi={10.1109/ICC42927.2021.9500964}
    }

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: Introduction

    intro/background
    intro/formulation
    intro/framework
    intro/rl-support

.. toctree::
    :hidden:
    :caption: Quick Start

    start/installation
    start/running
    start/simulation

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: Algorithms

    solver/overview
    solver/exact
    solver/heuristic
    solver/meta_heuristic
    solver/learning

.. toctree::
    :hidden:
    :caption: Evaluation

    evaluation/metrics
    evaluation/aspects

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: API Reference

    api
