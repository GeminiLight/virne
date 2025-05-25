.. virne documentation master file, created by
   sphinx-quickstart on Fri Feb 24 11:24:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Virne: An NFV-RA Benchmark
==========================

**Virne** is a comprehensive simulator and benchmark designed to address **resource allocation (RA) problems in network function virtualization (NFV)**, with a highlight on supporting **reinforcement learning (RL)**-based algorithms.

.. note::

  In the literature, RA in NFV is often termed Virtual Network Embedding (VNE), Virtual Network Function (VNF) placement, service function chain (SFC) deployment, or network slicing in 5G.

Virne offers a unified and comprehensive framework for NFV-RA, with the following key features:

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
      Implements 30+ NFV-RA algorithms (exact, heuristics, meta-heuristics, RL-based) in a modular, extensible architecture.

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
  Virne offers a streamlined workflow for supporting comprehensive experimentation of NFV-RA algorithms. (a) customize simulation configurations (b) launch event-driven network system (c) process service requests (d) record results for analysis.

Particularly, Virne highlights the support for deep reinforcement learning (RL) algorithms, providing a unified Gym-style environment and RL pipeline.

.. image:: _static/virne-rl-support.png
  :width: 1000
  :alt: Unified Gym-style Environment and RL Pipeline in Virne 

.. note::

  The RL pipeline in Virne is designed to be flexible and extensible, allowing researchers to easily integrate their own RL algorithms and environments.


Citations
---------

❤️ If you find Virne helpful to your research, please feel free to cite our related papers.


Benchmark Paper
~~~~~~~~~~~~~~~

**Virne Benchmark** (paper`  & `code <https://github.com/GeminiLight/virne>`__)

..  code-block:: bib

    @article{tfwang-2025-virne,
      title={Virne: A Comprehensive Benchmark for Deep RL-based Network Resource Allocation in NFV},
      author={Wang, Tianfu and Deng, Liwei and Chen, Xi and Wang, Junyang and He, Huiguo and Ding, Leilei and Wu, Wei and Fan, Qilin and Xiong, Hui},
      year={2025},
    }

Algorithmic Papers
~~~~~~~~~~~~~~~~~~

**[IJCAI-2024] FlagVNE** (`paper <https://arxiv.org/abs/2404.12633>`__ & `code <https://github.com/GeminiLight/flag-vne>`__)

..  code-block:: bib

    @INPROCEEDINGS{ijcai-2024-flagvne,
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

    @INPROCEEDINGS{tfw-icc-2021-drl-sfcp,
      author={Wang, Tianfu and Fan, Qilin and Li, Xiuhua and Zhang, Xu and Xiong, Qingyu and Fu, Shu and Gao, Min},
      booktitle={ICC 2021 - IEEE International Conference on Communications}, 
      title={DRL-SFCP: Adaptive Service Function Chains Placement with Deep Reinforcement Learning}, 
      year={2021},
      volume={},
      number={},
      pages={1-6},
      doi={10.1109/ICC42927.2021.9500964}
    }


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: introduction

    intro/background
    intro/problem
    intro/framework
    intro/rl-support

.. toctree::
    :hidden:
    :caption: get start

    start/installation
    start/running
    start/simulation

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: Solver List

    solver/exact
    solver/heuristic
    solver/meta_heuristic
    solver/learning

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: API Reference

    api
