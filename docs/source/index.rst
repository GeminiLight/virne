.. virne documentation master file, created by
   sphinx-quickstart on Fri Feb 24 11:24:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Virne: A Simulator for NV-RA
===================================

.. note::

  **Virne** is still under development. 
  If you have any questions, 
  please open an new issue on `Github <https://github.com/GeminiLight/virne>`_  or 
  contact me via email: wtfly2018@gmail.com.


**Virne** is a simulator designed to address **resource allocation problems in network virtualization**. This category of problems is often referred to by various names, including:

- **Virtual Network Embedding (VNE)**
- **Virtual Network Function Placement (VNF Placement)**
- **Service Function Chain Deployment (SFC Deployment)**
- **Network Slicing**

The main goal of Virne is to provide a unified and flexible framework for solving these problems. Its main characteristics are as follows.

.. grid:: 12 4 4 4

    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-info sd-text-white sd-font-weight-bold
        :class-card: sd-outline-info  sd-rounded-1
        :columns: 12 6 6 4

        Rich Implementations
        ^^^^^^^^^^^^^^^^^^^^
        Provide 20+ solvers, including exact, heuristic, meta-heuristic, and machine learning-based algorithms.

    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success  sd-rounded-1
        :columns: 12 6 6 4

        Flexible Extensions 
        ^^^^^^^^^^^^^^^^^^^^^^
        Provide a variety of network topologies, network attributes, and RL environments, which can be easily extended.
        
    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
        :class-card: sd-outline-primary  sd-rounded-1
        :columns: 12 6 6 4

        Light-Weight
        ^^^^^^^^^^^^
        Implement concisely with less necessary dependencies, and can be extended easily for specific algorithms.

.. image:: _static/workflow.jpg
  :width: 1000
  :alt: Overall Workflow of Virne

Citations
---------

If you find Virne helpful to your research, please feel free to cite our related papers.

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


Supported Features
------------------

- **Diverse Network Topologies for Simulation**
  
  - Simple Network Structures: e.g. Star for Centralized Network, Path for Chain-style Network, etc.
  - Random Network Topologies: e.g. Waxman Graph, Edge Probabilistic Connection Graph, etc.
  - Real-world Network Topologies: e.g. Abilene, Geant, etc.
  
- **Multiple level Attributes for QoS**: 

  - Graph Level: e.g. the global requirements of user service requests, etc.
  - Node level: e.g. computing resource, server position, energy consumption, etc.
  - Link level: e.g. bandwidth resource, communication delay, etc.
  
- **Unified Reinforcement Learning Interface for Extension**
  
  - Provide serval RL Environments in gym.Env-style.
  - Implement the many RL training algorithms, including MCTS, PPO, A2C, etc.
  - Support the integration of RL algorithms from other libraries.
  
- **Various Simulation Scenarios**
  
  - Admission control: Early Reject some not cost-effective service requests.
  - Cloud-Edge: Heteregenous infrastructure with different QoS provision.
  - Time window: Globally process the a batch service requests in a time window.

- **Predefined QoS Awarenesses** (Additional Constraints/ Objectives)

  - [x] Position (Node level)
  - [x] Latency (Graph, Node and Link level)
  - [x] Security (Graph, Node and Link level)
  - [ ] Congestion (Graph, Node and Link level)
  - [ ] Energy (Graph, Node and Link level)
  - [x] Reliability (Graph, Node and Link level)
  - [ ] Dynamic (Graph, Node and Link level)
  - [ ] Parallelization
  - [ ] Privacy

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

.. toctree::
    :hidden:
    :caption: get start

    start/installation
    start/usage
    start/solver_list

.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: customlization

    customlization/simulation
    customlization/sfc_deployment

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
