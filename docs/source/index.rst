.. virne documentation master file, created by
   sphinx-quickstart on Fri Feb 24 11:24:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Virne: A Simulator for NFV
===================================

.. note::

  **Virne** is still under development. 
  If you have any questions, 
  please open an new issue on `Github <https://github.com/GeminiLight/virne>`_  or 
  contact me via email: wtfly2018@gmail.com.

**Virne** is a simulator for **resource allocation problems in network virtualization**, mainly for **virtual network embedding (VNE)**. 
It also is adaptable to VNE's variants, such as **service function chain deployment (SFC Deployment)**, **network slicing**, etc. 
Specifically, it provides a unified interface for various VNE algorithms, and provides a variety of network topologies, network attributes, and RL environments.

Its main characteristics are as follows.

.. grid:: 12 4 4 4

    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-info sd-text-white sd-font-weight-bold
        :class-card: sd-outline-info  sd-rounded-1
        :columns: 12 6 6 4

        Rich Implementations
        ^^^^^^^^^^^^^^^^^^^^
        Provide 20+ solvers, including exact, heuristic, meta-heuristic, and learning-based algorithms.

    .. grid-item-card::
        :class-item: sd-font-weight-bold
        :class-header: sd-bg-success sd-text-white sd-font-weight-bold
        :class-card: sd-outline-success  sd-rounded-1
        :columns: 12 6 6 4

        Extensible Development
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


Supported Features
------------------

- **Adaptation to VNE Variants**
 
  - Service Function Chain Deployment (SFC Deployment)
  - Network Slicing

- **Diverse Network Topologies**
  
  - Star Graph: Data Center Network
  - 2D-grid Graph: Grid Network
  - Waxman Graph: General Network
  - Path Graph: Chain-style Network
  - Edge Probabilistic Connection Graph
  - Customlized Topology 
  
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

.. image:: _static/workflow.jpg
  :width: 1000
  :alt: Overall Workflow of Virne


.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: introduction

    intro/background
    intro/formulation

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


.. you can use the ``lumache.get_random_ingredients()`` function:


Citation
--------

If you find Virne helpful to your research, please feel free to cite our related papers.

.. Project Citation
.. ~~~~~~~~~~~~~~~~

.. ..  code-block:: bib

..     @misc{tfw-virne-2023,
..       author = {Tianfu Wang},
..       title = {Virne: A Python Framework for Virtual Network Embedding},
..       year = {2023},
..       publisher = {GitHub},
..       journal = {GitHub repository},
..       howpublished = {\url{https://github.com/GeminiLight/virne}},
..       commit = {cf68db31dbf07db0976d1afeda09d5c6070936d3}
..     }



Our Related Papers
~~~~~~~~~~~~~~~~~~

**[IJCAI-2024] FlagVNE**

..  code-block:: bib

    @INPROCEEDINGS{ijcai-2024-flagvne,
      title={FlagVNE: A Flexible and Generalizable RL Framework for Network Resource Allocation},
      author={Wang, Tianfu and Fan, Qilin and Wang, Chao and Ding, Leilei and Yuan, Nicholas Jing and Xiong, Hui},
      booktitle={Proceedings of the 33rd International Joint Conference on Artificial Intelligence},
      year={2024},
    }

**[TSC-2023] HRL-ACRA**

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

**[ICC-2021] DRL-SFCP**

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