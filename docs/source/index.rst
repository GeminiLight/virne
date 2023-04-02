.. virne documentation master file, created by
   sphinx-quickstart on Fri Feb 24 11:24:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Virne: A Unified Framework for VNE
==================================

.. note::

  **Virne** is still under development. If you have any questions, please contact me via email: wtfly2018@gmail.com.

**Virne** is a Python framework for Virtual Network Embedding (VNE) with the following characteristics:

#. **Lightweight**: Environments and algorithms are implemented concisely, using three necessary dependencies (networkx, numpy, pandas).
#. **Develop efficiently**: General operation methods are implemented uniformly and several environments for RL are supplied in gym.Env-style.
#. **Rich implementations**: Various algorithms are preset here and unified interfaces for calling are provided.

Supported features

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


Contents
----------


.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: introduction

    intro/background
    intro/formulation
    .. intro/introduction

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
========

..  code-block:: bib

    @misc{tfw-virne-2023,
      author = {Tianfu Wang},
      title = {Virne: A Python Framework for Virtual Network Embedding},
      year = {2023},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/GeminiLight/virne}},
      commit = {cf68db31dbf07db0976d1afeda09d5c6070936d3}
    }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`