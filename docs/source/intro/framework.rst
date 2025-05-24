Framework
=========

Virne is designed as a comprehensive and unified benchmarking framework for Network Function Virtualization Resource Allocation (NFV-RA), with a strong emphasis on supporting deep Reinforcement Learning (RL)-based methods. Our goal is to provide an accessible platform for reproducible research and standardized evaluation across diverse network scenarios.

.. card::
    :class-card: sd-outline-info  sd-rounded-1
    :class-body: sd-font-weight-bold

    #. Simulation Configuration
    #. Network System
    #. Algorithm Implementation
    #. Auxiliary Utilities
    #. Evaluation Criteria

.. image:: ../_static/virne-architecture.png
  :width: 1000
  :alt: Overall Architecture of Virne

Design Principles
-----------------

The development of Virne is guided by three core design principles, following established software engineering practices:

.. grid:: 12 4 4 4

  .. grid-item-card::
    :class-item: sd-font-weight-bold
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info  sd-rounded-1
    :columns: 12 6 6 4

    Versatile Customization
    ^^^^^^^^^^^^^^^^^^^^^^^
    Virne allows for extensive customization to meet diverse simulation needs across various network scenarios and conditions, ensuring high adaptability.

  .. grid-item-card::
    :class-item: sd-font-weight-bold
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success  sd-rounded-1
    :columns: 12 6 6 4

    Scalable Modularity
    ^^^^^^^^^^^^^^^^^^^
    The platform is built with a modular architecture. This design supports flexible configurations and makes Virne easily extensible for new algorithms or network environments.

  .. grid-item-card::
    :class-item: sd-font-weight-bold
    :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
    :class-card: sd-outline-primary  sd-rounded-1
    :columns: 12 6 6 4

    Intuitive Usability
    ^^^^^^^^^^^^^^^^^^^
    We prioritize a user-friendly interface and workflow. This enables researchers to focus on experimental outcomes and insights rather than getting bogged down by implementation complexities.

Architecture Overview
---------------------

The primary modules of Virne are:

Simulation Configuration
^^^^^^^^^^^^^^^^^^^^^^^^
This module allows users to define and customize the network environment. Virne can accurately model a wide array of NFV scenarios, from cloud data centers to edge and 5G networks. Key customizable elements include:

* **Network Topologies**: Users can select from various synthetic topology generation methods or use real-world physical infrastructure topologies (e.g., from SNDLib).
* **Resource Availability**: Define multiple resource types (e.g., CPU, GPU, bandwidth) and their distribution across network nodes, links, and the overall graph.
* **Service Requirements**: Specify additional service needs like latency constraints, energy efficiency targets, or reliability metrics.

Network System
^^^^^^^^^^^^^^
Based on the configurations, Virne instantiates an event-driven simulator. This module consists of:

* **Physical Network (PN)**: The underlying infrastructure.
* **Virtual Network (VN) Requests**: A series of sequentially arriving service requests.

Each VN request arrival is treated as a discrete event, creating an instance that the NFV-RA algorithm must solve. The system then evaluates the solution's feasibility and updates network resources.

Algorithm Implementation
^^^^^^^^^^^^^^^^^^^^^^^^
Virne features a modular architecture that simplifies the implementation and integration of diverse NFV-RA algorithms, including exact solvers, heuristics, meta-heuristics, and advanced learning-based methods. For RL-based approaches, Virne provides a unified pipeline (as detailed in Figure 3 of our paper) that standardizes:

* **NFV-RA as a Markov Decision Process (MDP)**: Modeling the solution construction sequentially.
* **Policy Architectures**: Support for various neural network architectures (e.g., MLP, CNN, GCN, GAT).
* **RL Training Methods**: Integration of algorithms like PPO, A3C, etc.
* **Gym-style Environments**: Facilitating the development and testing of RL agents.

Auxiliary Utilities
^^^^^^^^^^^^^^^^^^^
To enhance usability and streamline analysis, Virne includes several key utilities:

* **System Controller**: Manages the simulation of physical and virtual networks.
* **Solution Monitor**: Tracks solution feasibility and performance metrics during execution.
* **Visualization Tools**: Provide interactive and visual representations of simulation results for intuitive analysis.

Evaluation Criteria
^^^^^^^^^^^^^^^^^^^
Virne offers a comprehensive suite of metrics and practical perspectives for systematic evaluation:

* **Standard Performance Metrics**: Including Request Acceptance Rate (RAC), Long-term Revenue-to-Cost (LRC), Long-term Average Revenue (LAR), and Average Solving Time (AST).
* **Practicality Perspectives**:
    * **Solvability**: The algorithm's ability to find feasible solutions.
    * **Generalization**: Performance reliability across varied network conditions and traffic patterns.
    * **Scalability**: Effectiveness in handling increases in network size and problem complexity.

Workflow
--------

Virne enables a streamlined workflow for comprehensive experimentation:
1.  **Customize Simulation**: Define network scenarios and conditions via configuration files.
2.  **Instantiate System**: The network system is created, triggering service request events.
3.  **Algorithm Interaction**: At each event, the selected NFV-RA algorithm processes the instance.
4.  **Record Results**: Processing details and final results are automatically recorded for analysis.

This framework is designed to serve as a unified and readily accessible tool for researchers from both the machine learning and networking communities, aiming to accelerate data-centric ML research in network optimization.
