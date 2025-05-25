Aspects
=======

Beyond standard quantitative performance metrics, Virne promotes a comprehensive assessment of Network Function Virtualization Resource Allocation (NFV-RA) algorithms by examining them from multiple **practicality perspectives**. These aspects help to understand the real-world viability, robustness, and adaptability of different approaches, guiding further development and deployment.

Key Practicality Perspectives
-----------------------------

Virne emphasizes three primary aspects to evaluate the practicality of NFV-RA algorithms:

.. grid:: 1 1 1 3
   :gutter: 3

   .. grid-item-card::
      :class-header: sd-bg-info sd-text-white sd-font-weight-bold
      :class-card: sd-outline-info sd-rounded-1

      Solvability
      ^^^^^^^^^^^

      **Solvability** refers to an algorithm's fundamental ability to find feasible solutions for NFV-RA instances. This perspective often involves:

      * Evaluating performance on static, offline instances to isolate algorithmic capability from dynamic network effects.
      * Assessing how solution quality or success rate varies with problem complexity (e.g., VN size or constraint tightness).

      Understanding solvability helps to distinguish between failures due to an algorithm's inherent limitations versus those caused by transient, unsolvable network states in online scenarios.

   .. grid-item-card::
      :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
      :class-card: sd-outline-primary sd-rounded-1

      Generalization
      ^^^^^^^^^^^^^^

      **Generalization** indicates an algorithm's ability to maintain reliable and effective performance across a variety of network conditions and problem instances different from those it might have been optimized or trained for. This involves evaluating:

      * Performance under fluctuating traffic rates and loads.
      * Adaptability to changes in the distribution of VN request characteristics (e.g., resource demands, VN sizes).
      * Robustness when deployed in different Physical Network (PN) topologies or scales.

      Strong generalization is crucial for algorithms intended for real-world deployment where network conditions are dynamic and often unpredictable.

   .. grid-item-card::
      :class-header: sd-bg-secondary sd-text-white sd-font-weight-bold
      :class-card: sd-outline-secondary sd-rounded-1

      Scalability
      ^^^^^^^^^^^

      **Scalability** measures how effectively an NFV-RA algorithm performs as the size and complexity of the problem increase. This is typically assessed by:

      * Evaluating solution quality and resource efficiency on large-scale network topologies (both PN and VN).
      * Analyzing the growth trend of the Average Solving Time (AST) as network sizes (number of physical or virtual nodes) increase.

      Good scalability ensures that an algorithm remains computationally feasible and effective when applied to extensive, real-world network infrastructures.

Evaluation Methodologies
------------------------

.. card::
   :class-header: sd-bg-success sd-text-white sd-font-weight-bold
   :class-card: sd-outline-success sd-rounded-1

   Holistic Assessment Framework
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   By providing interfaces and methodologies to evaluate these practical aspects, Virne enables a more holistic understanding of each algorithm's strengths and weaknesses, offering data-driven guidance for future research directions and practical deployments in NFV environments.

   The framework supports systematic evaluation across different:

   * **Problem scales**: From small test instances to large-scale enterprise networks
   * **Network conditions**: Varying traffic patterns, topology changes, and resource availability
   * **Temporal scenarios**: Both static offline analysis and dynamic online deployment

.. note::
   These practicality perspectives complement traditional performance metrics by providing insights into real-world deployment considerations that are often overlooked in purely algorithmic evaluations.