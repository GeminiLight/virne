Formulation
===========

Network Function Virtualization Resource Allocation (NFV-RA) is a critical challenge in modern networking. It involves efficiently mapping service requests, modeled as Virtual Networks (VNs) composed of interconnected Virtual Network Functions (VNFs), onto a shared physical network (PN) infrastructure. This process must satisfy various resource demands and constraints. NFV-RA is recognized as an NP-hard combinatorial optimization problem, necessitating sophisticated solution strategies.

This page details the formal problem definition used within Virne, starting with a basic cost optimization model and then discussing common extensions.

System Model
------------

Both the physical infrastructure and the virtual service requests are modeled as attributed graphs.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :class-header: sd-bg-info sd-text-white sd-font-weight-bold
      :class-card: sd-outline-info sd-rounded-1

      Physical Network (PN)
      ^^^^^^^^^^^^^^^^^^^^^

      The Physical Network, :math:`\mathcal{G}_p`, represents the underlying infrastructure and is modeled as an undirected graph:

      .. math::
         \mathcal{G}_p = (\mathcal{N}_p, \mathcal{L}_p)

      where:

      * :math:`\mathcal{N}_p` is the set of physical nodes (e.g., servers).
      * :math:`\mathcal{L}_p` is the set of physical links interconnecting these nodes.

      Each physical node :math:`n_p \in \mathcal{N}_p` is associated with available computing resources, denoted as :math:`C(n_p)` (e.g., CPU, RAM). Each physical link :math:`l_p \in \mathcal{L}_p` has an available bandwidth capacity, :math:`B(l_p)`.

   .. grid-item-card::
      :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
      :class-card: sd-outline-primary sd-rounded-1

      Virtual Network (VN)
      ^^^^^^^^^^^^^^^^^^^^

      A service request, or Virtual Network :math:`\mathcal{G}_v`, is also modeled as an undirected graph:

      .. math::
         \mathcal{G}_v = (\mathcal{N}_v, \mathcal{L}_v, \omega, \varpi)

      where:

      * :math:`\mathcal{N}_v` is the set of virtual nodes, representing VNFs or service components.
      * :math:`\mathcal{L}_v` is the set of virtual links, representing the required connectivity and traffic flow between virtual nodes.
      * :math:`\omega` is the arrival time of the VN request.
      * :math:`\varpi` is the requested lifetime (duration) of the VN.

      Each virtual node :math:`n_v \in \mathcal{N}_v` has specific computing resource demands, :math:`C(n_v)`. Each virtual link :math:`l_v \in \mathcal{L}_v` has a required bandwidth demand, :math:`B(l_v)`.

An NFV-RA problem instance :math:`I` is defined by the pair :math:`I = (\mathcal{G}_v, \mathcal{G}_p)`, representing a VN request arriving at a snapshot of the PN. Virne aims to find an efficient mapping for these instances.

.. note::
   Figure 1 in our research paper provides a visual illustration of this NFV-RA problem model.

Basic Formulation for Cost Optimization
---------------------------------------

The core NFV-RA problem involves deciding how to embed incoming VNs onto the PN while optimizing certain objectives, typically related to resource efficiency.

.. card::
   :class-header: sd-bg-success sd-text-white sd-font-weight-bold
   :class-card: sd-outline-success sd-rounded-1

   Embedding Process
   ^^^^^^^^^^^^^^^^^

   Mapping a VN :math:`\mathcal{G}_v` onto a sub-portion of the PN, :math:`\mathcal{G}_{p'}`, is represented by a mapping function :math:`f_{\mathcal{G}}: \mathcal{G}_v \rightarrow \mathcal{G}_{p'}`. This consists of two sub-processes:

   1.  **Node Mapping (:math:`f_N`)**: Assigning each virtual node :math:`n_v \in \mathcal{N}_v` to a suitable physical node :math:`n_p = f_N(n_v) \in \mathcal{N}_p`.
   2.  **Link Mapping (:math:`f_L`)**: Routing each virtual link :math:`l_v \in \mathcal{L}_v` over a physical path :math:`\rho_p = f_L(l_v)` in :math:`\mathcal{L}_p` that connects the physical nodes hosting the endpoints of :math:`l_v`.

.. card::
   :class-header: sd-bg-warning sd-text-white sd-font-weight-bold
   :class-card: sd-outline-warning sd-rounded-1

   Boolean Variables for Mapping Decisions
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   To formalize the constraints and objectives, we use boolean variables:

   * :math:`x_{i}^{m} = 1` if virtual node :math:`n_{v}^{m}` is placed on physical node :math:`n_{p}^{i}`, and :math:`0` otherwise.
   * :math:`y_{i,j}^{m,w} = 1` if virtual link :math:`l_{v}^{m,w}` (connecting :math:`n_{v}^{m}` and :math:`n_{v}^{w}`) traverses physical link :math:`l_{p}^{i,j}` (connecting :math:`n_{p}^{i}` and :math:`n_{p}^{j}`), and :math:`0` otherwise.

   Here, :math:`m, w` are identifiers for virtual nodes, and :math:`i, j, k` are identifiers for physical nodes.

Embedding Constraints
~~~~~~~~~~~~~~~~~~~~~

A VN request is successfully embedded if a feasible mapping solution is found that satisfies the following constraints:

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card::
      :class-header: sd-bg-info sd-text-white sd-font-weight-bold
      :class-card: sd-outline-info sd-rounded-1

      Node Constraints
      ^^^^^^^^^^^^^^^^

      **1. VN Node Assignment**: Each virtual node must be mapped to exactly one physical node.
      
      .. math::
         \sum_{n_{p}^{i} \in \mathcal{N}_p} x_{i}^{m} = 1, \quad \forall n_{v}^{m} \in \mathcal{N}_v \quad (3)

      **2. PN Node Capacity**: Each physical node can host at most one virtual node *from the same incoming VN request*.
      
      .. math::
         \sum_{n_{v}^{m} \in \mathcal{N}_v} x_{i}^{m} \le 1, \quad \forall n_{p}^{i} \in \mathcal{N}_p \quad (4)

      **3. Node Resource Availability**: The computing resources available at a physical node must meet or exceed the demands of the virtual node mapped to it.
      
      .. math::
         x_{i}^{m} C(n_{v}^{m}) \le C(n_{p}^{i}), \quad \forall n_{v}^{m} \in \mathcal{N}_v, n_{p}^{i} \in \mathcal{N}_p \quad (5)

   .. grid-item-card::
      :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
      :class-card: sd-outline-primary sd-rounded-1

      Link Constraints
      ^^^^^^^^^^^^^^^^

      **4. Flow Conservation**: For each virtual link, a valid path must be established in the PN between the physical nodes hosting its endpoints. Let :math:`\Omega(n_p^k)` be the set of neighbors of physical node :math:`n_p^k`.
      
      .. math::
         \sum_{n_{p}^{j} \in \Omega(n_{p}^{k})} y_{k,j}^{m,w} - \sum_{n_{p}^{i} \in \Omega(n_{p}^{k})} y_{i,k}^{m,w} = x_{k}^{m} - x_{k}^{w}, \quad \forall l_{v}^{m,w} \in \mathcal{L}_v, n_{p}^{k} \in \mathcal{N}_p \quad (6)

      **5. Loop Prevention**: Virtual links should be routed acyclically.
      
      .. math::
         y_{i,j}^{m,w} + y_{j,w}^{m,w} \le 1, \quad \forall l_{m,w}^{v} \in \mathcal{L}_{v}, l_{i,j}^{p} \in \mathcal{L}_{p} \quad (7)

   .. grid-item-card::
      :class-header: sd-bg-secondary sd-text-white sd-font-weight-bold
      :class-card: sd-outline-secondary sd-rounded-1

      Resource Constraints
      ^^^^^^^^^^^^^^^^^^^^

      **6. Link Bandwidth Availability**: The sum of bandwidth demands of virtual links routed over a physical link must not exceed its available bandwidth.
      
      .. math::
         \sum_{l_{v}^{m,w} \in \mathcal{L}_v} (y_{i,j}^{m,w} + y_{j,i}^{m,w}) B(l_{v}^{m,w}) \le B(l_{p}^{i,j}), \quad \forall l_{p}^{i,j} \in \mathcal{L}_p \quad (8)

Constraints (3), (4), and (5) cover node mapping, while (6), (7), and (8) cover link mapping.

Optimization Objective
~~~~~~~~~~~~~~~~~~~~~~

.. card::
   :class-header: sd-bg-success sd-text-white sd-font-weight-bold
   :class-card: sd-outline-success sd-rounded-1

   Revenue-to-Cost Ratio (R2C)
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^

   The primary objective in NFV-RA, especially for online service requests, is often to maximize the overall resource utilization, which facilitates long-term resource profit and request acceptance. A widely used metric to assess solution quality (:math:`S`) for an instance :math:`I` is the **Revenue-to-Cost Ratio (R2C)**:

   .. math::
      \text{maximize} \quad R2C(S) = \frac{\chi \cdot REV(S)}{COST(S)} \quad (1)

   where:

   * :math:`\chi` is a binary variable indicating solution feasibility: :math:`\chi=1` if solution :math:`S` satisfies all constraints, and :math:`\chi=0` otherwise.
   * :math:`REV(S)` is the revenue generated by embedding the VN :math:`\mathcal{G}_v`. If :math:`\chi=1`, it's calculated as the sum of resources requested by the VN:
     
     .. math::
        REV(S) = \sum_{n_v \in \mathcal{N}_v} C(n_v) + \sum_{l_v \in \mathcal{L}_v} B(l_v)

   * :math:`COST(S)` is the resource consumption in the PN due to embedding :math:`\mathcal{G}_v`. If :math:`\chi=1`, it's calculated as:
     
     .. math::
        COST(S) = \sum_{n_v \in \mathcal{N}_v} C(n_v) + \sum_{l_v \in \mathcal{L}_v} (|f_{\mathcal{L}}(l_v)| \times B(l_v))
     
     Here, $|f_{\mathcal{L}}(l_v)|$ quantifies the length (e.g., number of hops) of the physical path $\rho_p$ routing the virtual link $l_v$[cite: 58, 338].

Extensions for Emerging Network Scenarios
-----------------------------------------

The basic NFV-RA model can be extended to address unique challenges in emerging network scenarios. Virne supports such extensions. Key examples include:

Representative Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~

Heterogeneous Resourcing Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In networks with diverse physical node capabilities (e.g., different types of CPUs, GPUs, memory amounts), the node resource constraint (5) must hold for *each type* of resource $C \in \mathcal{C}$.

.. card::
    :class-header: sd-bg-warning sd-text-white sd-font-weight-bold
    :class-card: sd-outline-warning sd-rounded-1

    Resource Heterogeneity
    ^^^^^^^^^^^^^^^^^^^^^^
    
    .. math::
        x_{i}^{m} C_k(n_{v}^{m}) \le C_k(n_{p}^{i}), \quad \forall n_{v}^{m} \in \mathcal{N}_v, n_{p}^{i} \in \mathcal{N}_p, \forall k \in \mathcal{C}_{\text{types}}


Latency-aware Edge Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For time-sensitive services (e.g., in edge computing, 5G), virtual links $l_v$ may have maximum tolerable latency $D(l_v)$. The cumulative propagation delay $D(\rho_p)$ of the physical path $\rho_p$ routing $l_v$ must not exceed this:


.. card::
    :class-header: sd-bg-info sd-text-white sd-font-weight-bold
    :class-card: sd-outline-info sd-rounded-1

    Latency Requirements
    ^^^^^^^^^^^^^^^^^^^^
    
    .. math::
        D(\rho_p) = \sum_{l_p \in \rho_p} D(l_p) \le D(l_v)

Energy Efficient Networks
^^^^^^^^^^^^^^^^^^^^^^^^^

In green networking, minimizing energy consumption is crucial. The energy consumed by a physical node $n_p$, denoted $E(n_p)$, can depend on its status (idle/active) and workload. The optimization objective can be modified to a multi-objective function, e.g.:


.. card::
    :class-header: sd-bg-success sd-text-white sd-font-weight-bold
    :class-card: sd-outline-success sd-rounded-1

    Energy Efficiency
    ^^^^^^^^^^^^^^^^^
    
    .. math::
        \text{maximize} \quad -w_a \sum_{n_p \in \mathcal{N}_p} E(n_p) + w_b \cdot R2C(S)
    
    where $w_a$ and $w_b$ are weights for the different objectives.

.. note::

   These are illustrative extensions. The flexible design of Virne allows for the incorporation of various other constraints and objectives relevant to specific NFV-RA research problems. Refer to Appendix A.2 of our research paper for further discussion on these extensions.
