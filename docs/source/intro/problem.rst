Network Resource Allocation Problem
============================================


.. card::
    :class-card: sd-outline-info  sd-rounded-1
    :class-body: sd-font-weight-bold

    
    #. An NP-hard combinatorial optimization problem
    #. Two subproblems: node mapping and link mapping
    #. Various evaluation metrics: 


The objective of Virtual Network Embedding (VNE) is to map the virtual nodes and links 
onto the substrate network while satisfying various constraints, 
such as resource availability, capacity, and latency requirements. 
VNE can be divided into two subproblems: node mapping and link mapping.

.. image:: ../_static/vne-example.png
   :width: 1000
   :alt: An example of resource allocation problem in network virtualization (source: `IJCAI'24 - FlagVNE<https://arxiv.org/abs/2404.12633>`_)

System Model
------------
In a practical network system, users' service requests continuously arrive at the network infrastructure.

- **Physical Network**: Network infrastructure is virtualized as a physical network
- **Virtual Network**: User service requests are virtualized as virtual networks

.. card::
   :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
   :class-card: sd-outline-info  sd-rounded-1
   :class-footer: sd-font-weight-bold

   Physical Network
   ^^^

   The physical network is modelled as a undirected graph :math:`G^p=(N^p, L^p)`, where

   - :math:`N^p` is the set of physical nodes. Each physical node :math:`n^p \in N^p` has a set of computing resources, such as CPU, GPU memory, and bandwidth, which are represented as a vector :math:`C(n^p)`.
   - :math:`L^p` is the set of physical links. Each physical link :math:`l^p \in L^p` has a bandwidth capacity :math:`B(l^p)`.

.. card::
   :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
   :class-card: sd-outline-info  sd-rounded-1
   :class-footer: sd-font-weight-bold

   Virtual Network
   ^^^

   Each virtual network is modelled as a undirected graph :math:`G^v=(N^v, L^v, d^v)`, where

   - :math:`N^v` is the set of virtual nodes. Each virtual node :math:`n^v \in N^v` has a set of resource requirements, such as CPU, GPU memory, and bandwidth, which are represented as a vector :math:`R(n^v)`.
   - :math:`L^v` is the set of virtual links. Each virtual link :math:`l^v \in L^v` has a bandwidth requirement :math:`B(l^v)`.
   - :math:`d^v` is the lifetime of the user service request. Once the VNR is accepted, it will be maintained for :math:`d^v` time slots.


Mapping Process
---------------

The mapping process aims to map the virtual nodes and links onto the substrate network with minimal resource cost while satisfying various QoS constraints.

This graph mapping process :math:`f: G^v \rightarrow G^p` can be divided into two subproblems: node mapping and link mapping.

.. card::
   :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
   :class-card: sd-outline-info  sd-rounded-1
   :class-footer: sd-font-weight-bold

   Node Mapping :math:`f_n: N^v \rightarrow N^p`
   ^^^

   Node mapping involves assigning each virtual node :math:`n^v \in N^v` to a physical node :math:`n^p \in N^p`.

   In this process, the following constraints should be satisfied:

   - **One-to-one mapping constraints**: Each virtual node should be mapped to exactly one substrate node.

   .. math::
       :label: formulation-eq-node-1

       f_n(n^v) = n^p, \quad \forall n^v \in N^v

   - **Computing Resource Availability**: The computing resources required by the virtual node should be available on the physical node.

   .. math::
       :label: formulation-eq-node-2

       C(n^p) \geq C(n^v), \quad \forall n^v \in N^v, n^p = f_n(n^v)


.. card::
   :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
   :class-card: sd-outline-info  sd-rounded-1
   :class-footer: sd-font-weight-bold

   Link Mapping :math:`f_l: L^v \rightarrow P^p`
   ^^^

   Link mapping involves finding a physical path :math:`p^p \in P^p` for each virtual link :math:`l^v \in L^v`.

   In this process, the following constraints should be satisfied:
    
   - **Connectivity constraints**: The mapping should preserve the connectivity of the virtual network, i.e., if there is a virtual link between two virtual nodes, the corresponding physical nodes should be connected by a physical link.

   .. math::
       :label: formulation-eq-link-1

         f_n(n^v_1) \neq f_n(n^v_2) \Rightarrow \exists l^p \in L^p, f_l(l^v) = l^p, n^v_1, n^v_2 \in N^v


   - **Link resource constraint**:  The sum of the bandwidth requirements of the virtual links mapped to a physical link cannot exceed its capacity.
   - **Link-to-path mapping constraint**: Each virtual link can only be mapped to a path consisting of physical links.
   - **Path length constraint**:  The length of the path used to map a virtual link cannot exceed a predefined maximum value, resulting from the QoS requirements (e.g., delay).


Evaluation Metric
-----------------


The evaluation of Virtual Network Embedding (VNE) algorithms is crucial for measuring their performance and comparing their results. Evaluation metrics aim to capture different aspects of the embedding process, such as resource utilization, link mapping efficiency, or service quality. In general, there are several evaluation metrics that are commonly used to assess the performance of VNE algorithms.

- **Acceptance Ratio**: The acceptance ratio is a metric that evaluates the number of VN requests that are successfully embedded in the SN. It is calculated as the ratio of the number of successful embeddings to the total number of embedding requests. A higher acceptance ratio indicates a better embedding performance.

- **Revenue-to-cost**: Revenue-to-cost measures the revenue generated by embedding virtual networks against the cost incurred in embedding them. The revenue is usually derived from the services offered by the virtual networks, while the cost includes the resources consumed during embedding, such as the bandwidth usage and energy consumption. R/C is a crucial metric for service providers, as it helps them optimize their resource utilization and improve their profitability. The higher the R/C ratio, the better the profitability of the service provider.

- **Cost**: The embedding cost is the primary metric for evaluating the VNE problem. It represents the amount of resources used to embed a virtual network (VN) in the substrate network (SN). The embedding cost can be measured by adding up the cost of embedding individual virtual nodes and links and can be expressed in terms of computational resources, bandwidth, or any other resource that is being utilized in the embedding process. A lower embedding cost indicates a better embedding performance.

- **Revenue**: Revenue is a metric used to evaluate the business aspect of the VNE problem. It represents the revenue generated by the virtual network provider (VNP) by embedding the VN in the SN. The revenue can be calculated by taking into account the amount of resources used to embed the VN, the number of requests, and the revenue generated per request. A higher revenue indicates a better embedding performance from a business perspective.

- **Profit**: Profit is another metric used to evaluate the business aspect of the VNE problem. It represents the profit generated by the VNP by embedding the VN in the SN, which is calculated by subtracting the embedding cost from the revenue. A higher profit indicates a better embedding performance from a business perspective.

- **Resource Utilization**: Resource utilization measures how effectively the resources in the substrate network are being used. It can be measured by the percentage of resources utilized in the SN after embedding the VN. A higher resource utilization indicates a better embedding performance, as the resources are being utilized efficiently.

- **Delay**: Delay is a metric that evaluates the end-to-end delay in the VN after it is embedded in the SN. It can be measured by the average delay experienced by the packets in the VN or the maximum delay experienced by any packet in the VN. A lower delay indicates a better embedding performance.

- **Throughput**: Throughput is a metric that evaluates the amount of data that can be transmitted through the VN after it is embedded in the SN. It can be measured by the average throughput of the packets in the VN or the maximum throughput of any packet in the VN. A higher throughput indicates a better embedding performance.

These evaluation metrics are used to measure the performance of VNE algorithms and compare them with each other. However, the choice of evaluation metrics depends on the specific requirements and objectives of the VNE problem.