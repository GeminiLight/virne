Background
========================================

Network virtualization
----------------------

Network virtualization (NV) emerges as a pioneering technology that facilitates dynamic management of Internet architecture.
Through network slicing and shared infrastructure, NV enables the deployment of multiple user service requests within the same underlying infrastructure.


NV technology is widely used in various network environments, including cloud computing, edge computing, Internet of Things (IoT), and 5G networks.

.. tab-set::

    .. tab-item:: Cloud computing
        :sync: key1

        .. card::
            :class-header: sd-bg-primary  sd-text-white sd-font-weight-bold
            :class-card: sd-outline-primary  sd-rounded-1
            :class-footer: sd-font-weight-bold

            NV-enabled Cloud computing
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            In cloud computing, customers often require the ability to deploy complex virtual networks with multiple nodes and links. 
            VNE can be used to efficiently map these virtual networks onto physical infrastructure, ensuring high performance and availability.
    
    .. tab-item:: Edge computing
        :sync: key2

        .. card::
            :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
            :class-card: sd-outline-primary  sd-rounded-1
            :class-footer: sd-font-weight-bold

            NV-enabled Edge computing
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            Edge computing involves distributing computing resources and services closer to the edge of the network, near where the data is generated. 
            VNE can be used to map virtual networks onto edge infrastructure, ensuring efficient use of resources and optimal performance.

    .. tab-item:: Internet of Things
        :sync: key3

        .. card::
            :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
            :class-card: sd-outline-primary  sd-rounded-1
            :class-footer: sd-font-weight-bold

            NV-enabled Internet of Things
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            IoT devices often require connectivity and access to various services. VNE can be used to create virtual networks that connect these devices to each other and to services in the cloud.

    .. tab-item:: 5G Networks
        :sync: key4

        .. card::
            :class-header: sd-bg-primary sd-text-white sd-font-weight-bold
            :class-card: sd-outline-primary  sd-rounded-1
            :class-footer: sd-font-weight-bold

            NV-enabled 5G Networks
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            With the emergence of 5G technology, there is a need for efficient and optimized network resource utilization. VNE can be used to allocate virtual network resources in a 5G network, improving performance and reducing costs.

Resource Allocation Problem
---------------------------

Under the NV paradigm, 

- The user service requests are abstracted as **Virtual Network (VNs)** Requests
- The underlying infrastructure is abstracted as a **Physical Network (PN)**.

A primary challenge in NV involves the efficient embedding of VNs on the PN, which referred to as 

- Virtual Network Embedding (VNE)
- Virtual Network Function Placement (VNF Placement)
- Service Function Chain Deployment (SFC Deployment)

In network environments, multiple tenants request virtual networks with varying resource requirements, 
and each tenant's virtual network should be isolated from others. 
To support the rapid and efficient deployment of virtual networks, 
network virtualization technology is used to abstract the underlying physical infrastructure 
and provide a logical view of the network to the tenants. 
The VNE problem involves mapping virtual networks to the physical infrastructure while ensuring various constraints and QoS requirements .

.. hint::

    This resource allocation is a **NP-hard online combinatorial optimization problem**.

