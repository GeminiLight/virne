Background
==========

Virtual Network Embedding (VNE) is a crucial problem in the field of network virtualization that aims to allocate virtual resources (nodes, links, and other resources) requested by virtual networks on the physical infrastructure (nodes, links, and other resources) provided by the underlying physical network. The VNE problem has emerged as a critical research area due to the rapid growth of cloud computing and the increasing demand for the deployment of multiple services and applications that require flexible and efficient use of network resources.

System Description
------------------

In network environments, multiple tenants request virtual networks with varying resource requirements, 
and each tenant's virtual network should be isolated from others. 
To support the rapid and efficient deployment of virtual networks, 
network virtualization technology is used to abstract the underlying physical infrastructure 
and provide a logical view of the network to the tenants. 
The VNE problem involves mapping virtual networks to the physical infrastructure while ensuring various constraints and QoS requirements .


Realistic Scenarios
-------------------


.. card::
   :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
   :class-card: sd-outline-info  sd-rounded-1
   :class-footer: sd-font-weight-bold

   Cloud computing
   ^^^

   In cloud computing, customers often require the ability to deploy complex virtual networks with multiple nodes and links. VNE can be used to efficiently map these virtual networks onto physical infrastructure, ensuring high performance and availability.


.. card::
   :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
   :class-card: sd-outline-info  sd-rounded-1
   :class-footer: sd-font-weight-bold

   5G networks
   ^^^

   With the emergence of 5G technology, there is a need for efficient and optimized network resource utilization. VNE can be used to allocate virtual network resources in a 5G network, improving performance and reducing costs.


.. card::
   :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
   :class-card: sd-outline-info  sd-rounded-1
   :class-footer: sd-font-weight-bold

   Internet of Things (IoT)
   ^^^

   IoT devices often require connectivity and access to various services. VNE can be used to create virtual networks that connect these devices to each other and to services in the cloud.


.. card::
   :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
   :class-card: sd-outline-info  sd-rounded-1
   :class-footer: sd-font-weight-bold

   Edge computing
   ^^^

   Edge computing involves distributing computing resources and services closer to the edge of the network, near where the data is generated. VNE can be used to map virtual networks onto edge infrastructure, ensuring efficient use of resources and optimal performance.


.. card::
   :class-header: sd-bg-info  sd-text-white sd-font-weight-bold
   :class-card: sd-outline-info  sd-rounded-1
   :class-footer: sd-font-weight-bold

   Content delivery networks (CDN)
   ^^^

   CDNs are used to deliver content to users around the world. VNE can be used to map virtual networks onto physical infrastructure in a CDN, ensuring fast and efficient delivery of content.
