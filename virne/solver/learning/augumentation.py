import copy
import numpy as np
import networkx as nx

from virne.network import BaseNetwork, PhysicalNetwork, VirtualNetwork


def del_physical_links(p_net: PhysicalNetwork, resource_threshold: int = 0):
    def available_link(n1, n2):
        for k, v in resource_threshold.items():
            p_link = p_net.links[(n1, n2)]
        return p_link[k] >= v
    sub_graph = p_net.get_subgraph_view(filter_edge=available_link)
    return sub_graph

def add_physical_links(p_net: PhysicalNetwork, num_added_links: int = 0, link_resources: int = 0):
    augumented_p_net = copy.deepcopy(p_net)
    existing_links = list(augumented_p_net.edges)
    # randomly add links
    link_attrs_dict = {}
    for l_attr in p_net.link_attrs.values():
        link_attrs_dict[l_attr.name] = l_attr

    p_net.link_attr_types
    p_net.get_link_attrs(p_net.link_attr_types)
    for i in range(num_added_links):
        while True:
            src = np.random.randint(0, augumented_p_net.num_nodes)
            dst = np.random.randint(0, augumented_p_net.num_nodes)
            if src != dst and (src, dst) not in existing_links:
                break
        augumented_p_net.add_edge(src, dst, **link_attrs_dict)
    return augumented_p_net

def add_virtual_links(v_net: VirtualNetwork, num_added_links: int = 0, link_resources: int = 0):
    augumented_v_net = copy.deepcopy(v_net)
    existing_links = list(augumented_v_net.edges)

    link_attrs_dict = {}
    for l_attr in v_net.link_attrs.values():
        link_attrs_dict[l_attr.name] = l_attr

    # randomly add links
    for i in range(num_added_links):
        while True:
            src = np.random.randint(0, augumented_v_net.num_nodes)
            dst = np.random.randint(0, augumented_v_net.num_nodes)
            if src != dst and (src, dst) not in existing_links:
                break
        augumented_v_net.add_edge(src, dst, **link_attrs_dict)

