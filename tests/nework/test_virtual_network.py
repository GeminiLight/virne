import pytest
import numpy as np
import networkx as nx
import logging
from unittest.mock import PropertyMock, patch
from virne.network.virtual_network import VirtualNetwork


@pytest.fixture
def basic_vn():
    """A VirtualNetwork instance with default initialization."""
    return VirtualNetwork()

@pytest.fixture
def vn_with_resource_settings():
    """A VirtualNetwork instance with predefined resource attribute settings (with required fields)."""
    node_settings = [
        {'name': 'cpu', 'type': 'resource', 'owner': 'node', 'distribution': 'uniform', 'low': 10, 'high': 50, 'dtype': int},
        {'name': 'ram', 'type': 'resource', 'owner': 'node', 'distribution': 'uniform', 'low': 100, 'high': 200, 'dtype': int}
    ]
    link_settings = [
        {'name': 'bw', 'type': 'resource', 'owner': 'link', 'distribution': 'uniform', 'low': 50, 'high': 100, 'dtype': int},
        {'name': 'latency', 'type': 'resource', 'owner': 'link', 'distribution': 'uniform', 'low': 5, 'high': 10, 'dtype': int}
    ]
    vn = VirtualNetwork(node_attrs_setting=node_settings, link_attrs_setting=link_settings)
    return vn

class TestVirtualNetwork:

    def test_initialization_default(self, basic_vn):
        pass