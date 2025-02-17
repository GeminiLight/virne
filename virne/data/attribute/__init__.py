from .base_attribute import *
from .node_attribute import *
from .link_attribute import *
from .graph_attribute import *


ATTRIBUTES_DICT = {
    # Information
    ('node', 'status'): NodeStatusAttribute,
    ('link', 'status'): LinkStatusAttribute,
    ('node', 'extrema'): NodeExtremaAttribute,
    ('link', 'extrema'): LinkExtremaAttribute,
    # Resource Constraints
    ('node', 'resource'): NodeResourceAttribute,
    ('link', 'resource'): LinkResourceAttribute,
    # QoS Requirements
    ('node', 'position'): NodePositionAttribute,
    ('link', 'latency'): LinkLatencyAttribute,
}


def create_attr_from_dict(attrs_dict):
    dict_copy = copy.deepcopy(attrs_dict)
    name = dict_copy.pop('name')
    owner = dict_copy.pop('owner')
    type = dict_copy.pop('type')
    assert (owner, type) in ATTRIBUTES_DICT.keys(), ValueError('Unsupproted attribute!')
    AttributeClass = ATTRIBUTES_DICT.get((owner, type))
    return AttributeClass(name, **dict_copy)


def create_attrs_from_setting(attrs_setting):
    attrs = {}
    for attr_dict in attrs_setting:
        attr = create_attr_from_dict(attr_dict)
        attrs[attr.name] = attr
    return attrs


if __name__ == '__main__':
    AttributeClass = ATTRIBUTES_DICT[('node', 'resource')]
    test_node_resource_attr = NodeResourceAttribute(name='cpu')
    print(test_node_resource_attr)
