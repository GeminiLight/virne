import networkx as nx
from data.attribute import Attribute


def get_bfs_tree_level(network, source):
    node_level_dict = nx.single_source_shortest_path_length(network, source)
    max_depth = max(node_level_dict.values())
    level_list = [[] for i in range(max_depth + 1)]
    for k, v in node_level_dict.items():
        level_list[v].append(k)
    return level_list

def flatten_recurrent_dict(recurrent_dict):
   for i in getattr(recurrent_dict, 'values', lambda :recurrent_dict)():
        if isinstance(i, (str, float, int)):
            yield i
        elif i is not None:
            yield from flatten_recurrent_dict(i)
        elif i is None:
            raise ValueError('Unsupported types')
        else:
            pass
