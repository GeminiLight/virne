import networkx as nx


def get_bfs_tree_level(network, source):
    node_level_dict = nx.single_source_shortest_path_length(network, source)
    max_depth = max(node_level_dict.values())
    level_list = [[] for i in range(max_depth + 1)]
    for k, v in node_level_dict.items():
        level_list[v].append(k)
    return level_list
