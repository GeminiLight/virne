import networkx as nx


class TopologyGenerator:
    """
    Utility class to generate various networkx topologies for BaseNetwork.
    """
    @staticmethod
    def generate(type: str, num_nodes: int, **kwargs) -> nx.Graph:
        assert num_nodes >= 1, "num_nodes must be >= 1."
        match type:
            case 'path':
                return nx.path_graph(num_nodes)
            case 'star':
                return nx.star_graph(num_nodes - 1)
            case 'grid_2d':
                m = kwargs.get('m')
                n = kwargs.get('n')
                if m is None or n is None:
                    raise ValueError("'grid_2d' type requires 'm' and 'n' keyword arguments.")
                return nx.grid_2d_graph(m, n, periodic=False)
            case 'waxman':
                wm_alpha = kwargs.get('wm_alpha', 0.5)
                wm_beta = kwargs.get('wm_beta', 0.2)
                not_connected = True
                while not_connected:
                    G = nx.waxman_graph(num_nodes, wm_alpha, wm_beta)
                    not_connected = not nx.is_connected(G)
                return G
            case 'random':
                random_prob = kwargs.get('random_prob', 0.5)
                not_connected = True
                while not_connected:
                    G = nx.erdos_renyi_graph(num_nodes, random_prob, directed=False)
                    not_connected = not nx.is_connected(G)
                return G
            case 'balanced_tree':
                # Simple balanced tree where all internal nodes have same branching factor
                # r = branching factor (children per node)
                # h = height (levels from root to leaves)
                # Total nodes = 1 + r + r^2 + ... + r^h = (r^(h+1) - 1) / (r - 1)
                branching_factor = kwargs.get('r', 2)
                height = kwargs.get('h', 3)
                return nx.balanced_tree(branching_factor, height)
            case 'fat_tree':
                # Fat-Tree datacenter topology where num_nodes = number of hosts
                # k = number of ports per switch (must be even)
                # Creates a 3-layer Fat-Tree with k^3/4 hosts
                k = kwargs.get('k')
                if k is None:
                    # Calculate k from num_nodes: num_nodes = k^3/4, so k = (4*num_nodes)^(1/3)
                    k = int(round((4 * num_nodes) ** (1/3)))
                    # Ensure k is even
                    if k % 2 != 0:
                        k += 1
                return TopologyGenerator._generate_fat_tree(k)
            case 'tree':
                # Simple tree topology with switches (internal nodes) and hosts (leaves)
                # num_nodes = number of hosts (leaf nodes)
                # branching_factor = children per internal node (default: 2 for binary tree)
                branching_factor = kwargs.get('branching_factor', 2)
                return TopologyGenerator._generate_tree(num_nodes, branching_factor)
            case _:
                raise NotImplementedError(f"Graph type '{type}' is not implemented.")

    @staticmethod
    def _generate_fat_tree(k: int) -> nx.Graph:
        """
        Generate a Fat-Tree topology with k-port switches.

        Args:
            k: Number of ports per switch (must be even)

        Returns:
            A Fat-Tree graph with:
            - (k/2)^2 core switches
            - k pods, each with k/2 aggregation and k/2 edge switches
            - (k/2) hosts per edge switch
            - Total hosts: k^3/4
        """
        if k % 2 != 0:
            raise ValueError("k must be even for Fat-Tree topology")

        G = nx.Graph()

        # Calculate number of switches at each layer
        num_pods = k
        num_core = (k // 2) ** 2
        num_agg_per_pod = k // 2
        num_edge_per_pod = k // 2
        num_hosts_per_edge = k // 2

        # Node ID tracking
        node_id = 0

        # Create core switches
        core_switches = list(range(node_id, node_id + num_core))
        node_id += num_core
        G.add_nodes_from(core_switches, layer='core')

        # Create pods
        for pod in range(num_pods):
            # Aggregation switches in this pod
            agg_switches = list(range(node_id, node_id + num_agg_per_pod))
            node_id += num_agg_per_pod
            G.add_nodes_from(agg_switches, layer='aggregation', pod=pod)

            # Edge switches in this pod
            edge_switches = list(range(node_id, node_id + num_edge_per_pod))
            node_id += num_edge_per_pod
            G.add_nodes_from(edge_switches, layer='edge', pod=pod)

            # Connect aggregation to core
            for i, agg in enumerate(agg_switches):
                # Each agg switch connects to k/2 core switches
                start_core = i * (k // 2)
                for j in range(k // 2):
                    core_idx = start_core + j
                    if core_idx < len(core_switches):
                        G.add_edge(agg, core_switches[core_idx])

            # Connect edge to aggregation (full bipartite within pod)
            for edge in edge_switches:
                for agg in agg_switches:
                    G.add_edge(edge, agg)

            # Create hosts and connect to edge switches
            for edge in edge_switches:
                hosts = list(range(node_id, node_id + num_hosts_per_edge))
                node_id += num_hosts_per_edge
                G.add_nodes_from(hosts, layer='host', pod=pod)
                for host in hosts:
                    G.add_edge(edge, host)

        return G

    @staticmethod
    def _generate_tree(num_hosts: int, branching_factor: int = 2) -> nx.Graph:
        """
        Generate a simple tree topology with switches and hosts.

        Internal nodes are switches (layer='switch', used for routing only).
        Leaf nodes are hosts (layer='host', can host VNFs).

        Args:
            num_hosts: Number of host (leaf) nodes
            branching_factor: Number of children per switch node (default: 2 for binary tree)

        Returns:
            NetworkX graph with:
            - Internal nodes: switches (layer='switch')
            - Leaf nodes: hosts (layer='host')

        Example:
            num_hosts=4, branching_factor=2 creates:

                    Root (0)
                   /        \
              Switch 1    Switch 2
              /    \      /      \
           Host 3  Host 4  Host 5  Host 6

            Total: 3 switches + 4 hosts = 7 nodes
        """
        import math

        G = nx.Graph()

        if num_hosts <= 0:
            raise ValueError("num_hosts must be > 0")
        if branching_factor < 1:
            raise ValueError("branching_factor must be >= 1")

        # Calculate tree height needed for num_hosts leaves
        # For a tree with branching factor b and height h:
        # Number of leaves = b^h
        height = math.ceil(math.log(num_hosts, branching_factor))

        # Calculate total number of internal nodes (switches)
        # For a complete tree: internal_nodes = (b^h - 1) / (b - 1)
        if branching_factor == 1:
            num_switches = height
        else:
            num_switches = (branching_factor ** height - 1) // (branching_factor - 1)

        node_id = 0

        # Create switches (internal nodes) using BFS
        switches = list(range(node_id, node_id + num_switches))
        node_id += num_switches
        G.add_nodes_from(switches, layer='switch')

        # Build tree structure
        for i, parent in enumerate(switches):
            for child_idx in range(branching_factor):
                child_id = (i * branching_factor) + child_idx + 1
                if child_id < num_switches:
                    # Connect to another switch
                    G.add_edge(parent, switches[child_id])

        # Add hosts (leaf nodes) - attach to leaf-level switches
        # Find switches at the last level
        leaf_level_start = (branching_factor ** (height - 1) - 1) // (branching_factor - 1) if branching_factor > 1 else height - 1
        leaf_switches = switches[leaf_level_start:num_switches]

        # Distribute hosts among leaf switches
        hosts_added = 0
        for switch in leaf_switches:
            if hosts_added >= num_hosts:
                break
            # Add host and connect to switch
            host_id = node_id
            G.add_node(host_id, layer='host')
            G.add_edge(switch, host_id)
            node_id += 1
            hosts_added += 1

        # If we need more hosts, distribute them among existing leaf switches
        switch_idx = 0
        while hosts_added < num_hosts:
            switch = leaf_switches[switch_idx % len(leaf_switches)]
            host_id = node_id
            G.add_node(host_id, layer='host')
            G.add_edge(switch, host_id)
            node_id += 1
            hosts_added += 1
            switch_idx += 1

        # Mark root node
        G.nodes[0]['is_root'] = True

        return G
