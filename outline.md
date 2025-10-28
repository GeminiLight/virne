Experimental Setup Outline: Binary Tree Topology

  1. Physical Network Design

  1.1 Topology Selection

  To match the paper's two topology scales (GEANT: 40 nodes, WX100: 100 nodes), we need two binary tree configurations:

  # Configuration 1: Small Tree (comparable to GEANT - 40 total nodes)
  # Binary tree with ~40 total nodes
  # If we want 40 total nodes with binary tree:
  #   - For 16 hosts: 15 switches + 16 hosts = 31 nodes
  #   - For 20 hosts: 19 switches + 20 hosts = 39 nodes ✓ CLOSE!
  #   - For 24 hosts: 23 switches + 24 hosts = 47 nodes

  small_tree_config = {
      'topology': {
          'type': 'tree',
          'num_nodes': 20,        # 20 HOST nodes (leaves)
          'branching_factor': 2   # Binary tree
      }
      # Total nodes: 19 switches + 20 hosts = 39 nodes ≈ GEANT (40)
  }

  # Configuration 2: Large Tree (comparable to WX100 - 100 total nodes)
  # Binary tree with ~100 total nodes
  # If we want 100 total nodes:
  #   - For 50 hosts: 49 switches + 50 hosts = 99 nodes ✓ CLOSE!
  #   - For 51 hosts: 50 switches + 51 hosts = 101 nodes ✓

  large_tree_config = {
      'topology': {
          'type': 'tree',
          'num_nodes': 50,        # 50 HOST nodes (leaves)
          'branching_factor': 2   # Binary tree
      }
      # Total nodes: 49 switches + 50 hosts = 99 nodes ≈ WX100 (100)
  }

  Key Decision: Should we match:
  - Total node count (39 vs 40, 99 vs 100)?  ← Recommended
  - Host node count (20 hosts, 50 hosts)?
  - Link count (GEANT has 61 links)?

  ---
  1.2 Resource Allocation

  Following the paper's specification:

  # Physical Network Resource Settings
  p_net_config = {
      'topology': {
          'type': 'tree',
          'num_nodes': 20,  # or 50 for large tree
          'branching_factor': 2
      },
      'node_attrs_setting': [
          # CPU: [50, 100] for hosts, 0 for switches
          {'name': 'cpu', 'type': 'resource', 'distribution': 'uniform',
           'low': 50, 'high': 100, 'dtype': 'int'},

          # Storage: [50, 100] for hosts, 0 for switches
          {'name': 'storage', 'type': 'resource', 'distribution': 'uniform',
           'low': 50, 'high': 100, 'dtype': 'int'},

          # GPU: [50, 100] for hosts, 0 for switches
          {'name': 'gpu', 'type': 'resource', 'distribution': 'uniform',
           'low': 50, 'high': 100, 'dtype': 'int'},
      ],
      'link_attrs_setting': [
          # Bandwidth: [50, 100] for ALL links
          {'name': 'bw', 'type': 'resource', 'distribution': 'uniform',
           'low': 50, 'high': 100, 'dtype': 'int'},
      ]
  }

  # CRITICAL: Post-processing to set switch resources to 0
  for node_id in p_net.nodes():
      if p_net.nodes[node_id].get('layer') == 'switch':
          p_net.nodes[node_id]['cpu'] = 0
          p_net.nodes[node_id]['storage'] = 0
          p_net.nodes[node_id]['gpu'] = 0

  ---
  2. Virtual Network Request (VNR) Generation

  2.1 VNR Characteristics

  vnr_config = {
      'num_vnrs': 1000,  # 1000 VNRs per simulation run

      'topology': {
          'type': 'random',           # Erdos-Renyi random graph
          'num_nodes': [2, 10],       # Varying sizes: 2 to 10 nodes
          'random_prob': 0.5          # 50% interconnection probability
      },

      'node_attrs_setting': [
          # CPU demand: [0, 20]
          {'name': 'cpu', 'distribution': 'uniform', 'low': 0, 'high': 20, 'dtype': 'int'},

          # Storage demand: [0, 20]
          {'name': 'storage', 'distribution': 'uniform', 'low': 0, 'high': 20, 'dtype': 'int'},

          # GPU demand: [0, 20]
          {'name': 'gpu', 'distribution': 'uniform', 'low': 0, 'high': 20, 'dtype': 'int'},
      ],

      'link_attrs_setting': [
          # Bandwidth demand: [0, 50]
          {'name': 'bw', 'distribution': 'uniform', 'low': 0, 'high': 50, 'dtype': 'int'},
      ],

      'lifetime': {
          'distribution': 'exponential',
          'mean': 500  # Average lifetime: 500 time units
      },

      'arrival': {
          'distribution': 'poisson',
          'rate': 0.04  # η = 0.04 for small tree (adjusted from paper's values)
      }
  }

  ---
  3. Experimental Parameters

  3.1 Arrival Rate (η) Selection

  The paper uses different rates for different topology sizes:
  - GEANT (40 nodes): η = 0.001
  - WX100 (100 nodes): η = 0.08

  Reasoning: Larger networks → higher capacity → higher arrival rate

  For binary tree:

  # Small Tree (39 nodes, 20 hosts)
  # Capacity ≈ GEANT (40 nodes)
  # But tree has FEWER hosts than GEANT (20 vs potentially all 40)
  # So we need to adjust η proportionally

  small_tree_eta = 0.001 * (20 / 40)  # ≈ 0.0005
  # Alternative: Match GEANT exactly
  small_tree_eta = 0.001  # Same as GEANT

  # Large Tree (99 nodes, 50 hosts)
  # Capacity ≈ WX100 (100 nodes)
  # Adjust proportionally
  large_tree_eta = 0.08 * (50 / 100)  # ≈ 0.04

  Recommendation: Start with GEANT/WX100 rates, then adjust based on acceptance rate

  ---
  4. Implementation Steps

  Step 1: Create Physical Network Generator

  # File: configs/p_net_tree_small.yaml
  topology:
    type: tree
    num_nodes: 20
    branching_factor: 2

  node_attrs_setting:
    - name: cpu
      type: resource
      owner: node
      distribution: uniform
      low: 50
      high: 100
      dtype: int
    - name: storage
      type: resource
      owner: node
      distribution: uniform
      low: 50
      high: 100
      dtype: int
    - name: gpu
      type: resource
      owner: node
      distribution: uniform
      low: 50
      high: 100
      dtype: int

  link_attrs_setting:
    - name: bw
      type: resource
      owner: link
      distribution: uniform
      low: 50
      high: 100
      dtype: int

  Step 2: Create VNR Simulator Configuration

  # File: configs/v_net_simulator.yaml
  num_v_nets: 1000
  v_net_size:
    min: 2
    max: 10
  topology:
    type: random
    random_prob: 0.5

  node_attrs_setting:
    - name: cpu
      distribution: uniform
      low: 0
      high: 20
      dtype: int
    - name: storage
      distribution: uniform
      low: 0
      high: 20
      dtype: int
    - name: gpu
      distribution: uniform
      low: 0
      high: 20
      dtype: int

  link_attrs_setting:
    - name: bw
      distribution: uniform
      low: 0
      high: 50
      dtype: int

  arrival_rate: 0.001  # η for small tree
  lifetime:
    distribution: exponential
    mean: 500

  Step 3: Modify Physical Network to Zero Out Switches

  # File: scripts/prepare_tree_topology.py

  from virne.network import PhysicalNetwork

  def prepare_tree_topology(config_path):
      """Load tree topology and set switch resources to 0."""

      # Load from config
      p_net = PhysicalNetwork.from_setting(config_path)

      # Post-process: Zero out switch resources
      num_switches = 0
      num_hosts = 0

      for node_id in p_net.nodes():
          layer = p_net.nodes[node_id].get('layer', 'unknown')

          if layer == 'switch':
              # Switches cannot host VNFs
              p_net.nodes[node_id]['cpu'] = 0
              p_net.nodes[node_id]['storage'] = 0
              p_net.nodes[node_id]['gpu'] = 0
              num_switches += 1
          else:  # layer == 'host'
              num_hosts += 1

      print(f"Tree topology prepared:")
      print(f"  Total nodes: {p_net.num_nodes}")
      print(f"  Switches: {num_switches}")
      print(f"  Hosts: {num_hosts}")
      print(f"  Links: {p_net.num_links}")

      return p_net

  Step 4: Run Experiment

  # File: experiments/tree_topology_experiment.py

  from virne.simulator import Simulator
  from virne.solver.heuristic import NodeRankSolver
  from scripts.prepare_tree_topology import prepare_tree_topology

  # 1. Setup physical network
  p_net = prepare_tree_topology('configs/p_net_tree_small.yaml')

  # 2. Setup solver
  solver = NodeRankSolver(...)

  # 3. Setup simulator
  simulator = Simulator(
      p_net=p_net,
      v_net_simulator_config='configs/v_net_simulator.yaml',
      solver=solver,
      num_epochs=1000  # 1000 VNRs
  )

  # 4. Run simulation
  results = simulator.run()

  # 5. Analyze results
  print(f"Acceptance rate: {results['acceptance_rate']}")
  print(f"Average revenue: {results['avg_revenue']}")
  print(f"Average cost: {results['avg_cost']}")

  ---
  5. Key Differences from Paper's Setup

  Topology Differences:

  | Aspect     | GEANT/WX100 (Paper) | Binary Tree (This Experiment) |
  |------------|---------------------|-------------------------------|
  | Structure  | Waxman random graph | Hierarchical tree             |
  | Redundancy | Multiple paths      | Single path between hosts     |
  | Bottleneck | Distributed         | Root node (backbone)          |
  | Realistic? | Yes (ISP networks)  | Datacenter-like               |
  | Host nodes | All nodes can host  | Only leaf nodes can host      |

  Expected Behavioral Differences:

  1. Lower acceptance rate in tree due to:
    - Single path → no alternative routes
    - Root bottleneck → congestion at backbone
    - Fewer hosting nodes (only leaves)
  2. Higher path costs due to:
    - All cross-subtree traffic goes through root
    - Longer average path length
  3. Different scalability because:
    - Tree height grows logarithmically
    - More layers = more switches = fewer hosting nodes

  ---
  6. Experimental Metrics to Track

  metrics = {
      'acceptance_rate': "Percentage of accepted VNRs",
      'avg_revenue': "Average revenue per VNR",
      'avg_cost': "Average embedding cost",
      'revenue_cost_ratio': "Revenue/cost ratio",
      'avg_path_length': "Average number of hops per virtual link",
      'backbone_utilization': "Bandwidth usage on root links",
      'solution_time': "Time to solve each VNR",
      'resource_utilization': {
          'cpu': "Average CPU utilization on hosts",
          'bandwidth': "Average link bandwidth utilization"
      }
  }

  ---
  7. Research Questions for Tree Topology

  1. How does tree structure affect acceptance rate?
    - Compare to GEANT/WX100 baseline
  2. Is the root a bottleneck?
    - Track bandwidth usage on root links vs. leaf links
  3. How does branching factor affect performance?
    - Test binary (b=2) vs ternary (b=3) vs quad (b=4)
  4. Does tree depth matter?
    - Compare shallow wide trees vs. deep narrow trees
  5. How does VNR size distribution interact with tree structure?
    - Are small VNRs easier to embed? (stay in same subtree)

  ---
  Summary Outline

  Phase 1: Setup (Week 1)

  - Create tree topology generator configurations
  - Implement switch resource zeroing
  - Validate topology structure

  Phase 2: VNR Generation (Week 1)

  - Configure VNR simulator
  - Generate 1000 VNRs with specified parameters
  - Validate VNR distributions

  Phase 3: Baseline Experiments (Week 2)

  - Run on small tree (20 hosts, η=0.001)
  - Run on large tree (50 hosts, η=0.04)
  - Compare with GEANT/WX100 results

  Phase 4: Parameter Tuning (Week 2-3)

  - Adjust η based on acceptance rates
  - Test different branching factors
  - Optimize solver parameters

  Phase 5: Analysis (Week 3)

  - Compute all metrics
  - Generate plots (acceptance rate, revenue, path length)
  - Compare tree vs. Waxman topologies

  Would you like me to implement any specific part of this outline?
