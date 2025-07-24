# Virne Framework: Modularity and Interface Documentation

This document provides a comprehensive description of Virne's modular architecture, concrete interfaces, data structures, and API contracts between modules, demonstrating the framework's extensibility for NFV resource allocation research.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Module Interfaces](#core-module-interfaces)
3. [Data Structures and Contracts](#data-structures-and-contracts)
4. [Feature Constructor to Policy Network Flow](#feature-constructor-to-policy-network-flow)
5. [Extensibility Mechanisms](#extensibility-mechanisms)
6. [Module Integration Examples](#module-integration-examples)
7. [API Reference](#api-reference)

## Architecture Overview

Virne follows a modular, registry-based architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Virne Framework Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│  Application Layer: Solvers (Exact, Heuristic, RL-based)       │
├─────────────────────────────────────────────────────────────────┤
│  Learning Layer: Feature Constructors ←→ Policy Networks        │
├─────────────────────────────────────────────────────────────────┤
│  Control Layer: Environment ←→ Controller ←→ Solution           │
├─────────────────────────────────────────────────────────────────┤
│  Network Layer: PhysicalNetwork ←→ VirtualNetwork               │
├─────────────────────────────────────────────────────────────────┤
│  Foundation Layer: Registries, Utils, Configuration             │
└─────────────────────────────────────────────────────────────────┘
```

## Core Module Interfaces

### 1. Solver Interface

**Base Contract:**
```python
class Solver:
    def __init__(self, controller: Controller, recorder: Recorder, 
                 counter: Counter, logger: Logger, config: DictConfig)
    def ready(self) -> None
    def solve(self, instance: dict) -> Solution
```

**Registry System:**
```python
@SolverRegistry.register('custom_solver', solver_type='learning')
class CustomSolver(Solver):
    def solve(self, instance: dict) -> Solution:
        # Implementation here
        pass
```

**Data Flow:**
```
Environment → Solver.solve(instance) → Solution → Environment.step()
```

### 2. Environment Interface

**Base Contract:**
```python
class BaseEnvironment:
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]
    def step(self, action: Any) -> Tuple[obs, reward, done, info]
    def get_observation(self) -> Dict[str, Any]
    def compute_reward(self) -> float
```

**Key Implementations:**
- `SolutionStepEnvironment`: Accepts complete solutions
- `JointPRStepEnvironment`: Step-by-step node placement

### 3. Controller Interface

**Core Operations:**
```python
class Controller:
    # Resource Management
    def deploy(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, 
               solution: Solution) -> bool
    def release(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, 
                solution: Solution) -> bool
    
    # Placement Operations
    def place_and_route(self, v_net: VirtualNetwork, p_net: PhysicalNetwork,
                       v_node_id: int, p_node_id: int, solution: Solution,
                       **kwargs) -> Tuple[bool, dict]
    
    # Candidate Finding
    def find_candidate_nodes(self, v_net: VirtualNetwork, 
                           p_net: PhysicalNetwork, v_node_id: int,
                           filter: List[int] = []) -> List[int]
```

## Data Structures and Contracts

### 1. Solution Data Structure

**Core Schema:**
```python
class Solution(ClassDict):
    # Identification
    v_net_id: int
    v_net_lifetime: float
    v_net_arrival_time: float
    
    # Mapping Results
    result: bool                    # Overall success/failure
    node_slots: OrderedDict         # {v_node_id: p_node_id}
    link_paths: OrderedDict         # {v_link: [p_node_path]}
    
    # Resource Metrics
    v_net_cost: float              # Total deployment cost
    v_net_revenue: float           # Expected revenue
    v_net_r2c_ratio: float         # Revenue-to-cost ratio
    
    # Constraint Satisfaction
    place_result: bool             # Node placement success
    route_result: bool             # Link routing success
    early_rejection: bool          # Pre-mapping rejection
    v_net_total_hard_constraint_violation: float
```

**Contract Requirements:**
- All resource allocation results must update these fields
- `node_slots` must contain mappings for all v_nodes if `result=True`
- `link_paths` must contain valid paths for all v_links if `result=True`

### 2. Network Data Structures

**Physical Network Interface:**
```python
class PhysicalNetwork(BaseNetwork):
    # Resource Attributes
    def get_node_attrs(self, attr_types: List[str]) -> List[Attribute]
    def get_link_attrs(self, attr_types: List[str]) -> List[Attribute]
    def get_node_attrs_data(self, attrs: List) -> np.ndarray
    def get_link_attrs_data(self, attrs: List) -> np.ndarray
    
    # Resource Aggregation
    def get_aggregation_attrs_data(self, attrs: List, aggr: str) -> np.ndarray
```

**Virtual Network Interface:**
```python
class VirtualNetwork(BaseNetwork):
    # Demand Specification
    arrival_time: float
    lifetime: float
    ranked_nodes: List[int]  # Node processing order
    
    # Same attribute interface as PhysicalNetwork
    def get_node_attrs(self, attr_types: List[str]) -> List[Attribute]
    def get_link_attrs(self, attr_types: List[str]) -> List[Attribute]
```

## Feature Constructor to Policy Network Flow

### 1. Feature Constructor Interface

**Base Contract:**
```python
class BaseFeatureConstructor:
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, 
                 config: DictConfig)
    
    def construct(self, p_net: PhysicalNetwork, v_net: VirtualNetwork,
                  solution: Solution, curr_v_node_id: int) -> Dict[str, Any]
    
    def guess_observation_space(self, p_net: PhysicalNetwork, 
                               v_net: VirtualNetwork) -> Dict[str, spaces.Space]
```

**Registry System:**
```python
@FeatureConstructorRegistry.register('custom_features')
class CustomFeatureConstructor(BaseFeatureConstructor):
    def construct(self, p_net, v_net, solution, curr_v_node_id):
        return {
            'custom_p_net_features': self._extract_p_net_features(p_net),
            'custom_v_node_features': self._extract_v_node_features(v_net, curr_v_node_id),
            'custom_context_features': self._extract_context(solution)
        }
```

### 2. Data Flow Pipeline

**Step-by-Step Process:**

```python
# 1. Feature Construction
feature_constructor = FeatureConstructorRegistry.get('p_net_v_node')(p_net, v_net, config)
obs = feature_constructor.construct(p_net, v_net, solution, curr_v_node_id)

# 2. Observation Structure
obs = {
    'p_net_x': np.ndarray,           # Shape: [num_p_nodes, p_node_features]
    'p_net_edge_index': np.ndarray,  # Shape: [2, num_p_edges]  
    'p_net_edge_attr': np.ndarray,   # Shape: [num_p_edges, p_edge_features]
    'v_node_x': np.ndarray           # Shape: [v_node_features]
}

# 3. Policy Network Processing
policy = ActorCriticRegistry.get('gcn_mlp')(config.model_params)
action_logits = policy.act(obs)      # Shape: [num_p_nodes]
value = policy.evaluate(obs)         # Shape: [1]

# 4. Action Selection
action_probs = torch.softmax(action_logits, dim=-1)
action = torch.multinomial(action_probs, 1).item()
```

### 3. Feature Construction Details

**Physical Network Features:**
```python
def _construct_p_net_features(self, p_net, v_net, solution, curr_v_node_id):
    # Node attributes (resources, capacity)
    p_node_attrs = self.obs_handler.get_node_attrs_obs(
        p_net, 
        node_attr_types=self.extracted_attr_types,
        node_attr_benchmarks=self.node_attr_benchmarks
    )
    
    # Node status (selected/available)
    p_nodes_status = self.obs_handler.get_p_net_nodes_status(
        p_net, v_net, solution['node_slots'], curr_v_node_id
    )
    
    # Topological metrics (degree, centrality)
    avg_distance = self.obs_handler.get_average_distance(
        p_net, solution['node_slots'], normalization=True
    )
    
    # Combine features
    node_data = np.concatenate([p_node_attrs, p_nodes_status, avg_distance], axis=-1)
    
    return {
        'x': node_data,
        'edge_index': self.obs_handler.get_link_index_obs(p_net),
        'edge_attr': self.obs_handler.get_link_attrs_obs(p_net, ...)
    }
```

**Virtual Node Features:**
```python
def _construct_v_node_features(self, p_net, v_net, solution, curr_v_node_id):
    # Node demand
    v_node_demand = self.obs_handler.get_v_node_demand(
        v_net, curr_v_node_id,
        node_attr_types=self.extracted_attr_types,
        node_attr_benchmarks=self.node_attr_benchmarks
    )
    
    # Link demands (aggregated)
    v_node_aggr_link_demands = self.obs_handler.get_v_node_aggr_link_demands(
        v_net, curr_v_node_id, aggr='mean',
        link_attr_types=self.extracted_attr_types
    )
    
    # Topological context
    num_neighbors = len(v_net.adj[curr_v_node_id]) / v_net.num_nodes
    
    return {
        'x': np.concatenate([v_node_demand, v_node_aggr_link_demands, [num_neighbors]])
    }
```

### 4. Policy Network Interface

**Actor-Critic Base:**
```python
class BaseActorCritic(nn.Module):
    def act(self, x: Dict[str, torch.Tensor]) -> torch.Tensor
    def evaluate(self, x: Dict[str, torch.Tensor]) -> torch.Tensor
```

**GNN-MLP Implementation:**
```python
class GnnMlpEncoder(nn.Module):
    def forward(self, p_net: Dict[str, torch.Tensor], 
                v_net_x: torch.Tensor) -> torch.Tensor:
        # Process physical network with GNN
        p_emb = self.gnn(p_net)  # [batch, num_p_nodes, embedding_dim]
        
        # Process virtual node with MLP  
        v_emb = self.mlp(v_net_x)  # [batch, embedding_dim]
        
        # Fusion: broadcast and add
        return p_emb + v_emb.unsqueeze(1).expand(-1, p_emb.shape[1], -1)
```

## Extensibility Mechanisms

### 1. Registry Pattern

**Solver Registration:**
```python
@SolverRegistry.register('my_custom_solver', solver_type='learning')
class MyCustomSolver(Solver):
    def solve(self, instance):
        # Custom algorithm implementation
        return solution
```

**Feature Constructor Registration:**
```python
@FeatureConstructorRegistry.register('graph_attention_features')
class GraphAttentionFeatureConstructor(BaseFeatureConstructor):
    def construct(self, p_net, v_net, solution, curr_v_node_id):
        # Custom feature extraction with attention mechanisms
        return custom_observation
```

**Policy Network Registration:**
```python
@ActorCriticRegistry.register('transformer_policy')
class TransformerActorCritic(BaseActorCritic):
    def __init__(self, **kwargs):
        self.transformer = TransformerEncoder(...)
    
    def act(self, obs):
        return self.transformer(obs['features'])
```

### 2. Configuration-Driven Extensibility

**Hydra Configuration:**
```yaml
# config/solver/my_solver.yaml
_target_: virne.solver.learning.rl_solver.RLSolver
solver_name: my_custom_rl
policy_type: transformer_policy
feature_constructor: graph_attention_features

model:
  embedding_dim: 256
  num_layers: 4
  attention_heads: 8

training:
  batch_size: 32
  learning_rate: 1e-4
```

### 3. Plugin Architecture

**Custom Environment:**
```python
class MultiObjectiveEnvironment(SolutionStepEnvironment):
    def compute_reward(self):
        # Multi-objective reward considering latency, cost, and reliability
        cost_reward = -self.solution['v_net_cost']
        latency_penalty = -self.calculate_latency_violation()
        reliability_bonus = self.calculate_reliability_score()
        return cost_reward + latency_penalty + reliability_bonus
```

## Module Integration Examples

### 1. Adding a New Algorithm

**Step 1: Implement Solver**
```python
@SolverRegistry.register('genetic_algorithm', solver_type='meta_heuristic')
class GeneticAlgorithmSolver(Solver):
    def solve(self, instance):
        v_net = instance['v_net']
        p_net = instance['p_net']
        
        # Initialize population
        population = self._initialize_population(v_net, p_net)
        
        # Evolution loop
        for generation in range(self.max_generations):
            # Selection, crossover, mutation
            population = self._evolve(population)
            
        # Return best solution
        return self._get_best_solution(population)
```

**Step 2: Configuration**
```yaml
# config/solver/genetic_algorithm.yaml
_target_: virne.solver.meta_heuristic.genetic_algorithm.GeneticAlgorithmSolver
solver_name: genetic_algorithm
population_size: 100
max_generations: 50
crossover_rate: 0.8
mutation_rate: 0.1
```

### 2. Custom Feature Engineering

**Graph Neural Network Features:**
```python
@FeatureConstructorRegistry.register('gnn_enhanced')
class GNNEnhancedFeatureConstructor(BaseFeatureConstructor):
    def construct(self, p_net, v_net, solution, curr_v_node_id):
        # Multi-hop neighborhood aggregation
        p_net_features = self._multi_hop_aggregation(p_net, hops=3)
        
        # Virtual network structural encoding
        v_net_encoding = self._encode_v_net_structure(v_net)
        
        # Cross-network attention
        cross_attention = self._compute_cross_attention(
            p_net_features, v_net_encoding, curr_v_node_id
        )
        
        return {
            'p_net_x': p_net_features,
            'v_net_context': v_net_encoding,
            'cross_attention': cross_attention,
            'edge_index': self.obs_handler.get_link_index_obs(p_net),
            'edge_attr': self.obs_handler.get_link_attrs_obs(p_net)
        }
```

### 3. Multi-Agent Learning

**Distributed Policy Networks:**
```python
@ActorCriticRegistry.register('multi_agent_policy')
class MultiAgentActorCritic(BaseActorCritic):
    def __init__(self, num_agents=4, **kwargs):
        super().__init__()
        self.num_agents = num_agents
        self.agents = nn.ModuleList([
            SingleAgentNetwork(**kwargs) for _ in range(num_agents)
        ])
        self.coordinator = CoordinatorNetwork(**kwargs)
    
    def act(self, obs):
        # Each agent processes part of the physical network
        agent_outputs = []
        for i, agent in enumerate(self.agents):
            agent_obs = self._partition_observation(obs, i)
            agent_outputs.append(agent(agent_obs))
        
        # Coordinator aggregates and makes final decision
        return self.coordinator(agent_outputs)
```

## API Reference

### Core Classes

#### Solver
```python
class Solver:
    """Base class for all NFV resource allocation algorithms."""
    
    def __init__(self, controller: Controller, recorder: Recorder, 
                 counter: Counter, logger: Logger, config: DictConfig)
    
    def ready(self) -> None:
        """Prepare solver for execution."""
        
    def solve(self, instance: dict) -> Solution:
        """Solve the resource allocation problem.
        
        Args:
            instance: Dict containing 'v_net' and 'p_net'
            
        Returns:
            Solution object with mapping results
        """
```

#### BaseFeatureConstructor
```python
class BaseFeatureConstructor:
    """Base class for feature extraction from networks."""
    
    def construct(self, p_net: PhysicalNetwork, v_net: VirtualNetwork,
                  solution: Solution, curr_v_node_id: int) -> Dict[str, Any]:
        """Extract features for learning algorithms.
        
        Args:
            p_net: Physical network state
            v_net: Virtual network requirements  
            solution: Current partial mapping
            curr_v_node_id: Virtual node being placed
            
        Returns:
            Dict of feature tensors for policy network
        """
```

#### BaseActorCritic
```python
class BaseActorCritic(nn.Module):
    """Base class for policy networks."""
    
    def act(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate action logits.
        
        Args:
            x: Feature dictionary from FeatureConstructor
            
        Returns:
            Action logits tensor [batch_size, num_actions]
        """
        
    def evaluate(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Estimate state value.
        
        Args:
            x: Feature dictionary from FeatureConstructor
            
        Returns:
            Value estimate tensor [batch_size, 1]
        """
```

### Registry Functions

```python
# Register new solver
@SolverRegistry.register(name: str, solver_type: str = 'unknown')

# Register new feature constructor  
@FeatureConstructorRegistry.register(name: str)

# Register new policy network
@ActorCriticRegistry.register(name: str)

# Retrieve registered components
SolverRegistry.get(name: str) -> Type[Solver]
FeatureConstructorRegistry.get(name: str) -> Type[BaseFeatureConstructor]  
ActorCriticRegistry.get(name: str) -> Type[BaseActorCritic]
```

### Configuration Integration

**Command Line Usage:**
```bash
# Use custom solver
python main.py solver.solver_name=my_custom_solver

# Use custom feature constructor
python main.py rl.feature_constructor.type=graph_attention_features

# Use custom policy network  
python main.py rl.policy.actor_critic_name=transformer_policy
```

## Complete Integration Example

To demonstrate the full modularity, here's a complete example of integrating a new custom algorithm:

### 1. Custom Graph Transformer Solver

**Implementation:**
```python
# virne/solver/learning/graph_transformer_solver.py
import torch
import torch.nn as nn
from virne.solver.learning.rl_core.rl_solver import RLSolver
from virne.solver import SolverRegistry

@SolverRegistry.register('graph_transformer', solver_type='learning')
class GraphTransformerSolver(RLSolver):
    """Custom solver using Graph Transformer for NFV resource allocation."""
    
    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        # Create custom policy factory
        def make_policy():
            return ActorCriticRegistry.get('graph_transformer')(
                p_net_num_nodes=config.p_net_setting.num_nodes,
                p_net_x_dim=self._calculate_feature_dim(config),
                **config.nn
            )
        
        super().__init__(
            controller, recorder, counter, logger, config,
            make_policy=make_policy,
            obs_as_tensor=self._convert_obs_to_tensor,
            **kwargs
        )
    
    def _calculate_feature_dim(self, config):
        # Calculate based on feature constructor configuration
        base_dim = len(config.rl.feature_constructor.extracted_attr_types)
        if config.rl.feature_constructor.if_use_node_status_flags:
            base_dim += 2
        if config.rl.feature_constructor.if_use_degree_metric:
            base_dim += 1
        return base_dim
```

### 2. Custom Graph Transformer Policy

**Policy Network:**
```python
# virne/solver/learning/rl_policy/graph_transformer_policy.py
import torch
import torch.nn as nn
from virne.solver.learning.rl_policy.base_policy import BaseActorCritic, ActorCriticRegistry

@ActorCriticRegistry.register('graph_transformer')
class GraphTransformerActorCritic(BaseActorCritic):
    def __init__(self, p_net_num_nodes, p_net_x_dim, v_node_feature_dim, 
                 num_heads=8, num_layers=6, embedding_dim=256, **kwargs):
        super().__init__()
        
        # Shared encoder
        self.encoder = GraphTransformerEncoder(
            p_net_x_dim, v_node_feature_dim, embedding_dim, 
            num_heads, num_layers
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
    
    def act(self, obs):
        encoded = self.encoder(obs)  # [batch, num_p_nodes, embedding_dim]
        logits = self.actor(encoded).squeeze(-1)  # [batch, num_p_nodes]
        return logits
    
    def evaluate(self, obs):
        encoded = self.encoder(obs)  # [batch, num_p_nodes, embedding_dim]
        values = self.critic(encoded)  # [batch, num_p_nodes, 1]
        return values.mean(dim=1)  # [batch, 1]

class GraphTransformerEncoder(nn.Module):
    def __init__(self, p_net_x_dim, v_node_x_dim, embedding_dim, num_heads, num_layers):
        super().__init__()
        
        # Input projections
        self.p_net_proj = nn.Linear(p_net_x_dim, embedding_dim)
        self.v_node_proj = nn.Linear(v_node_x_dim, embedding_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, obs):
        # Extract features
        p_net_x = obs['p_net_x']  # [batch, num_p_nodes, p_net_x_dim]
        v_node_x = obs['v_node_x']  # [batch, v_node_x_dim]
        
        # Project to embedding space
        p_emb = self.p_net_proj(p_net_x)  # [batch, num_p_nodes, embedding_dim]
        v_emb = self.v_node_proj(v_node_x)  # [batch, embedding_dim]
        
        # Add virtual node as special token
        v_emb = v_emb.unsqueeze(1)  # [batch, 1, embedding_dim]
        combined = torch.cat([v_emb, p_emb], dim=1)  # [batch, 1+num_p_nodes, embedding_dim]
        
        # Apply transformer
        encoded = self.transformer(combined)  # [batch, 1+num_p_nodes, embedding_dim]
        
        # Return only physical node embeddings
        return encoded[:, 1:, :]  # [batch, num_p_nodes, embedding_dim]
```

### 3. Custom Feature Constructor

**Advanced Feature Engineering:**
```python
# virne/solver/learning/rl_core/custom_feature_constructor.py
import torch
import numpy as np
from virne.solver.learning.rl_core.feature_constructor import BaseFeatureConstructor, FeatureConstructorRegistry

@FeatureConstructorRegistry.register('graph_transformer_features')
class GraphTransformerFeatureConstructor(BaseFeatureConstructor):
    """Enhanced feature constructor for Graph Transformer with global context."""
    
    def construct(self, p_net, v_net, solution, curr_v_node_id):
        # Base features
        p_net_obs = self._construct_p_net_features(p_net, v_net, solution, curr_v_node_id)
        v_node_obs = self._construct_v_node_features(p_net, v_net, solution, curr_v_node_id)
        
        # Global context features
        global_context = self._construct_global_context(p_net, v_net, solution)
        
        # Enhanced virtual node features with global context
        enhanced_v_node_x = np.concatenate([
            v_node_obs['x'],
            global_context
        ])
        
        return {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
            'v_node_x': enhanced_v_node_x
        }
    
    def _construct_global_context(self, p_net, v_net, solution):
        """Extract global context features."""
        # Resource utilization
        total_p_resources = sum([p_net.nodes[n]['resource'] for n in p_net.nodes])
        used_p_resources = sum([
            p_net.nodes[p_node]['resource'] 
            for p_node in solution['node_slots'].values()
        ])
        utilization = used_p_resources / total_p_resources if total_p_resources > 0 else 0
        
        # Embedding progress
        progress = len(solution['node_slots']) / v_net.num_nodes
        
        # Network size ratio
        size_ratio = v_net.num_nodes / p_net.num_nodes
        
        # Connectivity metrics
        p_net_density = nx.density(p_net)
        v_net_density = nx.density(v_net)
        
        return np.array([
            utilization, progress, size_ratio, 
            p_net_density, v_net_density
        ], dtype=np.float32)
```

### 4. Configuration Integration

**Custom Configuration File:**
```yaml
# settings/solver/graph_transformer.yaml
defaults:
  - /learning

# Override solver
solver:
  solver_name: 'graph_transformer'

# Neural network configuration
nn:
  embedding_dim: 256
  num_heads: 8
  num_layers: 6
  dropout_prob: 0.1

# Feature constructor
rl:
  feature_constructor:
    name: "graph_transformer_features"
    extracted_attr_types: ["resource", "extrema"]
    if_use_node_status_flags: true
    if_use_aggregated_link_attrs: true
    if_use_degree_metric: true
    if_use_more_topological_metrics: true

# Training parameters
training:
  learning_rate: 0.0003
  batch_size: 64
  num_train_epochs: 100
```

### 5. Usage Examples

**Command Line Usage:**
```bash
# Use the new solver
python main.py --config-name=graph_transformer

# Override specific parameters
python main.py --config-name=graph_transformer nn.embedding_dim=512 nn.num_heads=16

# Use with different network settings
python main.py --config-name=graph_transformer p_net_setting=larger_topology
```

**Programmatic Usage:**
```python
# test_custom_solver.py
import hydra
from omegaconf import DictConfig
from virne.solver import SolverRegistry
from virne.core import Environment

@hydra.main(config_path="settings", config_name="graph_transformer", version_base=None)
def test_custom_solver(config: DictConfig):
    # Create environment
    env = Environment(config)
    
    # Get registered solver
    solver_class = SolverRegistry.get('graph_transformer')
    solver = solver_class(
        controller=env.controller,
        recorder=env.recorder, 
        counter=env.counter,
        logger=env.logger,
        config=config
    )
    
    # Train the solver
    solver.learn()
    
    # Test performance
    results = solver.evaluate()
    print(f"Results: {results}")

if __name__ == "__main__":
    test_custom_solver()
```

## Verification and Testing

### 1. Component Testing

**Test Feature Constructor:**
```python
# tests/test_feature_constructor.py
import pytest
import numpy as np
from virne.solver.learning.rl_core.feature_constructor import FeatureConstructorRegistry
from virne.network import PhysicalNetwork, VirtualNetwork
from virne.core import Solution

def test_graph_transformer_features():
    # Setup test networks
    p_net = PhysicalNetwork.generate_topology(num_nodes=20)
    v_net = VirtualNetwork.generate_topology(num_nodes=5)
    solution = Solution.from_v_net(v_net)
    
    # Test feature constructor
    constructor = FeatureConstructorRegistry.get('graph_transformer_features')(
        p_net, v_net, config=test_config
    )
    
    obs = constructor.construct(p_net, v_net, solution, curr_v_node_id=0)
    
    # Verify observation structure
    assert 'p_net_x' in obs
    assert 'v_node_x' in obs
    assert obs['p_net_x'].shape[0] == p_net.num_nodes
    assert len(obs['v_node_x']) > 0
```

**Test Policy Network:**
```python
# tests/test_policy_network.py
import torch
from virne.solver.learning.rl_policy.base_policy import ActorCriticRegistry

def test_graph_transformer_policy():
    policy = ActorCriticRegistry.get('graph_transformer')(
        p_net_num_nodes=20,
        p_net_x_dim=10,
        v_node_feature_dim=15,
        embedding_dim=64
    )
    
    # Test forward pass
    obs = {
        'p_net_x': torch.randn(2, 20, 10),  # batch_size=2
        'v_node_x': torch.randn(2, 15)
    }
    
    logits = policy.act(obs)
    values = policy.evaluate(obs)
    
    assert logits.shape == (2, 20)
    assert values.shape == (2, 1)
```

### 2. Integration Testing

**End-to-End Test:**
```python
# tests/test_integration.py
def test_complete_pipeline():
    """Test complete pipeline from feature construction to policy output."""
    # Setup
    config = load_test_config()
    env = create_test_environment(config)
    
    # Create solver
    solver = SolverRegistry.get('graph_transformer')(
        controller=env.controller,
        recorder=env.recorder,
        counter=env.counter,
        logger=env.logger,
        config=config
    )
    
    # Test solving process
    for i in range(10):  # Test multiple episodes
        obs = env.reset()
        done = False
        
        while not done:
            action = solver.select_action(obs)
            obs, reward, done, info = env.step(action)
            
        assert 'result' in info
        assert isinstance(info['result'], bool)
```

This comprehensive example demonstrates how Virne's modular architecture enables researchers to:

1. **Extend algorithms** by inheriting from base classes and registering with the framework
2. **Customize feature engineering** through the feature constructor interface
3. **Implement new neural architectures** via the policy network registry
4. **Configure behavior** through declarative YAML files
5. **Test components** independently and in integration

The registry pattern ensures loose coupling between components while maintaining type safety and discoverability. Configuration-driven design allows rapid experimentation without code changes. This modular architecture makes Virne highly extensible for NFV resource allocation research while maintaining consistency and reliability.

## Quick Reference Guide

### Common Extension Patterns

1. **Adding a new heuristic algorithm:**
   ```python
   @SolverRegistry.register('my_heuristic', solver_type='heuristic')
   class MyHeuristicSolver(Solver):
       def solve(self, instance): ...
   ```

2. **Adding a new RL policy:**
   ```python
   @ActorCriticRegistry.register('my_policy')
   class MyPolicy(BaseActorCritic):
       def act(self, obs): ...
       def evaluate(self, obs): ...
   ```

3. **Adding custom features:**
   ```python
   @FeatureConstructorRegistry.register('my_features')
   class MyFeatureConstructor(BaseFeatureConstructor):
       def construct(self, p_net, v_net, solution, curr_v_node_id): ...
   ```

### Configuration Templates

**Basic RL Solver:**
```yaml
solver:
  solver_name: 'my_rl_solver'
rl:
  feature_constructor:
    name: 'p_net_v_node'
  policy:
    actor_critic_name: 'gcn_mlp'
```

**Advanced Multi-Agent:**
```yaml
solver:
  solver_name: 'multi_agent_solver'
rl:
  feature_constructor:
    name: 'distributed_features'
  policy:
    actor_critic_name: 'multi_agent_policy'
    num_agents: 4
```

### Data Flow Summary

```
Configuration → Registry → Component Instantiation
     ↓
Network Data → Feature Constructor → Structured Observations
     ↓  
Observations → Policy Network → Actions/Values
     ↓
Actions → Controller → Resource Allocation
     ↓
Results → Solution → Environment Feedback
```

This documentation demonstrates that Virne's modular architecture provides clear interfaces, well-defined data contracts, and extensible design patterns that enable researchers to easily implement and experiment with new NFV resource allocation approaches while maintaining framework consistency and reliability.