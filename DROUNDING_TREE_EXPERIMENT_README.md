# D-Rounding Solver on Tree Topology with Clean Switches

This experiment runs the D-Rounding (Deterministic Rounding) solver on a tree topology with 200 VNR requests and 5 different random seeds.

## Key Features

- **Physical Network**: Tree topology with binary branching (branching_factor=2)
  - 16 host nodes (leaf nodes) with CPU resources (50-100 units)
  - ~15 switch nodes (internal nodes) with CPU=0 (routing-only, no VNF placement)
  - Bandwidth on links: 50-100 units

- **Virtual Networks**: 200 VNR requests
  - Random topology with 2-10 nodes
  - CPU demand: 0-20 units per node
  - Bandwidth demand: 0-50 units per link
  - Poisson arrival process (λ=0.04)
  - Exponential lifetime (mean=500 time units)

- **Solver**: D-Rounding (Deterministic Rounding)
  - Approximation algorithm using LP relaxation followed by deterministic rounding
  - Time limit: 10 seconds per VNR
  - Coordinated node and link mapping
  - Based on ViNEYard algorithm (Chowdhury et al., TON 2012)

- **Seeds**: 5 different random seeds (0, 1, 2, 3, 4) for statistical significance

## Algorithm Overview

D-Rounding (Deterministic Rounding) is an approximation algorithm that:

1. **LP Relaxation**: Formulates VNE as a Mixed Integer Program (MIP) and solves the LP relaxation (continuous variables instead of binary)
2. **Deterministic Rounding**: Rounds the fractional LP solution to integer values using a deterministic strategy
3. **Greedy Node Selection**: Selects physical nodes with highest LP values for mapping virtual nodes
4. **Path-based Link Mapping**: Uses k-shortest paths for routing virtual links

This approach is typically faster than exact MIP solvers while still providing good solution quality.

## Files Created

1. **Configuration Files**:
   - `settings/p_net_setting/tree_p_net_setting.yaml` - Physical network (tree topology)
   - `settings/v_sim_setting/v_sim_200_requests.yaml` - VNR simulator (200 requests)
   - `settings/main_tree_drounding.yaml` - Main config for D-Rounding experiment

2. **Experiment Scripts**:
   - `main_tree_drounding.py` - Modified main script with switch CPU=0 logic
   - `run_drounding_tree_5seeds.sh` - Bash script to run all 5 seeds
   - `run_drounding_tree_experiment.py` - Standalone Python script (alternative approach)

3. **Documentation**:
   - `DROUNDING_TREE_EXPERIMENT_README.md` - This file

## How to Run

### Option 1: Using the Bash Script (Recommended)

Run all 5 seeds sequentially:

```bash
./run_drounding_tree_5seeds.sh
```

This will automatically run the experiment with seeds 0, 1, 2, 3, and 4.

### Option 2: Using Python Main Script

Run a single experiment with a specific seed:

```bash
python main_tree_drounding.py \
    --config-name=main_tree_drounding \
    p_net_setting=tree_p_net_setting \
    v_sim_setting=v_sim_200_requests \
    solver.solver_name=d_round \
    experiment.seed=0
```

Change `experiment.seed=0` to any other seed value (1, 2, 3, 4) for different runs.

### Option 3: Using Standalone Script

Run all 5 seeds with a single Python script:

```bash
python run_drounding_tree_experiment.py
```

## Output and Metrics

Results are saved in: `virne/d_round/drounding_tree_seed_*/records/`

### Metrics Recorded

For each VNR request:
- `result`: Whether the VNR was successfully embedded (True/False)
- `v_net_cost`: Physical resources consumed
- `v_net_revenue`: Virtual resources requested
- `v_net_r2c_ratio`: Revenue-to-cost ratio
- `place_result`: Node placement success
- `route_result`: Link routing success
- `solution_time`: Time taken by D-Rounding solver

Summary metrics (aggregated):
- `acceptance_rate`: Percentage of successfully embedded VNRs
- `avg_r2c_ratio`: Average revenue-to-cost ratio
- `long_term_r2c_ratio`: Long-term revenue-to-cost ratio
- `total_revenue`: Total virtual resources requested
- `total_cost`: Total physical resources consumed
- `success_count`: Number of successful embeddings
- `place_failure_count`: Number of node placement failures
- `route_failure_count`: Number of link routing failures
- `early_rejection_count`: Number of early rejections

## Architecture

### Clean Switches (Routing-Only)

The key modification in this experiment is that **switches have CPU=0**, making them:
- Unable to host virtual network functions (VNFs)
- Only usable for routing traffic between hosts
- Mimicking real datacenter networks where switches are routing-only devices

This is implemented in the `set_switches_to_routing_only()` function in `main_tree_drounding.py`:

```python
def set_switches_to_routing_only(p_net):
    for node_id in p_net.nodes:
        layer = p_net.nodes[node_id].get('layer', 'host')
        if layer == 'switch':
            p_net.nodes[node_id]['cpu'] = 0
            p_net.nodes[node_id]['max_cpu'] = 0
    return p_net
```

### Tree Topology Structure

For 16 hosts with branching_factor=2:
```
                    Root (Switch 0)
                   /                \
            Switch 1                Switch 2
           /        \              /        \
      Switch 3   Switch 4     Switch 5   Switch 6
       /   \      /   \        /   \      /   \
    Host  Host  Host  Host  Host  Host  Host  Host
     ...
```

Total nodes: ~31 (15 switches + 16 hosts)

## Expected Runtime

- **Per VNR**: ~0.01-5 seconds (typically faster than MIP)
- **Per seed (200 VNRs)**: ~5-20 minutes (faster than MIP's 10-30 minutes)
- **Total (5 seeds)**: ~25-100 minutes (0.5-1.5 hours)

Note: D-Rounding solver has a 10-second time limit per VNR. The LP relaxation is typically much faster than exact MIP solving.

## Comparison with MIP

### D-Rounding Advantages:
- ✓ **Faster**: LP relaxation is faster than integer programming
- ✓ **Scalable**: Better performance on larger networks
- ✓ **Consistent runtime**: More predictable solving time

### D-Rounding Limitations:
- ✗ **Approximation**: Not guaranteed to find optimal solution
- ✗ **Acceptance rate**: May have lower acceptance rate than MIP
- ✗ **No optimality guarantee**: Solution quality depends on LP relaxation tightness

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'virne'"

**Solution**: Make sure you're in the project root directory and have installed the package:
```bash
cd /Users/luismomm/PycharmProjects/virne
pip install -e .
```

### Issue: "OR-Tools solver not found"

**Solution**: Install OR-Tools which includes the LP solver (Glop):
```bash
pip install ortools
```

### Issue: Experiments running too slowly

**Solution**: Reduce the number of VNRs:
- Edit `settings/v_sim_setting/v_sim_200_requests.yaml` and change `num_v_nets: 200` to a smaller number

### Issue: Low acceptance rate

**Potential causes**:
- Tree topology has limited connectivity compared to mesh topologies
- Switches with CPU=0 restrict VNF placement to leaf nodes only
- Large VNRs may not fit in the limited host resources
- LP relaxation may not provide tight bounds for this topology

## Analysis

After running all experiments, you can analyze the results using:

```python
import pandas as pd
import glob

# Load all D-Rounding results
drounding_results = []
for seed in range(5):
    file_path = f"virne/d_round/drounding_tree_seed_{seed}/records/summary.csv"
    df = pd.read_csv(file_path)
    df['seed'] = seed
    df['solver'] = 'D-Rounding'
    drounding_results.append(df)

# Combine all results
all_drounding = pd.concat(drounding_results, ignore_index=True)

# Calculate statistics
print("D-Rounding Results:")
print(all_drounding.groupby('seed')['acceptance_rate'].mean())
print(all_drounding[['acceptance_rate', 'avg_r2c_ratio']].describe())

# Compare with MIP results (if available)
mip_results = []
for seed in range(5):
    file_path = f"virne/mip/mip_tree_seed_{seed}/records/summary.csv"
    try:
        df = pd.read_csv(file_path)
        df['seed'] = seed
        df['solver'] = 'MIP'
        mip_results.append(df)
    except FileNotFoundError:
        print(f"MIP results for seed {seed} not found")

if mip_results:
    all_mip = pd.concat(mip_results, ignore_index=True)
    all_results = pd.concat([all_drounding, all_mip], ignore_index=True)

    # Compare acceptance rates
    comparison = all_results.groupby('solver')[['acceptance_rate', 'avg_r2c_ratio']].mean()
    print("\nSolver Comparison:")
    print(comparison)
```

## Related Files

- MIP experiment: `MIP_TREE_EXPERIMENT_README.md`
- Project instructions: `CLAUDE.md`
- ViRNE source code: `virne/` directory
- D-Rounding solver implementation: `virne/solver/exact/d_rounding.py`
- Tree topology generator: `virne/network/topology/topology_generator.py`

## References

- Mosharaf Chowdhury et al. "ViNEYard: Virtual Network Embedding Algorithms With Coordinated Node and Link Mapping". IEEE/ACM Transactions on Networking (TON), 2012.

## Contact

For questions or issues, please refer to the project documentation or contact the maintainer.
