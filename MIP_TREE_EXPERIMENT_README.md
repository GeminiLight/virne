# MIP Solver on Tree Topology with Clean Switches

This experiment runs the MIP (Mixed Integer Programming) solver on a tree topology with 200 VNR requests and 5 different random seeds.

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

- **Solver**: MIP (Mixed Integer Programming) using SCIP
  - Time limit: 10 seconds per VNR
  - Coordinated node and link mapping

- **Seeds**: 5 different random seeds (0, 1, 2, 3, 4) for statistical significance

## Files Created

1. **Configuration Files**:
   - `settings/p_net_setting/tree_p_net_setting.yaml` - Physical network (tree topology)
   - `settings/v_sim_setting/v_sim_200_requests.yaml` - VNR simulator (200 requests)

2. **Experiment Scripts**:
   - `main_tree_mip.py` - Modified main script with switch CPU=0 logic
   - `run_mip_tree_5seeds.sh` - Bash script to run all 5 seeds
   - `run_mip_tree_experiment.py` - Standalone Python script (alternative approach)

3. **Documentation**:
   - `MIP_TREE_EXPERIMENT_README.md` - This file

## How to Run

### Option 1: Using the Bash Script (Recommended)

Run all 5 seeds sequentially:

```bash
./run_mip_tree_5seeds.sh
```

This will automatically run the experiment with seeds 0, 1, 2, 3, and 4.

### Option 2: Using Python Main Script

Run a single experiment with a specific seed:

```bash
python main_tree_mip.py \
    p_net_setting=tree_p_net_setting \
    v_sim_setting=v_sim_200_requests \
    solver.solver_name=mip \
    experiment.seed=0
```

Change `experiment.seed=0` to any other seed value (1, 2, 3, 4) for different runs.

### Option 3: Using Standalone Script

Run all 5 seeds with a single Python script:

```bash
python run_mip_tree_experiment.py
```

## Output and Metrics

Results are saved in: `virne/mip/mip_tree_seed_*/records/`

### Metrics Recorded

For each VNR request:
- `result`: Whether the VNR was successfully embedded (True/False)
- `v_net_cost`: Physical resources consumed
- `v_net_revenue`: Virtual resources requested
- `v_net_r2c_ratio`: Revenue-to-cost ratio
- `place_result`: Node placement success
- `route_result`: Link routing success
- `solution_time`: Time taken by MIP solver

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

This is implemented in the `set_switches_to_routing_only()` function in `main_tree_mip.py`:

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

- **Per VNR**: ~0.1-10 seconds (depending on VNR size and complexity)
- **Per seed (200 VNRs)**: ~10-30 minutes
- **Total (5 seeds)**: ~50-150 minutes (1-2.5 hours)

Note: MIP solver has a 10-second time limit per VNR. If optimal solution is not found within this time, the best feasible solution is returned.

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'virne'"

**Solution**: Make sure you're in the project root directory and have installed the package:
```bash
cd /Users/luismomm/PycharmProjects/virne
pip install -e .
```

### Issue: "SCIP solver not found"

**Solution**: Install OR-Tools which includes SCIP:
```bash
pip install ortools
```

### Issue: Experiments running too slowly

**Solution**: Reduce the number of VNRs or increase the MIP time limit:
- Edit `settings/v_sim_setting/v_sim_200_requests.yaml` and change `num_v_nets: 200` to a smaller number
- Or edit the MIP solver timeout in the solver code

## Analysis

After running all experiments, you can analyze the results using:

```python
import pandas as pd
import glob

# Load all results
results = []
for seed in range(5):
    file_path = f"virne/mip/mip_tree_seed_{seed}/records/summary.csv"
    df = pd.read_csv(file_path)
    df['seed'] = seed
    results.append(df)

# Combine all results
all_results = pd.concat(results, ignore_index=True)

# Calculate statistics
print(all_results.groupby('seed')['acceptance_rate'].mean())
print(all_results[['acceptance_rate', 'avg_r2c_ratio']].describe())
```

## Related Files

- Original paper: `2404.12633v4.pdf` (if available)
- Project instructions: `CLAUDE.md`
- ViRNE source code: `virne/` directory
- MIP solver implementation: `virne/solver/exact/mip.py`
- Tree topology generator: `virne/network/topology/topology_generator.py`

## Contact

For questions or issues, please refer to the project documentation or contact the maintainer.