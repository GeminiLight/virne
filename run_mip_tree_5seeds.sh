#!/bin/bash

# Experiment: Run MIP solver on tree topology with 200 VNRs and 5 different seeds
# Physical network: Tree topology with switches as routing-only (CPU=0)
# Virtual networks: 200 requests with random topology

echo "======================================================================"
echo "MIP SOLVER EXPERIMENT ON TREE TOPOLOGY WITH CLEAN SWITCHES"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  - Physical network: Tree topology (16 hosts + switches)"
echo "  - Virtual network requests: 200"
echo "  - Solver: MIP (Mixed Integer Programming)"
echo "  - Seeds: 5 different random seeds (0, 1, 2, 3, 4)"
echo "  - Switch configuration: CPU=0 (routing-only)"
echo ""
echo "======================================================================"
echo ""

# Array of seeds
seeds=(0 1 2 3 4)

# Loop through each seed and run the experiment
for seed in "${seeds[@]}"; do
    echo ""
    echo "----------------------------------------------------------------------"
    echo "Running experiment with seed $seed"
    echo "----------------------------------------------------------------------"
    echo ""

    python main_tree_mip.py \
        experiment.seed=$seed \
        experiment.run_id=mip_tree_seed_$seed

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Experiment with seed $seed completed successfully"
        echo ""
    else
        echo ""
        echo "✗ Experiment with seed $seed failed"
        echo ""
        exit 1
    fi
done

echo ""
echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
echo "======================================================================"
echo ""
echo "Results saved in: virne/mip/mip_tree_seed_*/"
echo ""