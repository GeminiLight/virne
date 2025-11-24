#!/bin/bash

# Script to run all presentation simulations in background
# Runs all algorithms on both tree and fat-tree topologies with seeds 0-4

export PYTHONPATH=/Users/luismomm/PycharmProjects/virne

echo "Starting all presentation simulations..."
echo "Results will be saved in: apresentacao/simulacoes/"
echo ""

# Array of algorithms and topologies
ALGORITHMS=("ga" "mip" "mcts" "sa" "pl_rank" "rw_rank_bfs" "drounding")
TOPOLOGIES=("tree" "fat_tree")
SEEDS=(0 1 2 3 4)

# Counter for tracking
TOTAL=0
STARTED=0

# Calculate total simulations
for algo in "${ALGORITHMS[@]}"; do
    for topo in "${TOPOLOGIES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            ((TOTAL++))
        done
    done
done

echo "Total simulations to run: $TOTAL"
echo ""

# Run all simulations
for algo in "${ALGORITHMS[@]}"; do
    for topo in "${TOPOLOGIES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            SCRIPT="apresentacao/algoritmos/main_${topo}_${algo}.py"

            if [ -f "$SCRIPT" ]; then
                ((STARTED++))
                echo "[$STARTED/$TOTAL] Starting: ${topo} - ${algo} - seed ${seed}"

                nohup python "$SCRIPT" experiment.seed=${seed} \
                    > "apresentacao/simulacoes/logs/${topo}_${algo}_seed${seed}.log" 2>&1 &

                PID=$!
                echo "  -> PID: $PID"

                # Small delay to avoid overwhelming the system
                sleep 2
            else
                echo "[$STARTED/$TOTAL] SKIP: Script not found: $SCRIPT"
            fi
        done
    done
done

echo ""
echo "All simulations started!"
echo "Monitor progress with: tail -f apresentacao/simulacoes/logs/*.log"
echo "Check running processes: ps aux | grep python | grep apresentacao"
echo ""
