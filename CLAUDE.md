# Virtual Network Resource Allocator using Decision Trees

**Author:** Luis Antonio Momm Duarte

**Goal:** Use decision trees to select the best Virtual Network Embedding (VNE) algorithm for each request, optimizing acceptance rate and solution time.

--- original paper starts here --- 

---
We adopt two topologies, GEANT (40 nodes and 61 links)
and WX100 (100 nodes and 500 links) [Waxman, 1988], as
physical networks. See Appendix E.1 for these topologies’
descriptions. The multiple-type resources (i.e., CPU, storage,
GPU) of physical nodes and bandwidth resources of physical
links are uniformly generated within the range of [50, 100]
units. In each simulation run, we randomly generate 1000
VNRs with varying sizes ranging from 2 to 10. The virtual
nodes within each VNR are randomly interconnected with a
probability of 50%. Additionally, resource demands of each
VNR’s node and link requirements are uniformly generated
within the range of [0, 20] and [0, 50] units, respectively. The
lifetime of each VNR is exponentially distributed with an average of 500 time units. The arrival of these VNRs follows
a Poisson process with an average rate η, wherein η VNRs
are received per unit of time. In subsequent experiments, we
first train models with η = 0.001 on GEANT and η = 0.08
on WX100, due to their different capacities of physical resources. Then we manipulate the value of η to emulate network systems with different traffic throughputs and infer with
trained models to study the sensitivity of algorithms.
Implementations. During training, we first conduct metalearning in the initial 20 simulations and then focus on finetuning in the subsequent 10 simulations. We set the policy entropy threshold δ to 2. We implement neural network models
with PyTorch and decide reasonable values for hyperparameters following the guide of related studies [Huang et al., 2022;
Zhou et al., 2023; Wang et al., 2021a; He et al., 2023a;
Kingma and Ba, 2014; Joshi et al., 2022]. See Appendix E.2
for hyperparameter settings on neural networks and meta-RL.O

--- original paper ends here --- 


## Step-by-Step Implementation




### Step 1: Data Generation
- Run ~8 different ViRNE algorithms (heuristic, exact, meta-heuristic) on the two topologies above
- Vary topology parameters, resource distributions, and workload characteristics
- Collect metrics for each algorithm:
  - Acceptance rate (% of accepted requests)
  - Solution time (time to embed the virtual network)
- Save results to CSV files
- Use multiple random seeds for statistical significance

