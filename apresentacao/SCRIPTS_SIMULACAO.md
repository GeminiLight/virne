# Scripts de Simulação - VNE com XGBoost

## Resumo

Este documento lista **todos os scripts** necessários para executar as simulações de Virtual Network Embedding (VNE) em ambas as topologias (Tree e Fat-Tree) com todos os 8 algoritmos.

## Scripts por Topologia

### Tree Topology (7 scripts)

| Script | Algoritmo | Descrição |
|--------|-----------|-----------|
| `main_tree_ga.py` | GA_META | Genetic Algorithm meta-heuristic |
| `main_tree_mip.py` | MIP | Mixed Integer Programming (exact solver) |
| `main_tree_mcts.py` | MCTS | Monte Carlo Tree Search |
| `main_tree_sa.py` | SA_META | Simulated Annealing meta-heuristic |
| `main_tree_pl_rank.py` | PL_Rank | PageRank-based heuristic |
| `main_tree_rw_rank_bfs.py` | RW_Rank_BFS | Random Walk + BFS heuristic |
| `main_tree_drounding.py` | D_Round / R_Round | Deterministic/Randomized Rounding |

### Fat-Tree Topology (7 scripts)

| Script | Algoritmo | Descrição |
|--------|-----------|-----------|
| `main_fat_tree_ga.py` | GA_META | Genetic Algorithm for datacenter topology |
| `main_fat_tree_mip.py` | MIP | Mixed Integer Programming for Fat-Tree |
| `main_fat_tree_mcts.py` | MCTS | Monte Carlo Tree Search for Fat-Tree |
| `main_fat_tree_sa.py` | SA_META | Simulated Annealing for Fat-Tree |
| `main_fat_tree_pl_rank.py` | PL_Rank | PageRank heuristic for Fat-Tree |
| `main_fat_tree_rw_rank_bfs.py` | RW_Rank_BFS | Random Walk + BFS for Fat-Tree |
| `main_fat_tree_drounding.py` | D_Round / R_Round | Rounding algorithms for Fat-Tree |

## Total: 14 Scripts

- **8 Algoritmos** (D_Round e R_Round compartilham o mesmo script)
- **2 Topologias** (Tree e Fat-Tree)
- **5 Seeds** por execução (0-4)
- **200 VNRs** por simulação
- **Total: 80 simulações** (8 × 2 × 5)

## Características Especiais

### Topologia Tree
- Estrutura hierárquica tradicional
- Todos os nós podem hospedar VNs
- Configuração: `settings/p_net_setting/tree_p_net_setting.yaml`

### Topologia Fat-Tree
- Topologia de datacenter com switches e hosts
- **Switches (layer='switch')**: CPU=0 (routing-only, não hospedam VNs)
- **Hosts (layer='host')**: CPU>0 (hospedam VNs)
- Configuração: `settings/p_net_setting/fat_tree_p_net_setting.yaml`
- Função especial: `set_switches_to_routing_only()` em cada script

## Como Usar

### Executar um script individual:

```bash
# Exemplo: GA em Tree com seed 0
python main_tree_ga.py \
    p_net_setting=tree_p_net_setting \
    v_sim_setting=v_sim_setting \
    solver.solver_name=ga_meta \
    experiment.seed=0 \
    experiment.num_vnrs=200

# Exemplo: MCTS em Fat-Tree com seed 2
python main_fat_tree_mcts.py \
    p_net_setting=fat_tree_p_net_setting \
    v_sim_setting=v_sim_setting \
    solver.solver_name=mcts \
    experiment.seed=2 \
    experiment.num_vnrs=200

# Exemplo: D-Rounding em Tree com seed 1
python main_tree_drounding.py \
    p_net_setting=tree_p_net_setting \
    v_sim_setting=v_sim_setting \
    solver.solver_name=d_round \
    experiment.seed=1 \
    experiment.num_vnrs=200

# Exemplo: R-Rounding em Fat-Tree com seed 3
python main_fat_tree_drounding.py \
    p_net_setting=fat_tree_p_net_setting \
    v_sim_setting=v_sim_setting \
    solver.solver_name=r_round \
    experiment.seed=3 \
    experiment.num_vnrs=200
```

### Executar TODAS as simulações:

```bash
./run_complete_simulations.sh
```

Este script master executa automaticamente:
- 7 algoritmos × 2 topologias × 5 seeds = 70 simulações base
- D_Round e R_Round adicionam mais 10 simulações
- **Total: 80 simulações** (4-8 horas)

## Estrutura dos Scripts

Todos os scripts seguem a mesma estrutura básica:

```python
import hydra
from omegaconf import DictConfig
from virne.system import BaseSystem
from virne.utils.config import add_simulation_into_config, generate_run_id

@hydra.main(version_base=None, config_path="settings", config_name="main_XXX_YYY")
def run(config: DictConfig):
    # Configure run ID
    if config.experiment.run_id == 'auto':
        config.experiment.run_id = generate_run_id()
    add_simulation_into_config(config)

    # Create system
    system = BaseSystem.from_config(config)

    # (Fat-Tree only) Set switches to routing-only
    # system.env.p_net = set_switches_to_routing_only(system.env.p_net)

    # Run simulation
    system.run()

if __name__ == '__main__':
    run()
```

### Diferença Fat-Tree:

Scripts Fat-Tree incluem a função `set_switches_to_routing_only()`:

```python
def set_switches_to_routing_only(p_net):
    """
    Set CPU=0 for all switch nodes (layer='switch').
    This makes switches routing-only, unable to host virtual nodes.
    """
    for node_id in p_net.nodes:
        layer = p_net.nodes[node_id].get('layer', 'host')
        if layer == 'switch':
            p_net.nodes[node_id]['cpu'] = 0
            p_net.nodes[node_id]['max_cpu'] = 0
    return p_net
```

## Tempo Estimado por Algoritmo

| Algoritmo | Tempo/Seed (Tree) | Tempo/Seed (Fat-Tree) | Total (5 seeds) |
|-----------|-------------------|------------------------|-----------------|
| GA_META | 10-15 min | 12-18 min | 50-90 min |
| MIP | 5-10 min | 7-12 min | 25-60 min |
| MCTS | 2-3 min | 3-5 min | 10-25 min |
| SA_META | 1-2 min | 2-3 min | 5-15 min |
| PL_Rank | <1 min | 1-2 min | 3-10 min |
| RW_Rank_BFS | <1 min | 1-2 min | 3-10 min |
| D_Round | <1 min | 1-2 min | 3-10 min |
| R_Round | <1 min | 1-2 min | 3-10 min |

**Tempo Total Estimado: 4-8 horas**

## Dados Gerados

Cada simulação adiciona registros ao arquivo:
```
vnr_aggregated_data.csv
```

### Estrutura dos Dados:

**Features da VNR (7):**
- `v_net_num_nodes`, `v_net_num_edges`, `v_net_demand`
- `v_net_node_demand`, `v_net_link_demand`
- `v_net_lifetime`, `v_net_arrival_time`

**Features da Rede (6):**
- `p_net_available_resource`, `p_net_node_available_resource`
- `p_net_link_available_resource`, `p_net_node_resource_utilization`
- `p_net_link_resource_utilization`, `inservice_count`

**Resultados (4):**
- `algorithm` - nome do algoritmo
- `result` - True/False (aceito/rejeitado)
- `solving_time` - tempo de solução (segundos)
- `v_net_r2c_ratio` - revenue-to-cost ratio

**Identificadores (4):**
- `v_net_id`, `seed`, `event_time`, `event_type`

### Dados Esperados:

Após completar todas as simulações:
- **~16,000 registros** (200 VNRs × 8 algoritmos × 2 topologias × 5 seeds)
- **1,000-2,000 VNRs únicas** testadas com todos os 8 algoritmos

## Validação dos Dados

Após executar as simulações, validar com:

```bash
python validate_simulation_data.py --data-file vnr_aggregated_data.csv
```

Verificações realizadas:
- ✓ Todas colunas obrigatórias presentes
- ✓ 8 algoritmos encontrados
- ✓ 5 seeds presentes
- ✓ VNRs completas (testadas com todos algoritmos)
- ⚠ Taxa de NaN em colunas críticas (<50%)

## Próximos Passos

Após completar as simulações:

1. **Limpar dados duplicados:**
```bash
python -c "
import pandas as pd
df = pd.read_csv('vnr_aggregated_data.csv')
df = df[df['event_type'] == 1]
df['uid'] = df['v_net_id'].astype(str) + '_' + df['seed'].astype(str) + '_' + df['event_time'].astype(str) + '_' + df['algorithm']
df_clean = df.drop_duplicates(subset='uid', keep='first')
df_clean.to_csv('vnr_aggregated_data_clean.csv', index=False)
print(f'Limpeza: {len(df):,} → {len(df_clean):,} registros')
"
```

2. **Filtrar VNRs completas:**
```bash
python -c "
import pandas as pd
df = pd.read_csv('vnr_aggregated_data_clean.csv')
df['vnr_uid'] = df['v_net_id'].astype(str) + '_' + df['seed'].astype(str) + '_' + df['event_time'].astype(str)
vnr_counts = df.groupby('vnr_uid').size()
complete_vnrs = vnr_counts[vnr_counts == 8].index
df_complete = df[df['vnr_uid'].isin(complete_vnrs)]
df_complete.to_csv('vnr_complete_data.csv', index=False)
print(f'VNRs completas: {len(complete_vnrs):,} ({len(df_complete):,} registros)')
"
```

3. **Treinar XGBoost:**
```bash
python train_xgboost_v2.py --score-config acceptance_only --data-file vnr_complete_data.csv
```

4. **Comparar resultados:**
```bash
python compare_xgboost_v2.py --data-file vnr_complete_data.csv
```

## Localização dos Scripts

### Raiz do Projeto:
```
/Users/luismomm/PycharmProjects/virne/
├── main_tree_ga.py
├── main_tree_mip.py
├── main_tree_mcts.py
├── main_tree_sa.py
├── main_tree_pl_rank.py
├── main_tree_rw_rank_bfs.py
├── main_tree_drounding.py
├── main_fat_tree_ga.py
├── main_fat_tree_mip.py
├── main_fat_tree_mcts.py
├── main_fat_tree_sa.py
├── main_fat_tree_pl_rank.py
├── main_fat_tree_rw_rank_bfs.py
└── main_fat_tree_drounding.py
```

### Pasta de Apresentação:
```
/Users/luismomm/PycharmProjects/virne/apresentacao/algoritmos/
├── main_tree_ga.py
├── main_tree_mip.py
├── main_tree_mcts.py
├── main_tree_sa.py
├── main_tree_pl_rank.py
├── main_tree_rw_rank_bfs.py
├── main_tree_drounding.py
├── main_fat_tree_ga.py
├── main_fat_tree_mip.py
├── main_fat_tree_mcts.py
├── main_fat_tree_sa.py
├── main_fat_tree_pl_rank.py
├── main_fat_tree_rw_rank_bfs.py
└── main_fat_tree_drounding.py
```

**Nota:** Os scripts na pasta `apresentacao/algoritmos/` são cópias dos scripts da raiz para facilitar a organização e apresentação.

## Arquivos de Configuração

### Physical Network Settings:

**Tree:**
```yaml
# settings/p_net_setting/tree_p_net_setting.yaml
file_path: null
topology: tree  # hierárquica
num_nodes: 100
num_edges: 200
save_dir: dataset
node_attrs_setting:
  - name: cpu
    distribution: uniform
    low: 50
    high: 100
link_attrs_setting:
  - name: bw
    distribution: uniform
    low: 50
    high: 100
```

**Fat-Tree:**
```yaml
# settings/p_net_setting/fat_tree_p_net_setting.yaml
file_path: null
topology: fat_tree  # datacenter com switches e hosts
k: 4  # parâmetro k do Fat-Tree (16 hosts, 20 switches)
save_dir: dataset
node_attrs_setting:
  - name: cpu
    distribution: uniform
    low: 50
    high: 100
link_attrs_setting:
  - name: bw
    distribution: uniform
    low: 50
    high: 100
```

### Virtual Network Simulation Settings:

```yaml
# settings/v_sim_setting/v_sim_setting.yaml
num_v_nets: 1000  # será sobrescrito por experiment.num_vnrs
v_net_size:
  distribution: uniform
  low: 2
  high: 10
node_attrs_setting:
  - name: cpu
    distribution: uniform
    low: 0
    high: 20
link_attrs_setting:
  - name: bw
    distribution: uniform
    low: 0
    high: 50
lifetime:
  distribution: exponential
  scale: 500
arrival_rate:
  distribution: poisson
  lam: 0.08
```

## Suporte

Em caso de problemas:

1. **Verificar instalação do ViRNE:**
```bash
python -c "import virne; print(virne.__version__)"
```

2. **Ver logs detalhados:**
```bash
tail -f logs/tree_ga_seed0.log
```

3. **Testar um algoritmo individual primeiro:**
```bash
python main_tree_mcts.py \
    p_net_setting=tree_p_net_setting \
    v_sim_setting=v_sim_setting \
    solver.solver_name=mcts \
    experiment.seed=0 \
    experiment.num_vnrs=10
```

4. **Validar dados parciais:**
```bash
python validate_simulation_data.py
```

---

**Última atualização:** 2025-11-24
**Total de scripts:** 14 (7 Tree + 7 Fat-Tree)
**Total de simulações:** 80 (8 algoritmos × 2 topologias × 5 seeds)
