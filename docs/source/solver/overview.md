# Solver Registry

Virne selects solvers through `solver.solver_name`. For example:

```bash
python main.py solver.solver_name=nrm_rank
```

## How to Choose a Solver

Start with a solver that matches what you want to measure:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} `nrm_rank`
:class-card: solver-choice-card

**Fast baseline.** Deterministic and suitable for a first end-to-end run.
Use it in the {doc}`Quickstart <../start/running>`.
:::

:::{grid-item-card} `random_rank`
:class-card: solver-choice-card

**Random baseline.** Useful for sanity checks and comparison context. Set
`experiment.seed` for reproducible runs.
:::

:::{grid-item-card} `mip`
:class-card: solver-choice-card

**Exact-method baseline.** Best for small instances. It requires OR-Tools and
uses the implementation's current 10-second solve limit.
:::

:::{grid-item-card} `ppo_dual_gat+`
:class-card: solver-choice-card

**RL pipeline example.** Requires learning dependencies and training. See the
{doc}`RL Pipeline <../intro/rl-support>`.
:::

::::

For a first comparison, run `nrm_rank` and `random_rank` on the same scenario
and seed. Add `mip` only when the instance is small enough, and use
`ppo_dual_gat+` when the experiment specifically evaluates learning-based
methods.

## Complete Registry

The tables below are generated from `SolverRegistry` during the Sphinx build. They list the commands that are actually registered by the current code, including dynamically generated reinforcement-learning variants.

```{solver-registry}
```
