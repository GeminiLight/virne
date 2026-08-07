Evaluation Protocols
====================

Virne evaluates NFV-RA algorithms from three practicality perspectives:
**solvability**, **generalization**, and **scalability**. A useful experiment
must turn each perspective into a controlled comparison rather than changing
several settings at once.

Shared Experimental Rules
-------------------------

Apply these rules to every protocol:

* Change **one evaluation axis at a time**. Keep the solver, PN/VN settings,
  and decoding strategy fixed unless they are the variable being studied.
* Use at least five independent evaluation seeds, for example
  ``experiment.seed=0`` through ``experiment.seed=4``. Report the mean and
  standard deviation across seeds.
* Run separate seed values explicitly. ``experiment.num_simulations`` repeats
  the simulation loop but does not create a new seed for each repetition.
* For RL methods, train once and reuse the same checkpoint across the test
  conditions. Set ``training.num_train_epochs=0`` during evaluation.
* Keep each generated ``config.yaml`` together with ``summary.csv`` and the
  per-event records. The resolved configuration is part of the result.

.. list-table:: Protocol at a glance
   :header-rows: 1
   :widths: 19 28 27 26
   :class: evaluation-protocols

   * - Perspective
     - Change
     - Keep fixed
     - Report
   * - Solvability
     - VN size or demand
     - PN, load, solver, seeds
     - Acceptance, failures, R2C
   * - Generalization
     - One unseen condition
     - Checkpoint and inference settings
     - Absolute and relative metric change
   * - Scalability
     - PN/VN size or request count
     - Topology, workload, solver
     - Quality, runtime, peak memory

Solvability
-----------

**Question:** How often can the solver find a feasible mapping as individual
instances become harder?

Vary one source of difficulty, such as ``v_sim_setting.v_net_size.high``, while
keeping the PN and resource distributions fixed. To reduce failures caused by
competition between simultaneous requests, use a low arrival rate and short
lifetime. The following is a low-contention proxy for an independent-instance
study:

.. code-block:: bash

   for max_v_nodes in 4 6 8 10; do
     for eval_seed in 0 1 2 3 4; do
       python main.py \
         solver.solver_name=nrm_rank \
         v_sim_setting.num_v_nets=100 \
         v_sim_setting.v_net_size.high=${max_v_nodes} \
         v_sim_setting.arrival_rate.rate=0.001 \
         v_sim_setting.lifetime.scale=10 \
         experiment.seed=${eval_seed} \
         experiment.run_id=solvability-v${max_v_nodes}-s${eval_seed}
     done
   done

Report ``acceptance_rate``, ``place_failure_count``, ``route_failure_count``,
and the relevant R2C field. The current CLI does not expose a dedicated batch
of fully independent static instances, so describe this low-contention setup
as a proxy rather than an exact offline solvability test.

Generalization
--------------

**Question:** Does a fixed policy remain effective when the evaluation
distribution differs from the training distribution?

Start with an in-distribution test set, then change exactly one axis:

* **Traffic load:** ``v_sim_setting.arrival_rate.rate``
* **Request size:** ``v_sim_setting.v_net_size.low`` and ``high``
* **Physical topology:** ``+p_net_setting.topology.file_path=...``
* **Physical scale:** ``p_net_setting.topology.num_nodes``

For an RL solver, load the same checkpoint for every condition:

.. code-block:: bash

   python main.py \
     solver.solver_name=ppo_dual_gat+ \
     training.num_train_epochs=0 \
     solver.pretrained_model_path=/absolute/path/to/model.pkl \
     v_sim_setting.arrival_rate.rate=0.08 \
     experiment.seed=1 \
     experiment.run_id=generalization-rate008-s1

Do not retrain or tune the checkpoint on the shifted test condition. Report
the absolute metrics and their relative change from the in-distribution
baseline, using the same seed set for both conditions.

Scalability
-----------

**Question:** How do solution quality and resource requirements change as the
problem grows?

Choose one scale axis, such as PN nodes, maximum VN nodes, or number of VN
requests. For algorithmic scaling, keep the number and distribution of VN
requests fixed while increasing PN size. On Linux, ``/usr/bin/time -v`` can
also record peak memory:

.. code-block:: bash

   for p_nodes in 50 100 200; do
     /usr/bin/time -v python main.py \
       solver.solver_name=nrm_rank \
       p_net_setting.topology.num_nodes=${p_nodes} \
       v_sim_setting.num_v_nets=100 \
       experiment.seed=0 \
       experiment.run_id=scalability-p${p_nodes}-s0
   done

Repeat the sweep for every evaluation seed. Report acceptance and R2C metrics,
``clock_running_time``, and the maximum resident set size reported by
``/usr/bin/time -v``. As explained in :doc:`metrics`,
``clock_running_time`` is end-to-end runtime and is not solver-only Average
Solving Time (AST).

Reporting Checklist
-------------------

For each figure or table, state:

* the exact solver command or checkpoint;
* the controlled variable and all tested values;
* the PN/VN configuration and evaluation seeds;
* the number of requests per run;
* mean and standard deviation for every reported metric;
* whether runtime is solver-only or end-to-end; and
* any timeout, failed run, or excluded result.

Use :doc:`metrics` to map benchmark terminology to the actual CSV fields.
