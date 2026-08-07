5-minute Quickstart
===================

This example runs a small CPU simulation with a fast heuristic solver. It is a
better first check than ``python main.py`` alone, because the default
configuration trains an RL solver and processes 1,000 VN requests.

Run a Small Experiment
----------------------

From the repository root, with the ``virne`` Conda environment active, run:

.. code-block:: bash

   python main.py \
     solver.solver_name=nrm_rank \
     v_sim_setting.num_v_nets=10 \
     training.use_cuda=false \
     'logger.backends=[console]'

Virne generates a physical network and ten virtual requests, solves each
arrival event, and prints a summary. A successful run ends with:

.. code-block:: text

   --------------------   Complete   --------------------

The progress bar also shows the running acceptance rate (``ac``),
revenue-to-cost ratio (``r2c``), and number of in-service requests.

Inspect the Results
-------------------

Hydra places the run under ``results/``. The most useful files are:

.. code-block:: text

   results/virne/nrm_rank/<run-id>/config.yaml
   results/virne/nrm_rank/<run-id>/summary.csv
   results/virne/nrm_rank/<run-id>/records/*.csv

``config.yaml`` is the fully resolved experiment configuration,
``summary.csv`` contains one row of run-level metrics, and ``records/*.csv``
contains per-event details. See :doc:`../evaluation/metrics` for the field
definitions.

Override the Configuration
--------------------------

Virne uses `Hydra <https://hydra.cc/>`_, so any existing configuration value
can be overridden with ``key=value``. Prefix a new key with ``+``.

.. code-block:: bash

   # Select another registered solver
   python main.py solver.solver_name=random_rank

   # Load a physical topology from a GML file
   python main.py +p_net_setting.topology.file_path=./datasets/topology/Geant.gml

   # Use the offline network system
   python main.py system.if_offline_system=true

   # Preview the resolved configuration without running a simulation
   python main.py --cfg job

Use the :doc:`solver registry <../solver/overview>` for valid solver names and
:doc:`simulation scenarios <simulation>` for the canonical PN and VN settings.
