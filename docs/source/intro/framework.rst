Architecture
============

Virne connects configuration, network simulation, solvers, and recording in a
single experiment workflow. This page maps the conceptual architecture to the
implementation so that readers can find the relevant code quickly.

.. image:: ../_static/virne-architecture.png
   :width: 1000
   :alt: Overall architecture of Virne

Core Components
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - Component
     - Responsibility
     - Main implementation
   * - Configuration
     - Defines the experiment, system, solver, training, PN, and VN settings.
     - ``settings/main.yaml``, ``settings/learning.yaml``, and the PN/VN files
       under ``settings/``
   * - Network model
     - Represents physical and virtual networks and generates request events.
     - ``virne.network``
   * - System
     - Builds an online, offline, time-window, or changeable simulation and
       drives its event loop.
     - ``virne.system.BaseSystem`` and its subclasses
   * - Environment and controller
     - Check placement and routing feasibility, update resources, and expose
       solver-facing state transitions.
     - ``virne.core.environment`` and ``virne.core.Controller``
   * - Solver registry
     - Resolves ``solver.solver_name`` to an exact, heuristic,
       meta-heuristic, or learning-based implementation.
     - ``virne.solver.SolverRegistry``
   * - Recorder and counter
     - Store per-event records and compute run-level metrics.
     - ``virne.core.Recorder`` and ``virne.core.Counter``

Experiment Flow
---------------

When ``python main.py`` is executed, Virne follows this sequence:

1. Hydra composes ``settings/main.yaml`` and applies command-line overrides.
2. ``BaseSystem.from_config`` creates the network data, environment, solver,
   controller, recorder, counter, and logger.
3. The selected system emits VN arrival and departure events.
4. For each arrival, the selected solver returns a mapping solution.
5. The environment checks the solution, updates resources, and records the
   outcome.
6. Virne writes the resolved configuration, per-event records, and summary
   metrics to the run directory.

This separation lets the same solver run against different simulations and
lets multiple solvers share the same feasibility checks and metric
definitions.

Where to Go Next
----------------

* Run the complete workflow in the :doc:`5-minute quickstart
  <../start/running>`.
* Configure PN and VN generation in :doc:`simulation scenarios
  <../start/simulation>`.
* Browse the live :doc:`solver registry <../solver/overview>`.
* Learn how training fits into the system in :doc:`RL Pipeline <rl-support>`.
* Interpret generated CSV files with :doc:`evaluation metrics
  <../evaluation/metrics>`.
