API Reference
=============

This reference is generated from Virne's public classes and docstrings. Optional machine-learning and optimization dependencies are mocked while building the documentation, so importing the API does not install packages or require a GPU.

Network Models and Generators
-----------------------------

.. autoclass:: virne.network.base_network.BaseNetwork
   :members:
   :show-inheritance:

.. autoclass:: virne.network.physical_network.PhysicalNetwork
   :members:
   :show-inheritance:

.. autoclass:: virne.network.virtual_network.VirtualNetwork
   :members:
   :show-inheritance:

.. autoclass:: virne.network.virtual_network_request_simulator.VirtualNetworkRequestSimulator
   :members:
   :show-inheritance:

.. autoclass:: virne.network.dataset_generator.Generator
   :members:

.. autoclass:: virne.network.attribute.BaseAttribute
   :members:
   :show-inheritance:

.. autoclass:: virne.network.attribute.AttributeBenchmarkManager
   :members:

.. autoclass:: virne.network.attribute.AttributeBenchmarks
   :members:

.. autoclass:: virne.network.topology.TopologyGenerator
   :members:

.. autoclass:: virne.network.topology.TopologicalMetricCalculator
   :members:

.. autoclass:: virne.network.topology.TopologicalMetrics
   :members:

Core Simulation Objects
-----------------------

.. autoclass:: virne.core.environment.BaseEnvironment
   :members:
   :show-inheritance:

.. autoclass:: virne.core.environment.SolutionStepEnvironment
   :members:
   :show-inheritance:

.. autoclass:: virne.core.environment.JointPRStepEnvironment
   :members:
   :show-inheritance:

.. autoclass:: virne.core.recorder.Recorder
   :members:
   :show-inheritance:

.. autoclass:: virne.core.solution.Solution
   :members:

.. autoclass:: virne.core.counter.Counter
   :members:

.. autoclass:: virne.core.controller.Controller
   :members:

.. autoclass:: virne.core.logger.Logger
   :members:

Network System
--------------

.. autoclass:: virne.system.base_system.BaseSystem
   :members:
   :show-inheritance:

Solver Infrastructure
---------------------

.. automodule:: virne.solver.base_solver
   :members:
   :show-inheritance:
