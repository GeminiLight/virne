Running
=======

1. Run the default example
--------------------------

Before running the example, you could update the configuration file in ``settings/`` directory to set the parameters on simulation and algorithm.

.. code-block:: bash

   python main.py

2. Run with custom configuration
--------------------------------

Virne is built on `Hydra <https://hydra.cc/>`_, which allows you to override configuration parameters directly from the command line.

.. code-block:: bash

   python main.py CONFIG_NAME=NEW_VALUE

Some examples of command line arguments are:

.. code-block:: bash

   # Run with a specific nfv-ra algorithm
   python main.py solver.solver_name=nrm_rank

   # Run with a specific physical topology
   python main.py p_net_setting.topology.file_path=../../datasets/topology/Geant.gml

   # Run with a specific network system
   python main.py system.if_offline_system=true