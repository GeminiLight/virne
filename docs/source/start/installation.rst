Installation
============

.. note::

    Owing to the significant changes in the latest version of gym, 
    the current version of the code is not compatible with the latest version of gym.
    Please ensure that the version of gym is **0.21.0**.

Complete installation
---------------------

.. code-block:: bash

    # only cpu
    bash install.sh -c 0

    # use cuda (optional version: 10.2, 11.3)
    bash install.sh -c 11.3


Selective installation
----------------------


Necessary
~~~~~~~~~

.. code-block:: bash
    
    pip install networkx numpy pandas matplotlib pyyaml


Expansion
~~~~~~~~~

- Exact Solver or MCF Routing

.. code-block:: bash

    pip install ortools


- Deep Learning

.. code-block:: bash

    # use cuda
    conda install pytorch cudatoolkit=11.3 -c pytorch
    
    # only cpu
    conda install pytorch -c pytorch


- Reinfocement Learning

.. code-block:: bash

    pip install gym=0.21.0


- Graph Neural Network

.. code-block:: bash

    conda install pyg -c pyg -c conda-forge
