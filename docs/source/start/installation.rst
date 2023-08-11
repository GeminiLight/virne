Installation
============

.. note::

    Owing to the significant changes in the latest version of gym, 
    the current version of the code is not compatible with the latest version of gym.
    Please ensure that the version of gym is **0.22.0**.


Install with pip
----------------

.. code-block:: bash

    pip install virne


Install with script
-------------------

.. code-block:: bash

    # only cpu
    bash install.sh -c 0

    # use cuda (e.g. cuda 11.3)
    bash install.sh -c 11.3


Selective installation
----------------------


Necessary
~~~~~~~~~

.. code-block:: bash
    
    pip install numpy pandas matplotlib networkx pyyaml tqdm ortools colorama


Expansion
~~~~~~~~~

- Deep Learning

.. code-block:: bash

    # use cuda
    conda install pytorch cudatoolkit=11.3 -c pytorch
    pip install tensorboard

    # only cpu
    conda install pytorch -c pytorch
    pip install tensorboard


- Reinfocement Learning

.. code-block:: bash

    pip install gym=0.22.0


- Graph Neural Network

.. code-block:: bash

    conda install pyg -c pyg -c conda-forge
