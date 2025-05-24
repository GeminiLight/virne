Installation
============

.. note::

    Owing to the significant changes in the latest version of gym, 
    the current version of the code is not compatible with the latest version of gym.
    Please ensure that the version of gym is **0.22.0**.


Create a new conda environment
------------------------------

.. code-block:: bash

    conda create -n virne python=3.10
    conda activate virne

Install with script
-------------------

.. code-block:: bash

    # use cpu
    bash install.sh -c 0

    # use cuda (only support cuda=12.4 and torch=2.6.0)
    bash install.sh -c 12.4