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

Install with the script
-----------------------

The installation script currently targets Linux with Python 3.10. It supports a CPU-only environment or CUDA 12.4. Run it from the repository root after activating the Conda environment.

.. code-block:: bash

    # CPU-only PyTorch and PyG
    bash install.sh -c cpu

    # CUDA 12.4 with PyTorch 2.6.0
    bash install.sh -c 12.4

If ``-c`` is omitted, the script detects an NVIDIA GPU and otherwise installs the CPU build.
