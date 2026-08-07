Installation
============

.. important::

    The installation script currently targets Linux and Python 3.10. Virne
    depends on ``gym==0.22.0`` and is not yet compatible with newer Gym APIs.

Get the Source
--------------

.. code-block:: bash

    git clone https://github.com/GeminiLight/virne.git
    cd virne

Create a Conda Environment
--------------------------

.. code-block:: bash

    conda create -n virne python=3.10
    conda activate virne

Install with the Script
-----------------------

The installation script currently targets Linux with Python 3.10. It supports a CPU-only environment or CUDA 12.4. Run it from the repository root after activating the Conda environment.

.. code-block:: bash

    # CPU-only PyTorch and PyG
    bash install.sh -c cpu

    # CUDA 12.4 with PyTorch 2.6.0
    bash install.sh -c 12.4

If ``-c`` is omitted, the script detects an NVIDIA GPU and otherwise installs the CPU build.

Verify the Installation
-----------------------

Keep the ``virne`` Conda environment active and run this command from the
repository root:

.. code-block:: bash

    python -c "import virne; print(virne.__version__)"

The command should print the installed Virne version. You can then continue to
the :doc:`Quickstart <running>`.
