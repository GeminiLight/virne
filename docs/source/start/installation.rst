Installation
============

.. note::

    Virne's RL environments use the ``gymnasium==1.3.0`` API. NumPy remains
    capped below 2.0 so that numerical compatibility can be validated in a
    separate, reproducible upgrade.

Get the Source
--------------

.. code-block:: bash

    git clone https://github.com/GeminiLight/virne.git
    cd virne

Create a Virtual Environment
----------------------------

.. code-block:: bash

    python3 -m venv .venv
    source .venv/bin/activate

Install with the Script
-----------------------

The script supports CPU environments on Linux and macOS, plus CUDA 12.6, 12.8,
and 13.0 on Linux. It installs PyTorch 2.11.0, PyG 2.8.0.post1, Gymnasium 1.3.0,
and Virne itself in editable mode. Matching PyG acceleration wheels are selected
for the active platform and Python ABI. Run it from the repository root after
activating a Python 3.10 or 3.11 environment.

.. code-block:: bash

    # CPU-only PyTorch and PyG
    bash install.sh -c cpu

    # CUDA 12.6; 12.8 and 13.0 are also supported
    bash install.sh -c 12.6

If ``-c`` is omitted, the script installs the CPU build. This explicit default
avoids selecting a CUDA runtime that is incompatible with the host driver.

Verify the Installation
-----------------------

Keep the virtual environment active and run:

.. code-block:: bash

    python -c "import gymnasium, torch, torch_geometric, virne; print(virne.__version__, gymnasium.__version__, torch.__version__, torch_geometric.__version__)"

The command should print the installed Virne, Gymnasium, PyTorch, and PyG
versions. You can then continue to the :doc:`Quickstart <running>`.
