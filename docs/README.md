# Documentation

Run all commands from the repository root.

## Install dependencies

When you need a runnable Virne environment, install the runtime dependencies using the [installation guide](source/start/installation.rst), then register the checkout in editable mode without changing the selected CPU/CUDA packages:

```shell
python -m pip install --no-deps --editable .
```

Then install the documentation toolchain:

```shell
python -m pip install -r docs/requirements.txt
```

For a documentation-only preview, installing `docs/requirements.txt` is sufficient. Sphinx mocks optional machine-learning and optimization packages; NumPy and PyYAML remain real because Virne uses them while importing the registry and configuration utilities.

Read the Docs performs the project and documentation installation automatically through `.readthedocs.yaml`.

## Build HTML

```shell
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

## Locally Serve

```shell
python -m http.server 8000 --bind 127.0.0.1 --directory ./docs/build/html
```
