# Usage

## Build Project

```shell
sphinx-build -M html docs/source/ docs/build/
```

## Locally Serve

```shell
python -m http.server 8000 --bind 127.0.0.1 --directory ./docs/build/html
```