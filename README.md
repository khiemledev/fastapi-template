# eKYC Core API

![Pre-commit](https://github.com/khiemledev/golang-api-template/actions/workflows/pre-commit.yaml/badge.svg)


## Quick start

Prerequisites:

- Python >= 3.12.2

Install poetry to manage packages:

```bash
pip install poetry
```


Using poetry to install packages:

```bash
poetry install
```


Start the app:

```bash
uvicorn main:app --host 127.0.0.1 --worker 1 --port 8080
```

## Before you commit

Ensure style check by using pre-commit:

```bash
pre-commit install
```

Then, you can commit your code now.
