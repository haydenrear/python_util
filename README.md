# Python Util Library

A Python utility library that has been upgraded to use modern Python packaging.

## Migration to uv

This project has been migrated from using `setup.py` and `requirements.txt` to using the modern Python packaging system with `pyproject.toml` and [uv](https://github.com/astral-sh/uv) for dependency management.

## Benefits of uv

- Faster installation of dependencies
- Improved dependency resolution
- Reproducible builds
- Better caching
- Modern Python packaging standards compliance

## Installation

### Installing with uv

```bash
uv pip install -e .
```

### Installing with pip

```bash
pip install -e .
```

## Development

### Setting up a development environment

```bash
# Install uv if you don't have it already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Dependencies

The project dependencies are now specified in the `pyproject.toml` file instead of `requirements.txt`.

To add a new dependency:

```bash
uv add package_name
```

This will automatically update your `pyproject.toml` file.

## Legacy Files

The following files are kept for backward compatibility but are no longer the primary way to manage dependencies:

- `setup.py`
- `requirements.txt`