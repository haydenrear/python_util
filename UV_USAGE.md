# Using uv with python_util

This document provides examples of common tasks with [uv](https://github.com/astral-sh/uv), the fast Python package installer and resolver.

## Installation

Install uv using the official installer:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

## Basic Usage

### Installing the Package

```bash
# Install in development mode
uv pip install -e .

# Install from PyPI (when published)
uv pip install python_util
```

## Virtual Environment Management

```bash
# Create a virtual environment in the current directory
uv venv

# Activate the environment
# On Unix/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Create a virtual environment in a specific location
uv venv /path/to/venv
```

## Dependency Management

```bash
# Install a specific package
uv pip install numpy

# Install all dependencies from pyproject.toml
uv pip install .

# Add a dependency and update pyproject.toml
uv add pandas

# Add a development dependency
uv add --dev pytest

# Update a package
uv pip install --upgrade torch
```

## Dependency Synchronization

```bash
# Generate lock file
uv pip sync --generate-lock-file

# Install dependencies from lock file
uv pip sync
```

## Build and Distribution

```bash
# Build the package
uv pip build

# Build the package with specific options
uv pip build --wheel --sdist
```

## Performance Comparison

uv is significantly faster than pip for most operations:

- Installing packages: 10-100x faster
- Creating virtual environments: 3-5x faster
- Dependency resolution: 10-30x faster

## Migration from pip

If you have existing pip commands, you can usually replace `pip` with `uv pip`:

```bash
# pip command
pip install -r requirements.txt

# uv equivalent
uv pip install -r requirements.txt
```

## Best Practices

1. Use pyproject.toml instead of setup.py and requirements.txt
2. Leverage uv's synchronization capabilities for reproducible environments
3. Use .uvignore to control what files are included in builds
4. Combine with pre-commit hooks for consistent development environments

## Troubleshooting

If you encounter issues:

```bash
# Check uv version
uv --version

# Verbose output for debugging
uv pip install -v .

# Clean cache if needed
uv cache clean
```

## Additional Resources

- [Official uv Documentation](https://github.com/astral-sh/uv)
- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPA's Guide to Packaging](https://packaging.python.org/guides/distributing-packages-using-setuptools/)