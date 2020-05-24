# MultiScaleMNIST dataset

MNIST dataset for detection with multiple scales.

## Development

Requirements:

- Install `pre-commit` (https://pre-commit.com/#install)
- Install `poetry` (https://python-poetry.org/docs/#installation)
- Execute `pre-commit install`
- Use `poetry` to handle requirements
  - Execute `poetry add <package_name>` to add new library
  - Execute `poetry install` to create virtualenv and install packages

## Usage

- Install package with poetry `poetry install`
- Enter the shell `poetry shell`
- Adjust settings by modifying `config.py` or passing config file (see
  `multiscalemnist --help` for info)
- Run generator `multiscalemnist generate`
