#!/bin/bash
# Development environment setup script for Nimbo

set -e

echo "Setting up Nimbo development environment..."

# Create virtual environment
VENV_NAME="nimbo-dev"
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment: $VENV_NAME"
    python -m venv $VENV_NAME
fi

# Activate virtual environment
source $VENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in editable mode with dev dependencies
echo "Installing Nimbo with dev dependencies..."
pip install -e ".[dev,all]"

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
else
    echo "pre-commit not found, skipping hook installation"
fi

# Run initial checks
echo "Running initial checks..."

echo "  - Running black..."
black --check src/ tests/ || echo "    (run 'black src/ tests/' to fix)"

echo "  - Running isort..."
isort --check-only src/ tests/ || echo "    (run 'isort src/ tests/' to fix)"

echo "  - Running flake8..."
flake8 src/ tests/ || echo "    (some linting issues found)"

echo "  - Running mypy..."
mypy src/nimbo || echo "    (some type issues found)"

echo "  - Running pytest..."
pytest tests/ -v --tb=short || echo "    (some tests failed)"

echo ""
echo "Setup complete! Activate with: source $VENV_NAME/bin/activate"
