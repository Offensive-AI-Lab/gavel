# Contributing to GAVEL

Thank you for your interest in contributing to GAVEL! This document provides guidelines and information for contributors.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Offensive-AI-Lab/gavel.git
   cd gavel
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or: .venv\Scripts\activate  # Windows
   ```

3. **Install with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

We use the following tools to maintain code quality:

- **[Ruff](https://github.com/astral-sh/ruff)** for linting
- **[Black](https://github.com/psf/black)** for code formatting

### Running Linters

```bash
# Check for issues
ruff check gavel/ scripts/

# Auto-fix issues
ruff check --fix gavel/ scripts/

# Format code
black gavel/ scripts/
```

## Testing

We use **pytest** for testing. All tests are in the `tests/` directory.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=gavel --cov-report=term-missing

# Run specific test file
pytest tests/test_config.py -v
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_<module>.py`
- Use descriptive test function names: `test_<function>_<scenario>`
- Add fixtures to `tests/conftest.py` for reusability

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Write tests** for any new functionality
3. **Ensure all tests pass** locally
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] `ruff check` passes
- [ ] All tests pass
- [ ] Commit messages are clear

## Reporting Issues

When reporting bugs, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/traceback

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive environment.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
