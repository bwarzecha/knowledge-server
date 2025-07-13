## Development Tips

- Always source venv before running any command

## Testing and Code Quality

### Running Tests
```bash
source venv/bin/activate
python -m pytest tests/ -v
```

### Code Formatting
The project uses black (line length 100), isort, and flake8 for code formatting and linting:

```bash
source venv/bin/activate
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

### Pre-commit Cleanup
Before committing, run the complete cleanup:
```bash
source venv/bin/activate
black src/ tests/
isort src/ tests/
python -m pytest tests/
```