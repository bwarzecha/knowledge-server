## Development Tips

- Always source venv before running any command

## Testing and Code Quality

### Running Tests
Tests use a fast embedding model (`sentence-transformers/all-MiniLM-L6-v2`) for speed. Most tests that were previously slow should now complete quickly.

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

### Bedrock Tests
Tests that require AWS Bedrock (LLM calls) are marked with `@pytest.mark.bedrock` and are skipped by default to keep tests fast. To run them:

```bash
# Run only bedrock tests
python -m pytest tests/ -m bedrock -v

# Run all tests including bedrock
python -m pytest tests/ -m "not slow" -v

# Run everything (including bedrock and slow tests)
python -m pytest tests/ --tb=short
```

### Test Environment Configuration
The project includes a `.env.test` file with optimized settings for testing:
- Fast embedding model: `sentence-transformers/all-MiniLM-L6-v2` (22M parameters)
- MPS device for fast GPU acceleration on Apple Silicon
- Reduced token limits for faster processing

If you need to test with production embedding models, temporarily modify the test files or use the main `.env` configuration.

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