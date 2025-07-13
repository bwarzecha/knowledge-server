# Repository Maintenance & Commit Guidelines

## What We Keep vs. What We Exclude

### ✅ ALWAYS INCLUDE
- **Source code**: All Python files in `src/`
- **Tests**: Complete test suite in `tests/`
- **Documentation**: All `.md` files, design docs, architecture
- **Configuration**: `pyproject.toml`, `requirements.txt`, `.env.example`
- **Test samples**: OpenAPI specifications needed for tests
- **Research prototypes**: Python scripts only (`.py` files in `prototypes/`)
- **Project setup**: `.gitignore`, Claude settings, development guidelines

### ❌ ALWAYS EXCLUDE
- **Virtual environments**: `venv/`, `env/`, `.venv`
- **Model files**: `*.gguf`, `*.bin`, `*.safetensors`, `*.h5`, `*.pt`, `*.pth`
- **Model caches**: `model_cache/`, `models/`, HuggingFace cache directories
- **Large data exports**: `*.json` files in `prototypes/` (except small samples)
- **Environment files**: `.env` (actual secrets)
- **Build artifacts**: `__pycache__/`, `*.pyc`, `dist/`, `build/`
- **IDE files**: `.vscode/`, `.idea/`, `*.swp`
- **OS files**: `.DS_Store`, `Thumbs.db`
- **Database files**: ChromaDB persistence, `*.db`, `*.sqlite`
- **Log files**: `*.log`, `logs/`
- **Temporary files**: `.cache/`, `.pytest_cache/`, `*.tmp`

## Pre-Commit Checklist

### 1. Code Quality ✨
```bash
# Activate environment
source venv/bin/activate

# Auto-format and fix linting issues with ruff
ruff format src/ tests/
ruff check src/ tests/ --fix

# Run tests (all should pass)
python -m pytest tests/ -v
```

**Quick autofix command:**
```bash
# One-liner to format and fix most issues automatically
source venv/bin/activate && ruff format src/ tests/ && ruff check src/ tests/ --fix
```

### 2. Repository Cleanup 🧹
```bash
# Check what will be committed
git status

# Review .gitignore is working
find . -name "*.gguf" -o -name "*.bin" -o -name "model_cache" | head -5
# ^ Should show files but NOT be in git status

# Check for sensitive data
grep -r "api_key\|password\|secret" src/ tests/ || echo "Clean!"

# Verify test samples are included
ls open-api-small-samples/3.0/json/openapi-workshop/ | wc -l
# ^ Should show files (not empty)
```

### 3. Staging Strategy 📦
```bash
# Use git add -A for comprehensive staging
git add -A

# Verify staging looks correct
git status --porcelain | grep -E "^(A |M )" | head -10

# Check file count is reasonable (not 10,000+ files)
git status --porcelain | wc -l
```

## Commit Message Standards

### Format Template
```
<emoji> <type>: <concise description>

<detailed explanation of what and why>

## What's Changed
- Bullet point of key changes
- Focus on user/developer impact
- Include component affected

## Technical Details (if needed)
- Implementation specifics
- Performance implications
- Breaking changes

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Commit Types & Emojis
- 🎉 `:tada:` - Initial commit, major milestones
- ✨ `:sparkles:` - New features
- 🐛 `:bug:` - Bug fixes  
- 📚 `:books:` - Documentation updates
- 🎨 `:art:` - Code formatting, structure improvements
- ⚡ `:zap:` - Performance improvements
- ✅ `:white_check_mark:` - Adding/updating tests
- 🔧 `:wrench:` - Configuration changes
- 🚀 `:rocket:` - Deployment, releases
- 🔥 `:fire:` - Removing code/files

### Examples
```bash
# Feature addition
git commit -m "✨ Add MCP server with askAPI interface

Implements the final component to expose knowledge retrieval 
through clean askAPI() interface for LLM integration.

## What's Changed
- MCP server with streaming and batch modes
- Integration with Knowledge Retriever
- Configurable LLM provider support
- Complete end-to-end pipeline

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>"

# Bug fix
git commit -m "🐛 Fix circular reference detection in GraphBuilder

Resolves infinite loops when processing OpenAPI specs with 
complex schema dependencies.

## What's Changed  
- Enhanced cycle detection algorithm
- Added depth limiting with configurable max depth
- Improved error messages for debugging

Fixes reference expansion timeout issues in large API specs.

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>"
```

## File Size Guidelines

### Acceptable Sizes
- **Source files**: < 200 lines per file (following project tenets)
- **Test files**: < 500 lines per file
- **Documentation**: Any reasonable size
- **Sample files**: < 100KB per OpenAPI spec
- **Total commit**: < 50MB for normal commits

### When Files Are Too Large
- **Split large modules**: Break into smaller, focused components
- **Move large samples**: To external storage, reference by URL
- **Compress archives**: For necessary large files, use git-lfs
- **Separate commits**: For large documentation updates

## Regular Maintenance Tasks

### Weekly
- [ ] Run full test suite
- [ ] Check for unused dependencies
- [ ] Review and clean temporary files
- [ ] Update documentation if needed

### Before Major Features
- [ ] Update ARCHITECTURE.md implementation status
- [ ] Run performance benchmarks
- [ ] Check disk usage (`du -sh .git/`)
- [ ] Verify all tests pass

### Before Releases
- [ ] Update version in pyproject.toml
- [ ] Run comprehensive test suite
- [ ] Update success metrics in ARCHITECTURE.md
- [ ] Tag release with semantic versioning

## Emergency Cleanup Commands

```bash
# Remove large files from git history (DANGEROUS - use carefully)
git filter-branch --tree-filter 'rm -rf models/ model_cache/' HEAD

# Reset to clean state (loses uncommitted changes)
git reset --hard HEAD
git clean -fdx

# Check repository size
du -sh .git/
git count-objects -vH

# Find largest files in git
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | sort -k2 -nr | head -10
```

## Automation Ideas

Consider creating git hooks:
- **pre-commit**: Auto-format with ruff format/check
- **pre-push**: Run test suite  
- **commit-msg**: Validate commit message format

**Automated prep script available:**
```bash
# Use the provided script for complete preparation
./scripts/prep-commit.sh
```
This script automatically runs ruff formatting, linting with fixes, tests, and repository checks.

## Summary

**Golden Rules:**
1. **Code quality first** - Format, lint, test before commit
2. **Keep it lean** - Exclude large files, models, caches
3. **Meaningful commits** - Clear messages with context
4. **Test samples included** - Ensure tests can run
5. **Documentation current** - Update as implementation evolves

Following these guidelines ensures a clean, maintainable repository that's easy to clone, contributes to, and deploy.