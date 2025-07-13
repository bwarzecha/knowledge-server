#!/bin/bash
# Commit Preparation Script
# Usage: ./scripts/prep-commit.sh

set -e  # Exit on error

echo "🚀 Preparing repository for commit..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not in a git repository"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run: python -m venv venv"
    exit 1
fi

echo "📦 Activating virtual environment..."
source venv/bin/activate

echo "🎨 Formatting code..."
echo "  - Running black..."
black src/ tests/ --quiet

echo "  - Running isort..."
isort src/ tests/ --quiet

echo "🔍 Checking code quality..."
echo "  - Running flake8..."
if ! flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,W503,F541; then
    echo "❌ Linting failed. Please fix issues above."
    exit 1
fi

echo "✅ Running tests..."
if ! python -m pytest tests/ -q; then
    echo "❌ Tests failed. Please fix issues above."
    exit 1
fi

echo "🧹 Checking repository cleanliness..."

# Check for large files that shouldn't be committed
echo "  - Checking for large files..."
large_files=$(find . -size +10M -not -path "./venv/*" -not -path "./.git/*" -not -path "./model_cache/*" -not -path "./models/*" 2>/dev/null || true)
if [ -n "$large_files" ]; then
    echo "⚠️  Large files found (>10MB):"
    echo "$large_files"
    echo "Consider excluding these in .gitignore"
fi

# Check for sensitive data patterns
echo "  - Checking for sensitive data..."
sensitive=$(grep -r -i "api_key\|password\|secret\|token.*=" src/ tests/ 2>/dev/null | grep -v "example\|test\|mock" || true)
if [ -n "$sensitive" ]; then
    echo "⚠️  Potential sensitive data found:"
    echo "$sensitive"
    echo "Please review and ensure no real secrets are committed"
fi

# Check test samples are present
echo "  - Verifying test samples..."
if [ ! -d "open-api-small-samples/3.0/json/openapi-workshop" ]; then
    echo "❌ Test samples missing. Tests may fail."
    exit 1
fi

echo "📊 Repository status:"
git status --porcelain | head -20

file_count=$(git status --porcelain | wc -l)
echo "  - Files to be committed: $file_count"

if [ "$file_count" -gt 1000 ]; then
    echo "⚠️  Large number of files to commit. Review .gitignore"
fi

echo ""
echo "✨ Repository is ready for commit!"
echo ""
echo "Next steps:"
echo "  1. Review staged changes: git status"
echo "  2. Add files: git add -A"
echo "  3. Create commit with meaningful message"
echo "  4. See docs/COMMIT_GUIDELINES.md for message format"
echo ""
echo "Example commit command:"
echo "  git commit -m \"✨ Add feature X"
echo ""
echo "  Description of what and why..."
echo ""
echo "  🤖 Generated with [Claude Code](https://claude.ai/code)"
echo ""
echo "  Co-Authored-By: Claude <noreply@anthropic.com>\""