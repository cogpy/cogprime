#!/bin/bash
#
# Enhanced batch issue creation script with multiple fallback methods
#
# This script attempts to create all 15 CogPrime development issues using
# the best available method:
# 1. GitHub CLI (gh) if authenticated
# 2. Python script with GITHUB_TOKEN environment variable
# 3. Provides instructions for manual creation
#
# Prerequisites (try in order):
#   - Option 1: GitHub CLI (gh) installed and authenticated
#   - Option 2: Python 3 with requests/PyYAML + GITHUB_TOKEN environment variable
#   - Option 3: Manual creation via GitHub web UI or GitHub Actions
#
# Usage:
#   bash create_all_issues_enhanced.sh
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
REPO="cogpy/cogprime"

echo "=================================================="
echo "CogPrime Development Issues Creator"
echo "Repository: $REPO"
echo "=================================================="
echo

# Check Method 1: GitHub CLI
if command -v gh &> /dev/null; then
    if gh auth status &> /dev/null; then
        echo "✓ GitHub CLI detected and authenticated"
        echo "  Using Method 1: GitHub CLI (gh)"
        echo
        exec "$SCRIPT_DIR/create_all_issues.sh"
    else
        echo "⚠ GitHub CLI detected but not authenticated"
    fi
else
    echo "⚠ GitHub CLI (gh) not installed"
fi

# Check Method 2: Python script with GitHub token
if command -v python3 &> /dev/null || command -v python &> /dev/null; then
    PYTHON_CMD=$(command -v python3 || command -v python)
    
    if [ -n "$GITHUB_TOKEN" ]; then
        echo "✓ Python detected with GITHUB_TOKEN environment variable"
        echo "  Using Method 2: Python script with GitHub API"
        echo
        
        # Check for required Python packages
        if $PYTHON_CMD -c "import requests, yaml" &> /dev/null; then
            echo "  Required Python packages (requests, PyYAML) are installed"
        else
            echo "  Installing required Python packages..."
            pip install PyYAML requests || {
                echo "  ⚠ Failed to install required packages"
                echo "  Please run: pip install PyYAML requests"
                exit 1
            }
        fi
        
        # Run the Python script
        $PYTHON_CMD "$REPO_ROOT/.github/scripts/generate_issues.py" \
            --phase all \
            --dry-run False \
            --token "$GITHUB_TOKEN" \
            --repo "$REPO"
        
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo
            echo "✅ All issues created successfully via Python script!"
        fi
        exit $exit_code
    else
        echo "⚠ Python detected but GITHUB_TOKEN not set"
    fi
else
    echo "⚠ Python not found"
fi

# If we get here, no automated method is available
echo
echo "=================================================="
echo "❌ Unable to create issues automatically"
echo "=================================================="
echo
echo "No automated method is available. Please use one of these alternatives:"
echo
echo "Method 1: GitHub CLI"
echo "  1. Install GitHub CLI: https://cli.github.com/"
echo "  2. Authenticate: gh auth login"
echo "  3. Run: bash $SCRIPT_DIR/create_all_issues.sh"
echo
echo "Method 2: Python Script"
echo "  1. Install dependencies: pip install PyYAML requests"
echo "  2. Get a GitHub token: https://github.com/settings/tokens"
echo "  3. Set token: export GITHUB_TOKEN='your_token_here'"
echo "  4. Run this script again"
echo
echo "Method 3: GitHub Actions (Recommended)"
echo "  1. Go to: https://github.com/$REPO/actions"
echo "  2. Select 'Generate Development Issues' workflow"
echo "  3. Click 'Run workflow'"
echo "  4. Set phase to 'all' and dry_run to 'false'"
echo "  5. Click 'Run workflow'"
echo
echo "Method 4: Manual Creation"
echo "  1. Go to: https://github.com/$REPO/issues/new"
echo "  2. Copy content from markdown files in $SCRIPT_DIR"
echo "  3. Create each issue manually (15 total)"
echo
echo "For detailed instructions, see:"
echo "  - $REPO_ROOT/RUNNING_ISSUE_CREATION.md"
echo "  - $SCRIPT_DIR/README.md"
echo "  - $REPO_ROOT/CREATE_ISSUES.md"
echo
echo "=================================================="

exit 1
