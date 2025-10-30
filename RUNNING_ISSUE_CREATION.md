# Running the Issue Creation Script

## Overview

The `issues_to_create/create_all_issues.sh` script is ready to create 15 comprehensive development issues for the CogPrime roadmap. All necessary files are in place and the script is executable.

## Current Status

✅ Script exists at: `issues_to_create/create_all_issues.sh`
✅ All 15 markdown issue templates are present
✅ Script syntax is valid
✅ Script has execute permissions (`-rwxrwxr-x`)
✅ GitHub CLI (`gh`) is installed

## How to Run the Script

### Option 1: Direct Execution (Requires GitHub CLI Authentication)

If you have GitHub CLI authenticated:

```bash
cd /home/runner/work/cogprime/cogprime
bash issues_to_create/create_all_issues.sh
```

**Prerequisites:**
- GitHub CLI (`gh`) must be installed
- Must be authenticated: `gh auth login`
- Must have write access to the repository

### Option 2: Using GitHub Actions Workflow (Recommended)

The easiest method that doesn't require local GitHub CLI setup:

1. Go to the GitHub repository web interface
2. Navigate to **Actions** tab
3. Select "Generate Development Issues" workflow
4. Click "Run workflow"
5. Configure:
   - Phase: `all` (creates all 15 issues)
   - Issue Prefix: (leave empty or add custom prefix)
   - Dry Run: `false` (to actually create issues)
6. Click "Run workflow"

### Option 3: Using Python Script Directly

If you have a GitHub token:

```bash
cd /home/runner/work/cogprime/cogprime

# Install dependencies
pip install PyYAML requests

# Set your GitHub token
export GITHUB_TOKEN="your_personal_access_token"

# Run the script
python .github/scripts/generate_issues.py \
  --phase all \
  --dry-run False \
  --token "$GITHUB_TOKEN" \
  --repo "cogpy/cogprime"
```

### Option 4: Test Mode (Validation Only)

To validate the script without creating issues:

```bash
# Test the bash script exists and is valid
bash -n issues_to_create/create_all_issues.sh

# Test the Python script in dry-run mode
python .github/scripts/generate_issues.py \
  --phase all \
  --dry-run True \
  --token "dummy_token" \
  --repo "cogpy/cogprime"
```

## What Will Be Created

The script will create 15 issues across 3 phases:

- **Phase 2** (Issues 1-5): Advanced Features
  1. Enhanced Cognitive Capabilities Implementation
  2. Self-Reflection Mechanisms Development
  3. Meta-Learning Systems Implementation
  4. Cross-Domain Integration Framework
  5. Relevance Optimization System

- **Phase 3** (Issues 6-10): Framework Integration
  6. Vervaeke Framework Components Integration
  7. Historical Context Integration System
  8. Meaning-Making Systems Development
  9. Relevance Realization Implementation
  10. Meta-Cognitive Synthesis Framework

- **Phase 4** (Issues 11-15): Optimization
  11. Performance Tuning and Optimization
  12. Resource Management System
  13. Scalability Improvements
  14. System Refinement and Integration
  15. Meta-Strategic Enhancement System

Each issue includes:
- Comprehensive description
- Task checklist
- Acceptance criteria
- Priority level
- Effort estimates
- Dependencies
- Appropriate labels

## Troubleshooting

### "ERROR: GitHub CLI (gh) is not installed"
Install GitHub CLI from: https://cli.github.com/

### "ERROR: Not authenticated with GitHub"
Run: `gh auth login` and follow the prompts

### "Permission denied" errors
Ensure you have write access to the repository

### API rate limits
Use the `--dry-run True` option for testing without consuming rate limits

## Files Involved

- `issues_to_create/create_all_issues.sh` - Bash script using GitHub CLI
- `issues_to_create/*.md` - 15 markdown issue templates
- `.github/scripts/generate_issues.py` - Python script using GitHub API
- `.github/workflows/generate-development-issues.yml` - GitHub Actions workflow

## Next Steps

1. Choose your preferred method from the options above
2. Ensure you have necessary permissions and authentication
3. Run the script to create all 15 issues
4. Verify issues were created successfully in the GitHub Issues tab
5. Begin development work following the roadmap

---

*For more details, see `issues_to_create/README.md` and `CREATE_ISSUES.md`*
