# Creating the 15 CogPrime Development Issues

This document provides instructions for creating the 15 comprehensive development issues for the CogPrime roadmap (Phases 2, 3, and 4).

## Quick Start

**All 15 issue definitions are available as markdown files in the `issues_to_create/` directory.**

The fastest way to create all issues:

```bash
cd issues_to_create
bash create_all_issues.sh
```

See `issues_to_create/README.md` for detailed information about each issue.

## Overview

The issues have been defined in `.github/scripts/generate_issues.py` and cover:

- **Phase 2** (5 issues): Advanced Features
  1. Enhanced Cognitive Capabilities Implementation
  2. Self-Reflection Mechanisms Development
  3. Meta-Learning Systems Implementation
  4. Cross-Domain Integration Framework
  5. Relevance Optimization System

- **Phase 3** (5 issues): Framework Integration
  6. Vervaeke Framework Components Integration
  7. Historical Context Integration System
  8. Meaning-Making Systems Development
  9. Relevance Realization Implementation
  10. Meta-Cognitive Synthesis Framework

- **Phase 4** (5 issues): Optimization
  11. Performance Tuning and Optimization
  12. Resource Management System
  13. Scalability Improvements
  14. System Refinement and Integration
  15. Meta-Strategic Enhancement System

## Method 1: Using GitHub Actions Workflow (Recommended)

The easiest way to create these issues is through the GitHub Actions workflow:

1. Navigate to the **Actions** tab in the GitHub repository
2. Click on "Generate Development Issues" workflow
3. Click "Run workflow"
4. Configure the workflow:
   - **Phase**: Select `all` to create all 15 issues
   - **Issue Prefix**: Leave empty or add a prefix like `[Roadmap]`
   - **Dry Run**: Set to `false` to actually create the issues
5. Click "Run workflow" to start

The workflow will create all 15 issues with proper labels, descriptions, tasks, and acceptance criteria.

## Method 2: Using the Python Script Directly

If you have repository write access and a GitHub token:

```bash
# Install dependencies
pip install requests PyYAML

# Set your GitHub token
export GITHUB_TOKEN="your_github_personal_access_token"

# Run the script to create all issues
python .github/scripts/generate_issues.py \
  --phase all \
  --prefix "" \
  --dry-run False \
  --token "$GITHUB_TOKEN" \
  --repo "cogpy/cogprime"
```

### Creating Specific Phases

To create issues for specific phases only:

```bash
# Phase 2 only (5 issues)
python .github/scripts/generate_issues.py \
  --phase phase2 \
  --token "$GITHUB_TOKEN" \
  --repo "cogpy/cogprime"

# Phase 3 only (5 issues)
python .github/scripts/generate_issues.py \
  --phase phase3 \
  --token "$GITHUB_TOKEN" \
  --repo "cogpy/cogprime"

# Phase 4 only (5 issues)
python .github/scripts/generate_issues.py \
  --phase phase4 \
  --token "$GITHUB_TOKEN" \
  --repo "cogpy/cogprime"
```

## Method 3: Using GitHub CLI

If you have the GitHub CLI (`gh`) installed and authenticated:

```bash
# Authenticate if needed
gh auth login

# Use the helper script
cd /home/runner/work/cogprime/cogprime
export GITHUB_TOKEN=$(gh auth token)
python .github/scripts/generate_issues.py \
  --phase all \
  --dry-run False \
  --token "$GITHUB_TOKEN" \
  --repo "cogpy/cogprime"
```

## Preview Before Creating (Dry Run)

To see what issues would be created without actually creating them:

```bash
python .github/scripts/generate_issues.py \
  --phase all \
  --prefix "" \
  --dry-run True \
  --token "dummy_token" \
  --repo "cogpy/cogprime"
```

## Issue Format

Each issue includes:

- **Title**: Descriptive title of the development task
- **Description**: Comprehensive explanation of the task and its importance
- **Tasks**: Checklist of specific actionable items
- **Acceptance Criteria**: Clear success metrics and validation requirements  
- **Priority**: High/Medium/Low classification
- **Estimated Effort**: Time estimate in days
- **Phase**: Development phase classification
- **Dependencies**: Prerequisites and related tasks
- **Labels**: Categorization tags (phase-X, feature type, etc.)

## Verification

After creating the issues, verify:

1. All 15 issues are created (5 per phase)
2. Issues have correct labels applied
3. Task checklists are properly formatted
4. Dependencies are clearly listed
5. Acceptance criteria are measurable

## Troubleshooting

### Permission Errors
- Ensure your GitHub token has `repo` or `public_repo` scope with `issues:write` permission
- Verify you have write access to the repository

### API Rate Limits
- GitHub API has rate limits (5000/hour for authenticated users)
- Use `--dry-run True` for testing without consuming rate limit

### Network Issues
- Ensure you have internet connectivity
- Check if GitHub API is accessible from your network

## Additional Resources

- Issue generation script: `.github/scripts/generate_issues.py`
- Workflow definition: `.github/workflows/generate-development-issues.yml`
- Documentation: `.github/scripts/README.md`
- Roadmap: See README.md Development Roadmap section
