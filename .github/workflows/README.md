# GitHub Actions Workflows

## Create CogPrime Development Issues

**Workflow:** `create-cogprime-issues.yml`

This workflow creates all 15 CogPrime development issues for Phases 2-4 of the roadmap.

### How to Run

1. Go to the **Actions** tab: https://github.com/cogpy/cogprime/actions
2. Select "**Create CogPrime Development Issues**" from the workflow list
3. Click "**Run workflow**" button
4. Type `CREATE` in the confirmation field
5. Click "**Run workflow**" to start

### What It Does

- Creates 15 comprehensive development issues
- Automatically applies appropriate labels (phase-2, phase-3, phase-4, etc.)
- Includes detailed descriptions, tasks, acceptance criteria, and effort estimates
- Provides a summary of all created issues

### Safety Features

- Requires typing "CREATE" to confirm execution (prevents accidental runs)
- Validates confirmation before proceeding
- Provides detailed output and summary

### Issues Created

**Phase 2: Advanced Features (5 issues)**
1. Enhanced Cognitive Capabilities Implementation
2. Self-Reflection Mechanisms Development
3. Meta-Learning Systems Implementation
4. Cross-Domain Integration Framework
5. Relevance Optimization System

**Phase 3: Framework Integration (5 issues)**
6. Vervaeke Framework Components Integration
7. Historical Context Integration System
8. Meaning-Making Systems Development
9. Relevance Realization Implementation
10. Meta-Cognitive Synthesis Framework

**Phase 4: Optimization (5 issues)**
11. Performance Tuning and Optimization
12. Resource Management System
13. Scalability Improvements
14. System Refinement and Integration
15. Meta-Strategic Enhancement System

### Troubleshooting

**"Issue creation not confirmed"**
- Make sure you typed `CREATE` (all caps) in the confirmation field

**"Permission denied" or "403 Forbidden"**
- Ensure the workflow has `issues: write` permission (already configured)
- Check that the repository settings allow GitHub Actions to create issues

**"Rate limit exceeded"**
- Wait a few minutes and try again
- GitHub API has rate limits that may be reached if creating many issues rapidly

## Generate Development Issues (Advanced)

**Workflow:** `generate-development-issues.yml`

A more flexible workflow that allows:
- Selecting specific phases (phase2, phase3, phase4, or all)
- Adding custom prefixes to issue titles
- Dry-run mode to preview without creating

This is useful for testing or creating issues in batches.
