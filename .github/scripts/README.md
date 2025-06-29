# GitHub Actions Issue Generation Workflow

This directory contains GitHub Actions workflows and scripts for automatically generating detailed, actionable issues for the CogPrime development roadmap.

## Overview

The issue generation system creates comprehensive development issues for each phase of the CogPrime roadmap:

- **Phase 2: Advanced Features** - Enhanced cognitive capabilities, self-reflection, meta-learning, cross-domain integration, and relevance optimization
- **Phase 3: Framework Integration** - Vervaeke framework components, historical context integration, meaning-making systems, relevance realization, and meta-cognitive synthesis  
- **Phase 4: Optimization** - Performance tuning, resource management, scalability improvements, system refinement, and meta-strategic enhancement

## Workflow Usage

### Manual Execution

Navigate to the **Actions** tab in the GitHub repository and run the "Generate Development Issues" workflow with these options:

- **Phase**: Select which phase(s) to generate issues for (`all`, `phase2`, `phase3`, or `phase4`)
- **Issue Prefix**: Optional prefix for issue titles (e.g., `[Phase 2]`, `[Core Dev]`)
- **Dry Run**: Set to `true` to preview issues without creating them

### Local Testing

You can test the issue generation locally:

```bash
# Set up environment
pip install PyYAML requests

# Test Phase 2 issues (dry run)
python .github/scripts/generate_issues.py \
  --phase phase2 \
  --prefix "[Phase 2] " \
  --dry-run true \
  --token "your-github-token" \
  --repo "drzo/cogprime"

# Generate all issues (requires GitHub API access)
python .github/scripts/generate_issues.py \
  --phase all \
  --token "$GITHUB_TOKEN" \
  --repo "drzo/cogprime"
```

## Issue Format

Each generated issue follows a consistent, comprehensive format:

### Structure
- **Title**: Clear, descriptive title with optional prefix
- **Description**: Detailed explanation of the development task
- **Tasks**: Specific, actionable checklist items
- **Acceptance Criteria**: Clear success metrics and validation requirements
- **Priority**: High/Medium/Low priority classification
- **Estimated Effort**: Time estimate in days
- **Dependencies**: Prerequisites and related components
- **Labels**: Categorization tags for filtering and organization

### Example Issue

```markdown
**Description:**
Implement advanced cognitive processing capabilities that extend beyond the basic architecture established in Phase 1...

**Tasks:**
- [ ] Implement advanced pattern recognition algorithms in reasoning module
- [ ] Add sophisticated memory consolidation mechanisms
- [ ] Create adaptive attention allocation systems
...

**Acceptance Criteria:**
- Advanced reasoning capabilities demonstrate 30% improvement over baseline
- Memory consolidation reduces storage requirements by 20% while maintaining accuracy
...

**Priority:** High
**Estimated Effort:** 8-12 days
**Phase:** Phase 2

**Dependencies:**
- Completed Phase 1 core implementation
- Performance baseline metrics
```

## Generated Issues by Phase

### Phase 2: Advanced Features (5 issues)
1. **Enhanced Cognitive Capabilities Implementation** - Core processing improvements
2. **Self-Reflection Mechanisms Development** - Meta-cognitive awareness and introspection
3. **Meta-Learning Systems Implementation** - Learning how to learn effectively
4. **Cross-Domain Integration Framework** - Unified processing across domains
5. **Relevance Optimization System** - Dynamic resource prioritization

### Phase 3: Framework Integration (5 issues)  
1. **Vervaeke Framework Components Integration** - 4E cognition implementation
2. **Historical Context Integration System** - Temporal reasoning and episodic memory
3. **Meaning-Making Systems Development** - Semantic understanding and interpretation
4. **Relevance Realization Implementation** - Vervaeke's relevance framework
5. **Meta-Cognitive Synthesis Framework** - Higher-order thinking integration

### Phase 4: Optimization (5 issues)
1. **Performance Tuning and Optimization** - Algorithmic and computational efficiency
2. **Resource Management System** - Dynamic allocation and scheduling
3. **Scalability Improvements** - Distributed processing and growth handling
4. **System Refinement and Integration** - Seamless component operation
5. **Meta-Strategic Enhancement System** - Cognitive strategy optimization

## Customization

### Adding New Issues

To add issues for additional phases or modify existing ones:

1. Edit `.github/scripts/generate_issues.py`
2. Add new issue definitions following the `Issue` dataclass format
3. Create appropriate `get_phaseX_issues()` methods
4. Update the `generate_all_issues()` method to include new phases

### Modifying Issue Format

The issue format can be customized by modifying the `_format_issue_body()` method in the `CogPrimeIssueGenerator` class.

### Labels and Categories

Update the `labels` field in issue definitions to match your repository's labeling scheme.

## Integration with CogPrime Architecture

The generated issues are designed to integrate seamlessly with the existing CogPrime architecture:

- **Module Alignment**: Issues target specific modules (`src/modules/`)
- **Documentation Standards**: Follow existing patterns from `docs/KAWAII_HEXAPOD_ACTIONABLE_ISSUES.md`
- **Testing Integration**: Include test requirements compatible with `src/tests/`
- **Performance Targets**: Align with real-time cognitive processing requirements

## Contributing

When contributing new issues or modifications:

1. Follow the established issue format and quality standards
2. Ensure issues are actionable and have clear acceptance criteria
3. Include appropriate effort estimates and dependencies
4. Test locally before submitting changes
5. Update documentation to reflect any structural changes

## Troubleshooting

### Common Issues

- **API Rate Limits**: GitHub API has rate limits; use dry-run mode for testing
- **Token Permissions**: Ensure GitHub token has `issues:write` permissions
- **Network Connectivity**: Verify network access for API calls
- **Python Dependencies**: Install required packages with `pip install PyYAML requests`

### Debugging

Enable verbose output by modifying the script's logging level or use dry-run mode to preview generated content.