#!/bin/bash
#
# Batch create all 15 CogPrime development issues using GitHub CLI
#
# Prerequisites:
#   - GitHub CLI (gh) installed and authenticated
#   - Write access to the repository
#
# Usage:
#   bash create_all_issues.sh
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO="cogpy/cogprime"

echo "=================================================="
echo "Creating 15 CogPrime Development Issues"
echo "Repository: $REPO"
echo "=================================================="
echo

# Check if gh is installed and authenticated
if ! command -v gh &> /dev/null; then
    echo "ERROR: GitHub CLI (gh) is not installed"
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

if ! gh auth status &> /dev/null; then
    echo "ERROR: Not authenticated with GitHub"
    echo "Run: gh auth login"
    exit 1
fi

# Function to create an issue from markdown file
create_issue() {
    local num=$1
    local file=$2
    local title=$3
    local labels=$4
    
    echo "[$num/15] Creating issue: $title"
    
    # Extract the body (skip the title line and metadata)
    body=$(sed '1,/^## Description/d' "$file")
    
    if gh issue create \
        --repo "$REPO" \
        --title "$title" \
        --body "$body" \
        --label "$labels" > /dev/null; then
        echo "  ✅ Created successfully"
    else
        echo "  ❌ Failed to create"
        return 1
    fi
}

# Array of issues with their details
# Format: number|filename|title|labels

issues=(
    "1|01_Enhanced_Cognitive_Capabilities_Implementation.md|Enhanced Cognitive Capabilities Implementation|phase-2,enhancement,cognitive-architecture"
    "2|02_Self-Reflection_Mechanisms_Development.md|Self-Reflection Mechanisms Development|phase-2,meta-cognition,self-improvement"
    "3|03_Meta-Learning_Systems_Implementation.md|Meta-Learning Systems Implementation|phase-2,meta-learning,transfer-learning"
    "4|04_Cross-Domain_Integration_Framework.md|Cross-Domain Integration Framework|phase-2,integration,cross-domain"
    "5|05_Relevance_Optimization_System.md|Relevance Optimization System|phase-2,relevance,optimization"
    "6|06_Vervaeke_Framework_Components_Integration.md|Vervaeke Framework Components Integration|phase-3,vervaeke-framework,4e-cognition"
    "7|07_Historical_Context_Integration_System.md|Historical Context Integration System|phase-3,temporal-reasoning,historical-context"
    "8|08_Meaning-Making_Systems_Development.md|Meaning-Making Systems Development|phase-3,meaning-making,semantics"
    "9|09_Relevance_Realization_Implementation.md|Relevance Realization Implementation|phase-3,relevance-realization,vervaeke-framework"
    "10|10_Meta-Cognitive_Synthesis_Framework.md|Meta-Cognitive Synthesis Framework|phase-3,meta-cognition,synthesis"
    "11|11_Performance_Tuning_and_Optimization.md|Performance Tuning and Optimization|phase-4,optimization,performance"
    "12|12_Resource_Management_System.md|Resource Management System|phase-4,resource-management,scheduling"
    "13|13_Scalability_Improvements.md|Scalability Improvements|phase-4,scalability,distributed-systems"
    "14|14_System_Refinement_and_Integration.md|System Refinement and Integration|phase-4,integration,refinement"
    "15|15_Meta-Strategic_Enhancement_System.md|Meta-Strategic Enhancement System|phase-4,meta-strategy,optimization"
)

success_count=0
failed_count=0

# Create each issue
for issue_data in "${issues[@]}"; do
    IFS='|' read -r num filename title labels <<< "$issue_data"
    filepath="$SCRIPT_DIR/$filename"
    
    if [ ! -f "$filepath" ]; then
        echo "[$num/15] ERROR: File not found: $filename"
        ((failed_count++))
        continue
    fi
    
    if create_issue "$num" "$filepath" "$title" "$labels"; then
        ((success_count++))
    else
        ((failed_count++))
    fi
    
    # Small delay to avoid rate limiting
    sleep 1
done

echo
echo "=================================================="
echo "Summary"
echo "=================================================="
echo "Total issues:   15"
echo "Created:        $success_count"
echo "Failed:         $failed_count"
echo "=================================================="

if [ $success_count -eq 15 ]; then
    echo "✅ All issues created successfully!"
    exit 0
else
    echo "⚠️  Some issues failed to create"
    exit 1
fi
