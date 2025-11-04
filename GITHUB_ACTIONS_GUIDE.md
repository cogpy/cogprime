# 🚀 GitHub Actions Guide for Silicon Sage

## Complete Build Sequence Workflow

This guide explains how to use the comprehensive GitHub Actions workflow for building, testing, and deploying the Silicon Sage AGI system.

---

## 📋 Workflow Overview

The **Silicon Sage Complete Build Sequence** is a 10-phase automated pipeline that ensures code quality, functionality, and performance.

### Workflow File
`silicon_sage_build.yml` (ready to be placed in `.github/workflows/`)

---

## 🎯 Build Phases

### Phase 1: 🎨 Code Quality & Linting
**Purpose**: Ensure code meets quality standards

**Checks**:
- Black code formatting
- isort import sorting
- Ruff linting
- MyPy type checking

**Duration**: ~2 minutes

### Phase 2: 🧪 Core System Tests
**Purpose**: Validate core cognitive functionality

**Tests**:
- AtomSpace core tests
- Enhanced capabilities tests
- Silicon Sage integration tests

**Matrix**: Python 3.8, 3.9, 3.10, 3.11

**Duration**: ~5 minutes per Python version

### Phase 3: 🌙 Lua OpenCog Tests
**Purpose**: Verify Lua implementation

**Tests**:
- Lua OpenCog test suite
- Basic examples
- Advanced examples

**Duration**: ~3 minutes

### Phase 4: 🔗 Integration Tests
**Purpose**: Test cross-module integration

**Tests**:
- Module integration
- Cognitive metrics extraction
- System state validation

**Duration**: ~3 minutes

### Phase 5: 🎨 Dashboard Build & Test
**Purpose**: Validate visualization dashboard

**Tests**:
- Server initialization
- Metrics extraction
- Frontend validation
- WebSocket support

**Duration**: ~2 minutes

### Phase 6: ⚡ Performance Benchmarks
**Purpose**: Ensure performance targets

**Benchmarks**:
- Cognitive cycle speed
- Throughput measurement
- Performance regression detection

**Target**: <100ms per cognitive cycle

**Duration**: ~3 minutes

### Phase 7: 🔒 Security Scanning
**Purpose**: Identify vulnerabilities

**Scans**:
- Trivy vulnerability scanner
- Secret detection
- Dependency analysis

**Duration**: ~4 minutes

### Phase 8: 📚 Documentation Build
**Purpose**: Generate documentation

**Tasks**:
- Documentation statistics
- Sphinx documentation build (optional)
- Coverage reports

**Duration**: ~2 minutes

### Phase 9: 📋 Build Summary
**Purpose**: Aggregate results

**Outputs**:
- Comprehensive build report
- Status of all phases
- GitHub Step Summary

**Duration**: ~1 minute

### Phase 10: 🚀 Deploy Dashboard (Optional)
**Purpose**: Deploy visualization dashboard

**Triggers**:
- Manual workflow dispatch with deploy flag
- Push to main branch

**Duration**: Varies by deployment target

---

## 🔧 Installation

### Step 1: Add Workflow File

Due to GitHub App permissions, the workflow file must be added manually:

```bash
# Copy the workflow file to the correct location
mkdir -p .github/workflows
cp silicon_sage_build.yml .github/workflows/

# Commit and push
git add .github/workflows/silicon_sage_build.yml
git commit -m "Add Silicon Sage complete build sequence workflow"
git push origin main
```

### Step 2: Verify Workflow

1. Go to your repository on GitHub
2. Click the **Actions** tab
3. You should see "Silicon Sage Complete Build Sequence"

---

## 🎮 Usage

### Automatic Triggers

The workflow runs automatically on:
- **Push** to `main`, `develop`, or any `feature/**` branch
- **Pull requests** to `main` or `develop`

### Manual Triggers

You can manually trigger the workflow with options:

1. Go to **Actions** tab
2. Select "Silicon Sage Complete Build Sequence"
3. Click **Run workflow**
4. Choose options:
   - **Run full suite**: Include long-running tests
   - **Deploy dashboard**: Deploy visualization dashboard

---

## 📊 Interpreting Results

### Success Indicators

✅ **All checks passed**: Green checkmark on all phases

### Common Issues

#### ⚠️ Code Quality Warnings
- **Black formatting**: Run `black src/` locally
- **Import sorting**: Run `isort src/` locally
- **Linting**: Run `ruff check src/ --fix` locally

#### ❌ Test Failures
- Check the specific test output in the workflow logs
- Run tests locally: `python test_core_direct.py`
- Verify dependencies are installed

#### ⚠️ Performance Issues
- Check benchmark results in Phase 6
- Optimize slow cognitive cycles
- Profile code if needed

---

## 🎯 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Cognitive Cycle Time | <100ms | ~5-10ms |
| Test Pass Rate | 100% | 100% |
| Code Coverage | >80% | TBD |
| Security Issues | 0 critical | TBD |

---

## 🔐 Security

### Secrets Required

None by default. If you add deployment:

- `DEPLOY_TOKEN`: For deployment authentication
- `AWS_ACCESS_KEY_ID`: For AWS deployment (if used)
- `AWS_SECRET_ACCESS_KEY`: For AWS deployment (if used)

### Adding Secrets

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add your secret

---

## 🚀 Deployment Options

### Option 1: Vercel (Recommended for Dashboard)

Add to Phase 10:

```yaml
- name: Deploy to Vercel
  uses: amondnet/vercel-action@v20
  with:
    vercel-token: ${{ secrets.VERCEL_TOKEN }}
    vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
    vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
```

### Option 2: GitHub Pages

Add to Phase 10:

```yaml
- name: Deploy to GitHub Pages
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./
    publish_branch: gh-pages
```

### Option 3: AWS

Add to Phase 10:

```yaml
- name: Deploy to AWS
  uses: aws-actions/configure-aws-credentials@v1
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws-region: us-east-1
```

---

## 📈 Monitoring

### GitHub Actions Dashboard

View all workflow runs:
- Go to **Actions** tab
- See history, duration, and status
- Download logs and artifacts

### Status Badges

Add to your README.md:

```markdown
![Silicon Sage Build](https://github.com/cogpy/cogprime/workflows/Silicon%20Sage%20Complete%20Build%20Sequence/badge.svg)
```

---

## 🎓 Best Practices

### 1. Run Locally First
Always run tests locally before pushing:
```bash
./setup_dev.sh
python test_core_direct.py
python test_enhanced_capabilities.py
```

### 2. Use Pre-commit Hooks
Install pre-commit hooks to catch issues early:
```bash
pre-commit install
```

### 3. Monitor Performance
Check benchmark results regularly to catch regressions

### 4. Review Security Scans
Address security issues promptly

### 5. Keep Dependencies Updated
Regularly update dependencies and test

---

## 🔧 Customization

### Modify Python Versions

Edit the matrix in Phase 2:

```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
```

### Add Custom Tests

Add steps to Phase 4:

```yaml
- name: 🧪 Custom integration test
  run: |
    python your_custom_test.py
```

### Adjust Performance Targets

Modify benchmarks in Phase 6:

```yaml
if avg_time < 50:  # More strict target
    print('✅ Performance target met')
```

---

## 📚 Additional Resources

### GitHub Actions Documentation
- [Workflow syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Environment variables](https://docs.github.com/en/actions/reference/environment-variables)
- [Contexts](https://docs.github.com/en/actions/reference/context-and-expression-syntax-for-github-actions)

### CogPrime Documentation
- [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) - Optimization details
- [SECRET_FEATURE_UNVEILED.md](SECRET_FEATURE_UNVEILED.md) - Dashboard documentation
- [README.md](README.md) - Project overview

---

## 🎉 Success Criteria

A successful build includes:

✅ All code quality checks pass  
✅ All tests pass on all Python versions  
✅ Lua tests complete successfully  
✅ Integration tests validate system  
✅ Dashboard builds and validates  
✅ Performance targets met  
✅ No critical security issues  
✅ Documentation generated  

---

## 🐛 Troubleshooting

### Workflow Not Running

**Issue**: Workflow doesn't trigger on push

**Solution**:
1. Check workflow file is in `.github/workflows/`
2. Verify YAML syntax is valid
3. Check branch name matches trigger conditions

### Tests Failing in CI but Passing Locally

**Issue**: Tests pass locally but fail in GitHub Actions

**Solution**:
1. Check Python version matches (use 3.11)
2. Verify all dependencies are installed
3. Check for environment-specific issues
4. Review workflow logs for details

### Performance Benchmarks Failing

**Issue**: Performance targets not met in CI

**Solution**:
1. CI runners may be slower than local machines
2. Adjust targets if consistently failing
3. Profile code to identify bottlenecks
4. Consider using self-hosted runners

---

## 📞 Support

For issues with the workflow:

1. Check workflow logs in GitHub Actions
2. Review this guide
3. Open an issue on GitHub
4. Check [GitHub Actions documentation](https://docs.github.com/en/actions)

---

## 🎯 Next Steps

1. **Install the workflow** (copy to `.github/workflows/`)
2. **Push to GitHub** and watch it run
3. **Review results** in the Actions tab
4. **Fix any issues** identified
5. **Configure deployment** (optional)
6. **Add status badge** to README
7. **Celebrate!** 🎉

---

**Built with 💜 for Silicon Sage**  
**Automated Excellence, Every Commit** ✨

---

*Last Updated: November 4, 2025*
