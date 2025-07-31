# Deployment Guide for GitFlow Analytics

This guide covers the deployment process for GitFlow Analytics. **As of v1.0.1, the project uses fully automated releases** via GitHub Actions and semantic-release. Manual deployment is only needed for emergency situations.

## Table of Contents

1. [Current Status](#current-status)
2. [Automated Release Process](#automated-release-process)
3. [Conventional Commits](#conventional-commits)
4. [Monitoring Releases](#monitoring-releases)
5. [Manual Deployment (Emergency Only)](#manual-deployment-emergency-only)
6. [Legacy Manual Process](#legacy-manual-process)

## Current Status

**GitFlow Analytics v1.0.1 is live on PyPI!**

- 📦 **PyPI Package**: [gitflow-analytics](https://pypi.org/project/gitflow-analytics/)
- 🏷️ **Latest Version**: v1.0.1
- 🚀 **Installation**: `pip install gitflow-analytics`
- 🤖 **Release Method**: Fully automated via GitHub Actions

## Automated Release Process

The project uses **python-semantic-release** with GitHub Actions for fully automated deployments. This ensures:

- ✅ Consistent versioning across all platforms
- ✅ Automatic changelog generation
- ✅ Secure PyPI publishing via trusted publishing
- ✅ Atomic releases (git tag + GitHub release + PyPI upload)
- ✅ Comprehensive testing before release

### How It Works

1. **Developer commits** using conventional commit format
2. **GitHub Actions detects** push to main branch
3. **Semantic-release analyzes** commits to determine version bump
4. **Automated pipeline** runs:
   - Updates `src/gitflow_analytics/_version.py`
   - Creates git tag
   - Runs full test suite
   - Builds package
   - Publishes to PyPI
   - Creates GitHub release
   - Updates CHANGELOG.md

### Triggering a Release

Simply push commits with conventional commit messages to the `main` branch:

```bash
git commit -m "feat: add new analysis feature"
git push origin main  # This triggers the release pipeline
```

## Conventional Commits

Use conventional commit messages to trigger automatic version bumps:

### Commit Types and Version Impact

| Commit Type | Version Bump | Example |
|-------------|--------------|----------|
| `feat:` | Minor (0.X.0) | `feat: add JIRA integration` |
| `fix:` | Patch (0.0.X) | `fix: resolve identity resolution bug` |
| `docs:` | Patch (0.0.X) | `docs: update installation guide` |
| `chore:` | Patch (0.0.X) | `chore: update dependencies` |
| `style:` | Patch (0.0.X) | `style: fix code formatting` |
| `refactor:` | Patch (0.0.X) | `refactor: improve error handling` |
| `perf:` | Patch (0.0.X) | `perf: optimize commit processing` |
| `test:` | Patch (0.0.X) | `test: add integration tests` |
| `ci:` | Patch (0.0.X) | `ci: update GitHub Actions` |
| `build:` | Patch (0.0.X) | `build: update pyproject.toml` |

### Breaking Changes

For breaking changes (major version bump), add `BREAKING CHANGE:` in the commit body:

```bash
git commit -m "feat: redesign configuration format

BREAKING CHANGE: Configuration file format has changed from YAML to TOML.
Existing config.yaml files need to be converted to config.toml format."
```

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Examples:
```bash
feat(cli): add --validate-only flag for configuration testing
fix(cache): handle database lock errors gracefully  
docs: add PyPI installation instructions
chore(deps): update GitHub Actions to v4
```

## Monitoring Releases

### Check Release Status

1. **GitHub Actions**: Monitor the [semantic-release workflow](https://github.com/bobmatnyc/gitflow-analytics/actions/workflows/semantic-release.yml)
2. **PyPI**: Check the [package page](https://pypi.org/project/gitflow-analytics/) for new versions
3. **GitHub Releases**: View [releases page](https://github.com/bobmatnyc/gitflow-analytics/releases) for release notes

### Release Workflow Logs

The automated release process provides detailed logs:
- Semantic analysis results
- Version calculation logic
- Test execution results
- Build and publish status
- Error details if any step fails

### Troubleshooting Failed Releases

If the automated release fails:

1. **Check workflow logs** in GitHub Actions
2. **Common issues**:
   - Test failures block the release
   - PyPI trusted publishing configuration
   - GitHub token permissions
   - Semantic-release configuration errors

3. **Resolution**:
   - Fix the underlying issue
   - Push another commit to main
   - The workflow will retry automatically

## Manual Deployment (Emergency Only)

**⚠️ Warning**: Manual deployment should only be used in emergency situations when the automated system is not working.

### Prerequisites

```bash
pip install --upgrade pip build twine python-semantic-release
```

### Emergency Release Process

1. **Manual version bump:**
   ```bash
   # Edit src/gitflow_analytics/_version.py manually
   git commit -m "chore(release): emergency version bump to X.Y.Z"
   ```

2. **Create tag and build:**
   ```bash
   git tag -a vX.Y.Z -m "Emergency release X.Y.Z"
   python -m build
   ```

3. **Test and publish:**
   ```bash
   python -m twine check dist/*
   python -m twine upload dist/*
   ```

4. **Push to GitHub:**
   ```bash
   git push origin main --tags
   ```

### When to Use Manual Deployment

- GitHub Actions is down for extended periods
- Critical security fix needed immediately
- Automated system configuration is broken
- PyPI trusted publishing is not working

### Recovery After Manual Release

After manual deployment, ensure the automated system is working:

1. Verify the version in `_version.py` matches the released version
2. Ensure the git tag exists and is pushed
3. Test the next automated release with a patch commit
4. Update CHANGELOG.md if it wasn't auto-generated

## Legacy Manual Process

<details>
<summary>Click to expand legacy manual deployment process (archived)</summary>

**Note**: This section is kept for historical reference. The project now uses automated releases.

### Manual Build Process (Deprecated)

1. **Clean previous builds:**
   ```bash
   rm -rf dist/ build/ *.egg-info
   ```

2. **Build the package:**
   ```bash
   python -m build
   ```

3. **Verify the build:**
   ```bash
   ls -la dist/
   ```

### Manual PyPI Publishing (Deprecated)

1. **Setup PyPI credentials** in `~/.pypirc`
2. **Test with TestPyPI:**
   ```bash
   twine upload --repository testpypi dist/*
   ```
3. **Publish to PyPI:**
   ```bash
   twine upload dist/*
   ```

### Manual GitHub Release (Deprecated)

1. **Tag the version:**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   ```
2. **Create GitHub Release** via web interface
3. **Attach build artifacts**

</details>

## Development Workflow

### Feature Development and Release

1. **Feature development:**
   ```bash
   git checkout -b feature/new-analysis-metric
   # Make changes
   git commit -m "feat: add DORA metrics calculation"
   git push origin feature/new-analysis-metric
   # Create PR
   ```

2. **Merge to main:**
   ```bash
   # After PR approval and merge
   git checkout main
   git pull origin main
   # The semantic-release workflow automatically runs
   ```

3. **Release happens automatically:**
   - Version is calculated from commit messages
   - Package is built and tested
   - PyPI release is published
   - GitHub release is created
   - CHANGELOG.md is updated

### Hotfix Workflow

```bash
git checkout main
git pull origin main
git checkout -b hotfix/critical-bug
# Fix the issue
git commit -m "fix: resolve critical identity resolution bug"
git push origin hotfix/critical-bug
# Create PR, merge to main
# Patch release happens automatically
```

## Troubleshooting

### Common Issues

1. **Build fails:**
   - Ensure all dependencies are installed
   - Check for syntax errors in setup files
   - Verify `MANIFEST.in` includes all necessary files

2. **Upload fails:**
   - Check PyPI credentials
   - Ensure version doesn't already exist
   - Verify package name availability

3. **Import errors after installation:**
   - Check package structure
   - Verify all modules are included
   - Test locally with `pip install -e .`

### Getting Help

- [Python Packaging Guide](https://packaging.python.org)
- [PyPI Help](https://pypi.org/help/)
- [Project Issues](https://github.com/bobmatnyc/gitflow-analytics/issues)

## See Also

- [CLAUDE.md](../CLAUDE.md) - Developer instructions with detailed versioning info
- [README.md](../README.md) - User documentation and installation
- [CHANGELOG.md](../CHANGELOG.md) - Automatically generated version history
- [PyPI Package](https://pypi.org/project/gitflow-analytics/) - Official package page
- [GitHub Releases](https://github.com/bobmatnyc/gitflow-analytics/releases) - Release notes and downloads
- [GitHub Actions](https://github.com/bobmatnyc/gitflow-analytics/actions) - CI/CD pipeline status