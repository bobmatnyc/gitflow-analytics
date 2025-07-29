# Deployment Guide for GitFlow Analytics

This guide covers the deployment process for GitFlow Analytics, including versioning, building, and publishing to PyPI.

## Table of Contents

1. [Semantic Versioning](#semantic-versioning)
2. [Building the Package](#building-the-package)
3. [Publishing to PyPI](#publishing-to-pypi)
4. [GitHub Release Process](#github-release-process)
5. [Continuous Integration](#continuous-integration)

## Semantic Versioning

GitFlow Analytics follows [Semantic Versioning](https://semver.org/) (SemVer) principles:

- **MAJOR** version (X.0.0): Incompatible API changes
- **MINOR** version (0.X.0): Backwards-compatible functionality additions
- **PATCH** version (0.0.X): Backwards-compatible bug fixes

### Version Management

The version is stored in `src/gitflow_analytics/_version.py`:

```python
__version__ = "1.0.0"
```

To update the version:

1. Edit `src/gitflow_analytics/_version.py`
2. Update the `__version__` string
3. Commit the change with message: `chore: bump version to X.Y.Z`

## Building the Package

### Prerequisites

Ensure you have the latest build tools:

```bash
pip install --upgrade pip setuptools wheel build
```

### Build Process

1. **Clean previous builds:**
   ```bash
   rm -rf dist/ build/ *.egg-info
   ```

2. **Build the package:**
   ```bash
   python -m build
   ```

   This creates both wheel and source distributions in the `dist/` directory.

3. **Verify the build:**
   ```bash
   ls -la dist/
   # Should show:
   # gitflow_analytics-X.Y.Z-py3-none-any.whl
   # gitflow_analytics-X.Y.Z.tar.gz
   ```

## Publishing to PyPI

### Setup PyPI Credentials

1. **Create PyPI account:**
   - Register at [pypi.org](https://pypi.org)
   - Enable 2FA (recommended)

2. **Create API token:**
   - Go to Account Settings → API tokens
   - Create a new token with upload permissions
   - Save the token securely

3. **Configure credentials:**
   
   Create `~/.pypirc`:
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-YOUR_TOKEN_HERE

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-YOUR_TEST_TOKEN_HERE
   ```

### Publishing Process

1. **Test with TestPyPI first:**
   ```bash
   pip install twine
   twine upload --repository testpypi dist/*
   ```

2. **Verify test package:**
   ```bash
   pip install --index-url https://test.pypi.org/simple/ gitflow-analytics
   ```

3. **Publish to PyPI:**
   ```bash
   twine upload dist/*
   ```

4. **Verify publication:**
   ```bash
   pip install gitflow-analytics
   gitflow-analytics --version
   ```

## GitHub Release Process

### Creating a Release

1. **Tag the version:**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

2. **Create GitHub Release:**
   - Go to repository → Releases → Create new release
   - Select the tag
   - Title: `v1.0.0`
   - Generate release notes
   - Attach the wheel and tarball from `dist/`
   - Publish release

### Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `_version.py`
- [ ] Tagged in git
- [ ] Published to PyPI
- [ ] GitHub release created

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

### Setting up CI

1. Add PyPI token to GitHub secrets:
   - Repository → Settings → Secrets → Actions
   - Add `PYPI_API_TOKEN` with your PyPI token

2. The workflow will automatically:
   - Trigger on GitHub release creation
   - Build the package
   - Upload to PyPI

## Development Workflow

### Version Bump Workflow

1. **Feature development:**
   ```bash
   git checkout -b feature/new-feature
   # Make changes
   git commit -m "feat: add new feature"
   git push origin feature/new-feature
   # Create PR
   ```

2. **Prepare release:**
   ```bash
   git checkout main
   git pull origin main
   # Update version in _version.py
   git commit -m "chore: bump version to 1.1.0"
   ```

3. **Create release:**
   ```bash
   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin main --tags
   ```

4. **Post-release:**
   - CI automatically publishes to PyPI
   - Create GitHub release manually or via CI

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

- [CLAUDE.md](./CLAUDE.md) - Developer instructions
- [README.md](../README.md) - User documentation
- [CHANGELOG.md](../CHANGELOG.md) - Version history