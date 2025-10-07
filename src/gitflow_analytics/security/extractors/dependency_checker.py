"""Check for vulnerable dependencies in project files."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import toml

logger = logging.getLogger(__name__)


class DependencyChecker:
    """Check for known vulnerabilities in project dependencies."""

    def __init__(self, config: Any):
        """Initialize dependency checker.

        Args:
            config: Dependency scanning configuration
        """
        self.config = config
        self.vulnerability_cache = {}

    def check_files(self, files_changed: list[str], repo_path: Path) -> list[dict]:
        """Check dependency files for vulnerable packages.

        Args:
            files_changed: List of changed files
            repo_path: Repository path

        Returns:
            List of vulnerability findings
        """
        findings = []

        for file_path in files_changed:
            if self._is_dependency_file(file_path):
                full_path = repo_path / file_path
                if full_path.exists():
                    file_findings = self._check_dependency_file(full_path, file_path)
                    findings.extend(file_findings)

        return findings

    def _is_dependency_file(self, file_path: str) -> bool:
        """Check if file is a dependency specification file."""
        dependency_files = [
            "package.json",
            "package-lock.json",
            "yarn.lock",
            "requirements.txt",
            "Pipfile",
            "Pipfile.lock",
            "poetry.lock",
            "pyproject.toml",
            "go.mod",
            "go.sum",
            "Gemfile",
            "Gemfile.lock",
            "pom.xml",
            "build.gradle",
            "composer.json",
            "composer.lock",
        ]

        file_name = Path(file_path).name
        return file_name in dependency_files

    def _check_dependency_file(self, file_path: Path, relative_path: str) -> list[dict]:
        """Check a specific dependency file for vulnerabilities."""
        findings = []
        file_name = file_path.name

        try:
            if file_name == "package.json" and self.config.check_npm:
                dependencies = self._parse_package_json(file_path)
                findings.extend(self._check_npm_dependencies(dependencies, relative_path))

            elif file_name == "requirements.txt" and self.config.check_pip:
                dependencies = self._parse_requirements_txt(file_path)
                findings.extend(self._check_python_dependencies(dependencies, relative_path))

            elif file_name == "pyproject.toml" and self.config.check_pip:
                dependencies = self._parse_pyproject_toml(file_path)
                findings.extend(self._check_python_dependencies(dependencies, relative_path))

            elif file_name == "go.mod" and self.config.check_go:
                dependencies = self._parse_go_mod(file_path)
                findings.extend(self._check_go_dependencies(dependencies, relative_path))

            elif file_name == "Gemfile" and self.config.check_ruby:
                dependencies = self._parse_gemfile(file_path)
                findings.extend(self._check_ruby_dependencies(dependencies, relative_path))

        except Exception as e:
            logger.warning(f"Error checking dependency file {relative_path}: {e}")

        return findings

    def _parse_package_json(self, file_path: Path) -> dict[str, str]:
        """Parse package.json for dependencies."""
        dependencies = {}

        try:
            with open(file_path) as f:
                data = json.load(f)

            for dep_type in ["dependencies", "devDependencies", "peerDependencies"]:
                if dep_type in data:
                    for name, version_spec in data[dep_type].items():
                        # Clean version spec (remove ^, ~, etc.)
                        clean_version = re.sub(r"^[^\d]*", "", version_spec)
                        clean_version = clean_version.split(" ")[0]  # Handle version ranges
                        dependencies[name] = clean_version

        except Exception as e:
            logger.debug(f"Error parsing package.json: {e}")

        return dependencies

    def _parse_requirements_txt(self, file_path: Path) -> dict[str, str]:
        """Parse requirements.txt for Python packages."""
        dependencies = {}

        try:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Parse package==version or package>=version
                        match = re.match(r"^([a-zA-Z0-9\-_]+)([=<>!]+)(.+)$", line)
                        if match:
                            name = match.group(1)
                            version_spec = match.group(3)
                            # Clean version
                            clean_version = version_spec.split(",")[0].strip()
                            dependencies[name.lower()] = clean_version

        except Exception as e:
            logger.debug(f"Error parsing requirements.txt: {e}")

        return dependencies

    def _parse_pyproject_toml(self, file_path: Path) -> dict[str, str]:
        """Parse pyproject.toml for Python dependencies."""
        dependencies = {}

        try:
            with open(file_path) as f:
                data = toml.load(f)

            # Check different dependency sections
            sections = [
                ["project", "dependencies"],
                ["tool", "poetry", "dependencies"],
                ["tool", "poetry", "dev-dependencies"],
            ]

            for section_path in sections:
                section = data
                for key in section_path:
                    if key in section:
                        section = section[key]
                    else:
                        break
                else:
                    # Successfully navigated to the section
                    if isinstance(section, dict):
                        for name, spec in section.items():
                            if isinstance(spec, str):
                                # Simple version string
                                clean_version = re.sub(r"^[^\d]*", "", spec)
                                dependencies[name.lower()] = clean_version
                            elif isinstance(spec, dict) and "version" in spec:
                                # Poetry-style with version key
                                clean_version = re.sub(r"^[^\d]*", "", spec["version"])
                                dependencies[name.lower()] = clean_version

        except Exception as e:
            logger.debug(f"Error parsing pyproject.toml: {e}")

        return dependencies

    def _parse_go_mod(self, file_path: Path) -> dict[str, str]:
        """Parse go.mod for Go dependencies."""
        dependencies = {}

        try:
            with open(file_path) as f:
                in_require_block = False
                for line in f:
                    line = line.strip()

                    if line.startswith("require ("):
                        in_require_block = True
                        continue
                    elif line == ")":
                        in_require_block = False
                        continue

                    if in_require_block or line.startswith("require "):
                        # Parse: module/name v1.2.3
                        parts = line.replace("require ", "").split()
                        if len(parts) >= 2 and parts[1].startswith("v"):
                            dependencies[parts[0]] = parts[1]

        except Exception as e:
            logger.debug(f"Error parsing go.mod: {e}")

        return dependencies

    def _parse_gemfile(self, file_path: Path) -> dict[str, str]:
        """Parse Gemfile for Ruby dependencies."""
        dependencies = {}

        try:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    # Parse: gem 'name', '~> version'
                    match = re.match(r"gem\s+['\"]([^'\"]+)['\"](?:,\s*['\"]([^'\"]+)['\"])?", line)
                    if match:
                        name = match.group(1)
                        version_spec = match.group(2) if match.group(2) else "unknown"
                        clean_version = re.sub(r"^[^\d]*", "", version_spec)
                        dependencies[name] = clean_version

        except Exception as e:
            logger.debug(f"Error parsing Gemfile: {e}")

        return dependencies

    def _check_npm_dependencies(self, dependencies: dict[str, str], file_path: str) -> list[dict]:
        """Check NPM packages for vulnerabilities using GitHub Advisory Database."""
        findings = []

        for package_name, package_version in dependencies.items():
            vulnerabilities = self._query_vulnerability_db("npm", package_name, package_version)
            for vuln in vulnerabilities:
                findings.append(
                    {
                        "type": "dependency",
                        "vulnerability_type": "vulnerable_dependency",
                        "severity": vuln["severity"],
                        "package": package_name,
                        "version": package_version,
                        "file": file_path,
                        "cve": vuln.get("cve", ""),
                        "message": vuln.get(
                            "summary", f"Vulnerable {package_name}@{package_version}"
                        ),
                        "tool": "dependency_checker",
                        "confidence": "high",
                    }
                )

        return findings

    def _check_python_dependencies(
        self, dependencies: dict[str, str], file_path: str
    ) -> list[dict]:
        """Check Python packages for vulnerabilities."""
        findings = []

        for package_name, package_version in dependencies.items():
            vulnerabilities = self._query_vulnerability_db("pip", package_name, package_version)
            for vuln in vulnerabilities:
                findings.append(
                    {
                        "type": "dependency",
                        "vulnerability_type": "vulnerable_dependency",
                        "severity": vuln["severity"],
                        "package": package_name,
                        "version": package_version,
                        "file": file_path,
                        "cve": vuln.get("cve", ""),
                        "message": vuln.get(
                            "summary", f"Vulnerable {package_name}=={package_version}"
                        ),
                        "tool": "dependency_checker",
                        "confidence": "high",
                    }
                )

        return findings

    def _check_go_dependencies(self, dependencies: dict[str, str], file_path: str) -> list[dict]:
        """Check Go modules for vulnerabilities."""
        findings = []

        for module_name, module_version in dependencies.items():
            vulnerabilities = self._query_vulnerability_db("go", module_name, module_version)
            for vuln in vulnerabilities:
                findings.append(
                    {
                        "type": "dependency",
                        "vulnerability_type": "vulnerable_dependency",
                        "severity": vuln["severity"],
                        "package": module_name,
                        "version": module_version,
                        "file": file_path,
                        "cve": vuln.get("cve", ""),
                        "message": vuln.get(
                            "summary", f"Vulnerable {module_name}@{module_version}"
                        ),
                        "tool": "dependency_checker",
                        "confidence": "high",
                    }
                )

        return findings

    def _check_ruby_dependencies(self, dependencies: dict[str, str], file_path: str) -> list[dict]:
        """Check Ruby gems for vulnerabilities."""
        findings = []

        for gem_name, gem_version in dependencies.items():
            vulnerabilities = self._query_vulnerability_db("rubygems", gem_name, gem_version)
            for vuln in vulnerabilities:
                findings.append(
                    {
                        "type": "dependency",
                        "vulnerability_type": "vulnerable_dependency",
                        "severity": vuln["severity"],
                        "package": gem_name,
                        "version": gem_version,
                        "file": file_path,
                        "cve": vuln.get("cve", ""),
                        "message": vuln.get("summary", f"Vulnerable {gem_name} {gem_version}"),
                        "tool": "dependency_checker",
                        "confidence": "high",
                    }
                )

        return findings

    def _query_vulnerability_db(
        self, ecosystem: str, package: str, package_version: str
    ) -> list[dict]:
        """Query vulnerability database for package vulnerabilities.

        This is a simplified implementation. In production, you would:
        1. Use the GitHub Advisory Database API
        2. Cache results to avoid rate limiting
        3. Handle version ranges properly
        """
        # Check cache first
        cache_key = f"{ecosystem}:{package}:{package_version}"
        if cache_key in self.vulnerability_cache:
            return self.vulnerability_cache[cache_key]

        vulnerabilities = []

        # In a real implementation, you would query:
        # https://api.github.com/advisories
        # or use tools like:
        # - safety (Python)
        # - npm audit (Node.js)
        # - bundler-audit (Ruby)
        # - nancy (Go)

        # For now, return empty list (no vulnerabilities found)
        # This is where you'd integrate with actual vulnerability databases

        # Example of what would be returned:
        # vulnerabilities = [{
        #     "severity": "high",
        #     "cve": "CVE-2021-12345",
        #     "summary": "Remote code execution vulnerability",
        #     "affected_versions": "< 1.2.3",
        #     "patched_versions": ">= 1.2.3"
        # }]

        # Cache the result
        self.vulnerability_cache[cache_key] = vulnerabilities

        return vulnerabilities
