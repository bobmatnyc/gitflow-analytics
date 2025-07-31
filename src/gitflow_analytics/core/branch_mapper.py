"""Map git branches to projects based on naming conventions."""

import re
from pathlib import Path
from typing import Optional


class BranchToProjectMapper:
    """Maps git branches to project keys based on conventions."""

    def __init__(self, mapping_rules: Optional[dict[str, list[str]]] = None):
        """
        Initialize with custom mapping rules.

        Args:
            mapping_rules: Dict mapping project keys to list of branch patterns
                          e.g., {'FRONTEND': ['feature/fe-*', 'frontend/*']}
        """
        self.mapping_rules = mapping_rules or self._get_default_rules()
        self.compiled_rules = self._compile_patterns()

    def _get_default_rules(self) -> dict[str, list[str]]:
        """Get default branch mapping rules."""
        return {
            "FRONTEND": [
                r"^feature/fe[-/_]",
                r"^feature/frontend[-/_]",
                r"^frontend/",
                r"^fe/",
                r"[-/_]frontend[-/_]",
                r"[-/_]fe[-/_]",
                r"[-/_]ui[-/_]",
                r"[-/_]web[-/_]",
            ],
            "BACKEND": [
                r"^feature/be[-/_]",
                r"^feature/backend[-/_]",
                r"^backend/",
                r"^be/",
                r"^api/",
                r"[-/_]backend[-/_]",
                r"[-/_]be[-/_]",
                r"[-/_]api[-/_]",
                r"[-/_]server[-/_]",
            ],
            "SERVICE": [
                r"^feature/service[-/_]",
                r"^feature/svc[-/_]",
                r"^service/",
                r"^svc/",
                r"[-/_]service[-/_]",
                r"[-/_]svc[-/_]",
                r"[-/_]microservice[-/_]",
            ],
            "MOBILE": [
                r"^feature/mobile[-/_]",
                r"^feature/app[-/_]",
                r"^mobile/",
                r"^app/",
                r"^ios/",
                r"^android/",
                r"[-/_]mobile[-/_]",
                r"[-/_]app[-/_]",
                r"[-/_]ios[-/_]",
                r"[-/_]android[-/_]",
            ],
            "DATA": [
                r"^feature/data[-/_]",
                r"^feature/etl[-/_]",
                r"^data/",
                r"^etl/",
                r"^pipeline/",
                r"[-/_]data[-/_]",
                r"[-/_]etl[-/_]",
                r"[-/_]pipeline[-/_]",
                r"[-/_]analytics[-/_]",
            ],
            "INFRA": [
                r"^feature/infra[-/_]",
                r"^feature/devops[-/_]",
                r"^infra/",
                r"^devops/",
                r"^ops/",
                r"[-/_]infra[-/_]",
                r"[-/_]devops[-/_]",
                r"[-/_]ops[-/_]",
                r"[-/_]deployment[-/_]",
            ],
            "SCRAPER": [
                r"^feature/scraper[-/_]",
                r"^feature/crawler[-/_]",
                r"^scraper/",
                r"^crawler/",
                r"[-/_]scraper[-/_]",
                r"[-/_]crawler[-/_]",
                r"[-/_]scraping[-/_]",
            ],
        }

    def _compile_patterns(self) -> dict[str, list[re.Pattern]]:
        """Compile regex patterns for efficiency."""
        compiled = {}
        for project, patterns in self.mapping_rules.items():
            compiled[project] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        return compiled

    def map_branch_to_project(self, branch_name: str, repo_path: Optional[Path] = None) -> str:
        """
        Map a branch name to a project key.

        Args:
            branch_name: Git branch name
            repo_path: Optional repository path for context

        Returns:
            Project key or 'UNKNOWN'
        """
        if not branch_name or branch_name in ["main", "master", "develop", "development"]:
            # Try to infer from repo path if available
            if repo_path:
                return self._infer_from_repo_path(repo_path)
            return "UNKNOWN"

        # Check against compiled patterns
        for project, patterns in self.compiled_rules.items():
            for pattern in patterns:
                if pattern.search(branch_name):
                    return project

        # Try to extract from ticket references in branch name
        ticket_project = self._extract_from_ticket(branch_name)
        if ticket_project:
            return ticket_project

        # Try to infer from repo path if available
        if repo_path:
            return self._infer_from_repo_path(repo_path)

        return "UNKNOWN"

    def _extract_from_ticket(self, branch_name: str) -> Optional[str]:
        """Extract project from ticket reference in branch name."""
        # Common ticket patterns
        ticket_patterns = [
            r"([A-Z]{2,})-\d+",  # JIRA style: PROJ-123
            r"#([A-Z]{2,})\d+",  # Hash prefix: #PROJ123
            r"([A-Z]{2,})_\d+",  # Underscore: PROJ_123
        ]

        for pattern in ticket_patterns:
            match = re.search(pattern, branch_name, re.IGNORECASE)
            if match:
                prefix = match.group(1).upper()
                # Map common prefixes to projects
                prefix_map = {
                    "FE": "FRONTEND",
                    "BE": "BACKEND",
                    "SVC": "SERVICE",
                    "MOB": "MOBILE",
                    "DATA": "DATA",
                    "ETL": "DATA",
                    "INFRA": "INFRA",
                    "OPS": "INFRA",
                    "SCRAPE": "SCRAPER",
                    "CRAWL": "SCRAPER",
                }

                if prefix in prefix_map:
                    return prefix_map[prefix]

                # Check if prefix matches any project key
                for project in self.mapping_rules:
                    if prefix == project or prefix in project:
                        return project

        return None

    def _infer_from_repo_path(self, repo_path: Path) -> str:
        """Infer project from repository path."""
        repo_name = repo_path.name.lower()

        # Direct mapping
        path_map = {
            "frontend": "FRONTEND",
            "backend": "BACKEND",
            "service": "SERVICE",
            "service-ts": "SERVICE_TS",
            "services": "SERVICES",
            "mobile": "MOBILE",
            "ios": "MOBILE",
            "android": "MOBILE",
            "data": "DATA",
            "etl": "DATA",
            "infra": "INFRA",
            "infrastructure": "INFRA",
            "scraper": "SCRAPER",
            "crawler": "SCRAPER",
            "scrapers": "SCRAPER",
        }

        for key, project in path_map.items():
            if key in repo_name:
                return project

        # Check parent directory
        if repo_path.parent.name.lower() in path_map:
            return path_map[repo_path.parent.name.lower()]

        return "UNKNOWN"

    def add_mapping_rule(self, project: str, patterns: list[str]) -> None:
        """Add custom mapping rules for a project."""
        if project not in self.mapping_rules:
            self.mapping_rules[project] = []

        self.mapping_rules[project].extend(patterns)

        # Recompile patterns
        self.compiled_rules[project] = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.mapping_rules[project]
        ]
