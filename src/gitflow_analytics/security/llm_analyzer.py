"""LLM-based security analysis for comprehensive code review."""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class LLMSecurityAnalyzer:
    """Use LLM to analyze code changes for security issues that tools might miss."""

    def __init__(self, config: Any, cache_dir: Optional[Path] = None):
        """Initialize LLM security analyzer.

        Args:
            config: LLM security configuration
            cache_dir: Directory for caching LLM responses
        """
        self.config = config
        self.api_key = (
            config.api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = config.model
        self.cache_dir = cache_dir or Path(".gitflow-cache/llm_security")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache LLM responses for 7 days to save costs
        self.cache_ttl = timedelta(days=7)

    def analyze_commit(self, commit_data: dict) -> list[dict]:
        """Analyze a commit for security issues using LLM.

        Args:
            commit_data: Commit data with message, files_changed, etc.

        Returns:
            List of security findings
        """
        if not self.api_key:
            logger.debug("LLM API key not configured, skipping LLM security analysis")
            return []

        findings = []

        # Check cache first
        cache_key = self._get_cache_key(commit_data)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        try:
            # Analyze commit message and metadata
            commit_findings = self._analyze_commit_message(commit_data)
            findings.extend(commit_findings)

            # Analyze code changes if available
            if "diff_content" in commit_data:
                code_findings = self._analyze_code_changes(commit_data)
                findings.extend(code_findings)

            # Cache the results
            self._cache_result(cache_key, findings)

        except Exception as e:
            logger.warning(f"Error in LLM security analysis: {e}")

        return findings

    def _analyze_commit_message(self, commit_data: dict) -> list[dict]:
        """Analyze commit message for security implications."""
        prompt = self.config.commit_review_prompt.format(
            message=commit_data.get("message", ""),
            files=", ".join(commit_data.get("files_changed", [])),
            category=commit_data.get("category", "unknown"),
        )

        response = self._call_llm(prompt)
        return self._parse_llm_response(response, commit_data)

    def _analyze_code_changes(self, commit_data: dict) -> list[dict]:
        """Analyze actual code changes for security issues."""
        # Limit the amount of code sent to LLM for cost control
        lines_added = commit_data.get("diff_content", "")
        if len(lines_added.split("\n")) > self.config.max_lines_for_llm:
            lines_added = "\n".join(lines_added.split("\n")[: self.config.max_lines_for_llm])
            lines_added += "\n... (truncated for analysis)"

        prompt = self.config.code_review_prompt.format(
            files_changed=", ".join(commit_data.get("files_changed", [])), lines_added=lines_added
        )

        response = self._call_llm(prompt)
        return self._parse_llm_response(response, commit_data, is_code_analysis=True)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API with the given prompt."""
        if self.model.startswith("claude"):
            return self._call_anthropic(prompt)
        else:
            return self._call_openrouter(prompt)

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic's Claude API."""
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            data = {
                "model": self.model,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,  # Low temperature for consistent analysis
            }

            with httpx.Client() as client:
                response = client.post(
                    "https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=30
                )

            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            else:
                logger.warning(f"Claude API error: {response.status_code}")
                return ""

        except Exception as e:
            logger.warning(f"Error calling Claude API: {e}")
            return ""

    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API for various LLM models."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a security expert analyzing code for vulnerabilities. Be concise and specific.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 500,
                "temperature": 0.1,
            }

            with httpx.Client() as client:
                response = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30,
                )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.warning(f"OpenRouter API error: {response.status_code}")
                return ""

        except Exception as e:
            logger.warning(f"Error calling OpenRouter API: {e}")
            return ""

    def _parse_llm_response(
        self, response: str, commit_data: dict, is_code_analysis: bool = False
    ) -> list[dict]:
        """Parse LLM response and extract security findings."""
        findings = []

        if not response or "no security issues" in response.lower():
            return findings

        # Extract specific security concerns from the response
        security_keywords = {
            "authentication": ("high", "authentication"),
            "authorization": ("high", "authorization"),
            "injection": ("critical", "injection"),
            "sql": ("critical", "sql_injection"),
            "xss": ("high", "xss"),
            "csrf": ("high", "csrf"),
            "exposure": ("high", "data_exposure"),
            "credential": ("critical", "credential_exposure"),
            "secret": ("critical", "secret_exposure"),
            "crypto": ("high", "weak_cryptography"),
            "validation": ("medium", "input_validation"),
            "sanitization": ("medium", "input_sanitization"),
            "permission": ("high", "permission_issue"),
            "privilege": ("high", "privilege_escalation"),
            "buffer": ("critical", "buffer_overflow"),
            "race": ("high", "race_condition"),
            "session": ("high", "session_management"),
            "cookie": ("medium", "cookie_security"),
            "cors": ("medium", "cors_misconfiguration"),
            "encryption": ("high", "encryption_issue"),
        }

        # Check for security keywords in the response
        response_lower = response.lower()
        found_issues = []

        for keyword, (severity, issue_type) in security_keywords.items():
            if keyword in response_lower:
                found_issues.append((severity, issue_type))

        # Create findings based on detected issues
        if found_issues:
            # Extract the most severe issue
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            found_issues.sort(key=lambda x: severity_order.get(x[0], 999))

            finding = {
                "type": "security",
                "source": "llm_analysis",
                "vulnerability_type": found_issues[0][1],
                "severity": found_issues[0][0],
                "commit": commit_data.get("commit_hash_short", "unknown"),
                "message": self._extract_finding_message(response),
                "confidence": self._calculate_confidence(response),
                "analysis_type": "code" if is_code_analysis else "commit",
                "files": commit_data.get("files_changed", []),
            }

            findings.append(finding)

        return findings

    def _extract_finding_message(self, response: str) -> str:
        """Extract a concise finding message from LLM response."""
        # Take the first meaningful sentence
        sentences = response.split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and not sentence.lower().startswith(("the", "this", "it")):
                return sentence + "."

        # Fallback to truncated response
        return response[:200] + "..." if len(response) > 200 else response

    def _calculate_confidence(self, response: str) -> str:
        """Calculate confidence level based on LLM response characteristics."""
        response_lower = response.lower()

        # High confidence indicators
        high_confidence_words = [
            "definitely",
            "clearly",
            "certain",
            "obvious",
            "critical",
            "severe",
        ]
        if any(word in response_lower for word in high_confidence_words):
            return "high"

        # Low confidence indicators
        low_confidence_words = ["might", "could", "possibly", "perhaps", "may", "potential"]
        if any(word in response_lower for word in low_confidence_words):
            return "medium"

        return "high" if len(response) > 100 else "medium"

    def _get_cache_key(self, commit_data: dict) -> str:
        """Generate cache key for commit data."""
        key_parts = [
            commit_data.get("commit_hash", ""),
            str(sorted(commit_data.get("files_changed", []))),
            commit_data.get("message", "")[:100],
        ]
        key_str = "|".join(key_parts)
        # Simple hash for filename
        import hashlib

        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _get_cached_result(self, cache_key: str) -> Optional[list[dict]]:
        """Get cached result if it exists and is not expired."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            # Check if cache is expired
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - file_time > self.cache_ttl:
                cache_file.unlink()  # Delete expired cache
                return None

            with open(cache_file) as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Error reading cache: {e}")
            return None

    def _cache_result(self, cache_key: str, result: list[dict]) -> None:
        """Cache the analysis result."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(result, f)
        except Exception as e:
            logger.debug(f"Error writing cache: {e}")

    def generate_security_insights(self, all_findings: list[dict]) -> str:
        """Generate high-level security insights from all findings."""
        if not all_findings:
            return "No security issues detected in the analyzed period."

        # Aggregate findings
        by_severity = {}
        by_type = {}

        for finding in all_findings:
            severity = finding.get("severity", "unknown")
            vuln_type = finding.get("vulnerability_type", "unknown")

            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_type[vuln_type] = by_type.get(vuln_type, 0) + 1

        # Generate insights prompt
        prompt = f"""Analyze these security findings and provide strategic recommendations:

Findings by severity: {json.dumps(by_severity, indent=2)}
Findings by type: {json.dumps(by_type, indent=2)}

Provide:
1. Top 3 security risks to address
2. Recommended security improvements
3. Security training needs for the team

Be concise and actionable."""

        response = self._call_llm(prompt)
        return response if response else "Unable to generate security insights."
