#!/usr/bin/env python3
"""
Test script to verify Story Points analysis functionality.

This script will:
1. Verify story points extraction is a phase 2 analysis
2. Test JIRA integration for story points
3. Check if story points are correctly identified in tickets
4. Validate the story points extraction process
"""

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

# Add the gitflow-analytics package to the path
sys.path.insert(0, "/Users/masa/Projects/managed/gitflow-analytics/src")

try:
    from gitflow_analytics.config import Config
    from gitflow_analytics.extractors.story_points import StoryPointExtractor
    from gitflow_analytics.extractors.tickets import TicketExtractor
    from gitflow_analytics.integrations.jira_integration import JIRAIntegration
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class StoryPointsTester:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = Config.from_file(config_path)
        self.cache_dir = Path("/Users/masa/Clients/EWTN/gfa/.gitflow-cache")

        # Get JIRA credentials from environment
        self.jira_user = os.getenv("JIRA_ACCESS_USER")
        self.jira_token = os.getenv("JIRA_ACCESS_TOKEN")
        self.jira_base_url = "https://ewtn.atlassian.net"

        # Story point fields from config
        self.story_point_fields = [
            "customfield_10063",  # Story Points (primary)
            "customfield_10016",  # Story point estimate (backup)
            "timeestimate",  # Remaining Estimate
            "timeoriginalestimate",  # Original estimate
        ]

        self.results = {
            "phase_verification": {},
            "jira_connectivity": {},
            "ticket_analysis": {},
            "story_points_extraction": {},
        }

    def test_phase_verification(self):
        """Verify that story points analysis happens in phase 2."""
        print("🔍 Testing Phase Verification...")

        # Check if story points extractor is initialized during commit analysis
        try:
            from gitflow_analytics.core.analyzer import GitAnalyzer

            analyzer = GitAnalyzer(
                cache_dir=self.cache_dir,
                identity_resolver=None,
                story_point_extractor=None,
                ticket_extractor=None,
            )

            # Check if story point extractor exists
            has_story_point_extractor = hasattr(analyzer, "story_point_extractor")

            self.results["phase_verification"] = {
                "has_story_point_extractor": has_story_point_extractor,
                "phase": (
                    "Phase 1 (commit analysis)" if has_story_point_extractor else "Not in Phase 1"
                ),
            }

            print(f"✅ Story point extractor in GitAnalyzer: {has_story_point_extractor}")

        except Exception as e:
            print(f"❌ Error testing phase verification: {e}")
            self.results["phase_verification"] = {"error": str(e)}

    def test_jira_connectivity(self):
        """Test JIRA connectivity and field configuration."""
        print("🔍 Testing JIRA Connectivity...")

        if not self.jira_user or not self.jira_token:
            print("❌ JIRA credentials not found in environment")
            self.results["jira_connectivity"] = {
                "connected": False,
                "error": "Missing JIRA credentials",
            }
            return

        try:
            # Test basic JIRA connectivity
            auth = HTTPBasicAuth(self.jira_user, self.jira_token)
            response = requests.get(
                f"{self.jira_base_url}/rest/api/2/serverInfo", auth=auth, timeout=10
            )

            if response.status_code == 200:
                server_info = response.json()
                print(f"✅ JIRA connected: {server_info.get('serverTitle', 'Unknown')}")

                # Test field information
                fields_response = requests.get(
                    f"{self.jira_base_url}/rest/api/2/field", auth=auth, timeout=10
                )

                if fields_response.status_code == 200:
                    fields = fields_response.json()
                    story_point_field_info = {}

                    for field in fields:
                        field_id = field.get("id", "")
                        if field_id in self.story_point_fields:
                            story_point_field_info[field_id] = {
                                "name": field.get("name", ""),
                                "custom": field.get("custom", False),
                                "schema": field.get("schema", {}),
                            }

                    self.results["jira_connectivity"] = {
                        "connected": True,
                        "server_title": server_info.get("serverTitle"),
                        "story_point_fields_found": story_point_field_info,
                    }

                    print(f"✅ Found {len(story_point_field_info)} configured story point fields")
                    for field_id, info in story_point_field_info.items():
                        print(f"   - {field_id}: {info['name']}")
                else:
                    print(f"⚠️  Could not fetch field information: {fields_response.status_code}")
                    self.results["jira_connectivity"] = {
                        "connected": True,
                        "server_title": server_info.get("serverTitle"),
                        "fields_error": f"HTTP {fields_response.status_code}",
                    }
            else:
                print(f"❌ JIRA connection failed: HTTP {response.status_code}")
                self.results["jira_connectivity"] = {
                    "connected": False,
                    "error": f"HTTP {response.status_code}",
                }

        except Exception as e:
            print(f"❌ JIRA connectivity error: {e}")
            self.results["jira_connectivity"] = {"connected": False, "error": str(e)}

    def test_ticket_analysis(self):
        """Analyze tickets found in commits and check for story points."""
        print("🔍 Testing Ticket Analysis...")

        try:
            # Get sample tickets from database
            conn = sqlite3.connect(str(self.cache_dir / "gitflow_cache.db"))
            cursor = conn.execute(
                """
                SELECT DISTINCT ticket_references 
                FROM cached_commits 
                WHERE ticket_references IS NOT NULL 
                AND ticket_references != '[]' 
                LIMIT 20
            """
            )

            ticket_refs = []
            for row in cursor.fetchall():
                try:
                    refs = json.loads(row[0])
                    ticket_refs.extend(refs)
                except:
                    continue

            conn.close()

            # Extract unique JIRA tickets
            jira_tickets = set()
            for ref in ticket_refs:
                if isinstance(ref, str) and "-" in ref and ref.split("-")[0].isalpha():
                    jira_tickets.add(ref)

            print(f"✅ Found {len(jira_tickets)} unique JIRA tickets in commits")

            # Test a few tickets for story points
            sample_tickets = list(jira_tickets)[:5]
            ticket_analysis = {}

            if self.jira_user and self.jira_token:
                auth = HTTPBasicAuth(self.jira_user, self.jira_token)

                for ticket_id in sample_tickets:
                    try:
                        response = requests.get(
                            f"{self.jira_base_url}/rest/api/2/issue/{ticket_id}",
                            auth=auth,
                            timeout=10,
                        )

                        if response.status_code == 200:
                            issue = response.json()
                            fields = issue.get("fields", {})

                            # Check for story points in configured fields
                            story_points = None
                            story_point_field_used = None

                            for field_id in self.story_point_fields:
                                if field_id in fields and fields[field_id] is not None:
                                    try:
                                        story_points = float(fields[field_id])
                                        story_point_field_used = field_id
                                        break
                                    except (ValueError, TypeError):
                                        continue

                            ticket_analysis[ticket_id] = {
                                "exists": True,
                                "story_points": story_points,
                                "story_point_field": story_point_field_used,
                                "issue_type": fields.get("issuetype", {}).get("name"),
                                "status": fields.get("status", {}).get("name"),
                                "summary": fields.get("summary", "")[:100],
                            }

                        else:
                            ticket_analysis[ticket_id] = {
                                "exists": False,
                                "error": f"HTTP {response.status_code}",
                            }

                    except Exception as e:
                        ticket_analysis[ticket_id] = {"exists": False, "error": str(e)}

            self.results["ticket_analysis"] = {
                "total_jira_tickets": len(jira_tickets),
                "sample_tickets": ticket_analysis,
                "tickets_with_story_points": len(
                    [t for t in ticket_analysis.values() if t.get("story_points") is not None]
                ),
            }

            # Print results
            tickets_with_points = [
                tid for tid, info in ticket_analysis.items() if info.get("story_points") is not None
            ]

            print(f"✅ Analyzed {len(ticket_analysis)} sample tickets")
            print(f"✅ Found {len(tickets_with_points)} tickets with story points")

            for ticket_id in tickets_with_points:
                info = ticket_analysis[ticket_id]
                print(
                    f"   - {ticket_id}: {info['story_points']} points ({info['story_point_field']})"
                )

        except Exception as e:
            print(f"❌ Error in ticket analysis: {e}")
            self.results["ticket_analysis"] = {"error": str(e)}

    def test_story_points_extraction(self):
        """Test the story points extraction process."""
        print("🔍 Testing Story Points Extraction...")

        try:
            # Test text-based extraction
            sp_extractor = StoryPointExtractor()

            test_texts = [
                "RMVP-1030: Fix login issue [3 points]",
                "Story Points: 5 - Update user interface",
                "SP: 8 - Refactor authentication module",
                "Points: 2 - Bug fix for payment processing",
                "RMVP-1075 (5 story points) - New feature implementation",
            ]

            extraction_results = {}
            for text in test_texts:
                points = sp_extractor.extract_from_text(text)
                extraction_results[text] = points

            # Test JIRA integration extraction
            jira_integration = None
            if self.jira_user and self.jira_token:
                try:
                    jira_integration = JIRAIntegration(
                        base_url=self.jira_base_url,
                        username=self.jira_user,
                        api_token=self.jira_token,
                        story_point_fields=self.story_point_fields,
                    )
                    print("✅ JIRA integration initialized successfully")
                except Exception as e:
                    print(f"⚠️  JIRA integration initialization failed: {e}")

            self.results["story_points_extraction"] = {
                "text_extraction_results": extraction_results,
                "jira_integration_available": jira_integration is not None,
                "successful_extractions": len(
                    [p for p in extraction_results.values() if p is not None]
                ),
            }

            print(
                f"✅ Text extraction test: {len([p for p in extraction_results.values() if p is not None])}/{len(test_texts)} successful"
            )

            for text, points in extraction_results.items():
                if points:
                    print(f"   - '{text[:50]}...' → {points} points")

        except Exception as e:
            print(f"❌ Error in story points extraction test: {e}")
            self.results["story_points_extraction"] = {"error": str(e)}

    def generate_report(self):
        """Generate a comprehensive test report."""
        print("📝 Generating Story Points Analysis Report...")

        report_path = Path("story_points_analysis_report.md")

        with open(report_path, "w") as f:
            f.write("# Story Points Analysis Test Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Config**: {self.config_path}\n\n")

            # Phase Verification
            f.write("## Phase Verification\n\n")
            phase_info = self.results.get("phase_verification", {})
            if "error" in phase_info:
                f.write(f"❌ **Error**: {phase_info['error']}\n\n")
            else:
                f.write(
                    f"- **Story Point Extractor Available**: {phase_info.get('has_story_point_extractor', False)}\n"
                )
                f.write(f"- **Analysis Phase**: {phase_info.get('phase', 'Unknown')}\n\n")

            # JIRA Connectivity
            f.write("## JIRA Connectivity\n\n")
            jira_info = self.results.get("jira_connectivity", {})
            if jira_info.get("connected", False):
                f.write(f"✅ **Connected**: {jira_info.get('server_title', 'Unknown')}\n")

                fields_found = jira_info.get("story_point_fields_found", {})
                if fields_found:
                    f.write(f"- **Story Point Fields Found**: {len(fields_found)}\n")
                    for field_id, info in fields_found.items():
                        f.write(f"  - `{field_id}`: {info['name']}\n")
                else:
                    f.write("- **Story Point Fields**: None found or error accessing fields\n")
            else:
                f.write(f"❌ **Connection Failed**: {jira_info.get('error', 'Unknown error')}\n")
            f.write("\n")

            # Ticket Analysis
            f.write("## Ticket Analysis\n\n")
            ticket_info = self.results.get("ticket_analysis", {})
            if "error" in ticket_info:
                f.write(f"❌ **Error**: {ticket_info['error']}\n\n")
            else:
                f.write(
                    f"- **Total JIRA Tickets Found**: {ticket_info.get('total_jira_tickets', 0)}\n"
                )
                f.write(
                    f"- **Sample Tickets Analyzed**: {len(ticket_info.get('sample_tickets', {}))}\n"
                )
                f.write(
                    f"- **Tickets with Story Points**: {ticket_info.get('tickets_with_story_points', 0)}\n\n"
                )

                sample_tickets = ticket_info.get("sample_tickets", {})
                if sample_tickets:
                    f.write("### Sample Ticket Details\n\n")
                    for ticket_id, info in sample_tickets.items():
                        if info.get("exists", False):
                            f.write(f"**{ticket_id}**\n")
                            f.write(f"- Story Points: {info.get('story_points', 'None')}\n")
                            f.write(f"- Field Used: {info.get('story_point_field', 'None')}\n")
                            f.write(f"- Type: {info.get('issue_type', 'Unknown')}\n")
                            f.write(f"- Status: {info.get('status', 'Unknown')}\n")
                            f.write(f"- Summary: {info.get('summary', 'No summary')}\n\n")
                        else:
                            f.write(
                                f"**{ticket_id}**: ❌ {info.get('error', 'Not accessible')}\n\n"
                            )

            # Story Points Extraction
            f.write("## Story Points Extraction\n\n")
            extraction_info = self.results.get("story_points_extraction", {})
            if "error" in extraction_info:
                f.write(f"❌ **Error**: {extraction_info['error']}\n\n")
            else:
                f.write(
                    f"- **JIRA Integration Available**: {extraction_info.get('jira_integration_available', False)}\n"
                )
                f.write(
                    f"- **Text Extraction Success Rate**: {extraction_info.get('successful_extractions', 0)}/5\n\n"
                )

                text_results = extraction_info.get("text_extraction_results", {})
                if text_results:
                    f.write("### Text Extraction Test Results\n\n")
                    for text, points in text_results.items():
                        status = "✅" if points else "❌"
                        f.write(f"{status} `{text}` → {points} points\n")
                    f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            # Check if story points are being extracted
            if ticket_info.get("tickets_with_story_points", 0) > 0:
                f.write("✅ **Story points are available in JIRA tickets**\n")
                f.write(
                    "- The system should be able to extract story points during phase 2 analysis\n"
                )
                f.write("- Consider running analysis with JIRA integration enabled\n\n")
            else:
                f.write("⚠️  **No story points found in sample tickets**\n")
                f.write("- Check if story points are being set in JIRA tickets\n")
                f.write("- Verify the correct story point field IDs in configuration\n")
                f.write("- Consider adding story points to tickets for better analysis\n\n")

            if not jira_info.get("connected", False):
                f.write("❌ **JIRA integration not working**\n")
                f.write("- Check JIRA credentials in environment variables\n")
                f.write("- Verify JIRA base URL configuration\n")
                f.write("- Ensure network connectivity to JIRA instance\n\n")

        print(f"✅ Report generated: {report_path}")

    def run_test(self):
        """Run the complete story points analysis test."""
        print("🚀 Starting Story Points Analysis Test...")
        print()

        # Step 1: Phase verification
        self.test_phase_verification()

        # Step 2: JIRA connectivity
        self.test_jira_connectivity()

        # Step 3: Ticket analysis
        self.test_ticket_analysis()

        # Step 4: Story points extraction
        self.test_story_points_extraction()

        # Step 5: Generate report
        self.generate_report()

        print("\n🎉 Story Points Analysis Test complete!")


if __name__ == "__main__":
    config_path = "/Users/masa/Clients/EWTN/gfa/config.yaml"

    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    tester = StoryPointsTester(config_path)
    tester.run_test()
