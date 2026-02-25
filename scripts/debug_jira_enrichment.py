#!/usr/bin/env python3
"""Debug JIRA enrichment process."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")


def test_jira_enrichment():
    """Test JIRA enrichment with sample commit data."""
    from gitflow_analytics.core.cache import GitAnalysisCache
    from gitflow_analytics.integrations.jira_integration import JIRAIntegration

    # Initialize cache
    cache_dir = Path("./configs/test-ewtn-cache")
    cache = GitAnalysisCache(cache_dir)

    # Initialize JIRA integration
    base_url = "https://ewtn.atlassian.net"
    username = os.getenv("JIRA_ACCESS_USER")
    api_token = os.getenv("JIRA_ACCESS_TOKEN")
    story_point_fields = ["customfield_10016", "customfield_10063"]

    jira_integration = JIRAIntegration(
        base_url=base_url,
        username=username,
        api_token=api_token,
        cache=cache,
        story_point_fields=story_point_fields,
    )

    # Create sample commits with ticket references (matching what we see in DB)
    sample_commits = [
        {
            "hash": "da7d3a97",
            "message": "RMVP-1066 incorrectly flagged itemable fields as reorderable",
            "ticket_references": ["RMVP-1066", "RMVP-1066"],  # Already parsed as list
            "story_points": 0,  # Should be updated to 4
        },
        {
            "hash": "e9cd0994",
            "message": "RMVP-1066 missed committing change (match Roku tab id character case)",
            "ticket_references": ["RMVP-1066", "RMVP-1066"],  # Already parsed as list
            "story_points": 0,  # Should be updated to 4
        },
    ]

    print("🔍 Testing JIRA Enrichment Process")
    print("=" * 50)

    print("\n📋 Sample commits before enrichment:")
    for commit in sample_commits:
        print(
            f"   {commit['hash']}: {commit['story_points']} points, tickets: {commit['ticket_references']}"
        )

    print("\n🎫 Checking JIRA cache for RMVP-1066...")
    with cache.get_session() as session:
        from gitflow_analytics.models.database import IssueCache

        cached_ticket = (
            session.query(IssueCache)
            .filter(IssueCache.platform == "jira", IssueCache.issue_id == "RMVP-1066")
            .first()
        )

        if cached_ticket:
            print(
                f"   ✅ Found in cache: {cached_ticket.issue_id} = {cached_ticket.story_points} points"
            )
        else:
            print("   ❌ Not found in cache")

    print("\n🔧 Running JIRA enrichment...")
    try:
        jira_integration.enrich_commits_with_jira_data(sample_commits)
        print("   ✅ Enrichment completed without errors")
    except Exception as e:
        print(f"   ❌ Enrichment failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n📊 Sample commits after enrichment:")
    for commit in sample_commits:
        print(
            f"   {commit['hash']}: {commit['story_points']} points, tickets: {commit['ticket_references']}"
        )

    # Check if story points were assigned
    total_points = sum(commit.get("story_points", 0) for commit in sample_commits)
    print(f"\n📈 Total story points assigned: {total_points}")

    if total_points > 0:
        print("   ✅ SUCCESS: Story points were assigned!")
    else:
        print("   ❌ FAILURE: No story points were assigned")

        # Debug the _fetch_tickets_batch method
        print("\n🔍 Debugging _fetch_tickets_batch...")
        ticket_data = jira_integration._fetch_tickets_batch(["RMVP-1066"])
        print(f"   Ticket data returned: {ticket_data}")


def main():
    """Debug JIRA enrichment."""
    if not os.getenv("JIRA_ACCESS_USER") or not os.getenv("JIRA_ACCESS_TOKEN"):
        print("❌ JIRA credentials not found!")
        print("   Set JIRA_ACCESS_USER and JIRA_ACCESS_TOKEN environment variables")
        return

    test_jira_enrichment()


if __name__ == "__main__":
    main()
