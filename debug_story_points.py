#!/usr/bin/env python3
"""Debug story point extraction."""

from src.gitflow_analytics.extractors.story_points import StoryPointExtractor


def test_story_point_extraction():
    """Test story point extraction on sample commit messages."""
    print("🎯 Testing Story Point Extraction")
    print("=" * 50)
    
    extractor = StoryPointExtractor()
    
    # Test samples that should match
    test_messages = [
        "RMVP-876 Update episode timezones",
        "CNA-6078 Add support for Portuguese translations",
        "SITE-4238 Update navigation section with new menu items",
        "feat: major enhancements to developer analytics and reporting",
        "fix: correct email addresses for identity resolution",
        "Story Points: 5 - Add new feature",
        "[3sp] Fix bug in authentication",
        "estimate: 8 points for database refactoring",
        "SP5 - Quick UI fix",
        "points: 13 - Major backend changes"
    ]
    
    print("Testing patterns:")
    for pattern in extractor.patterns:
        print(f"  {pattern.pattern}")
    
    print(f"\nTesting {len(test_messages)} sample messages:")
    print("-" * 50)
    
    found_any = False
    for i, message in enumerate(test_messages, 1):
        points = extractor.extract_from_text(message)
        status = f"✅ {points} SP" if points else "❌ None"
        print(f"{i:2d}. {message[:60]:<60} → {status}")
        if points:
            found_any = True
    
    print(f"\n📊 Summary: {'Found story points in some messages' if found_any else 'No story points found'}")
    
    # Test custom JIRA/ticket patterns
    print(f"\n🎫 Testing EWTN/JIRA-style patterns:")
    
    jira_messages = [
        "RMVP-876: Update episode timezones [3sp]",
        "CNA-6078: Add Portuguese support (estimate: 5)",
        "SITE-4238: Navigation updates SP8",
        "HOSANNA-123: Story points = 2",
        "EWTN-456: Backend changes [5 pts]"
    ]
    
    for i, message in enumerate(jira_messages, 1):
        points = extractor.extract_from_text(message)
        status = f"✅ {points} SP" if points else "❌ None"
        print(f"{i:2d}. {message:<50} → {status}")
        
    return found_any


if __name__ == "__main__":
    test_story_point_extraction()