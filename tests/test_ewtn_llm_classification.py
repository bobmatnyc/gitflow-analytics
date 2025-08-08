#!/usr/bin/env python3
"""Test LLM classification with EWTN commits."""

import json
from pathlib import Path

# Sample EWTN commits that were classified as "other"
EWTN_TEST_COMMITS = [
    # Should be "feature"
    "RMVP-876 Update episode timezones",
    "CNA-6078 Add support for Portuguese translations",
    "SITE-4238 Update navigation section with new menu items",
    
    # Should be "bugfix"
    "RMVP-1008 Fix episode page not loading",
    "CNA-6543 Resolve translation error in footer",
    
    # Should be "maintenance"
    "Update dependencies to latest versions",
    "Clean up unused imports",
    "Refactor authentication flow",
    
    # Should be "integration"
    "Configure PostHog analytics tracking",
    "Update Auth0 integration settings",
    "Add Iubenda cookie consent",
    
    # Should be "content"
    "Update prayer text for daily devotions",
    "Add new homily for Sunday mass",
    "Update saint biography content",
    
    # Should be "media"
    "Update Roku app video player",
    "Configure live streaming for special events",
    "Add new episode to program schedule",
    
    # Should be "localization"
    "Add Portuguese translation for homepage",
    "Translate navigation menu to Spanish",
    "Update French language pack",
    
    # Git artifacts (should be filtered out)
    "Co-authored-by: John Doe <john@example.com>",
    "Signed-off-by: Jane Smith <jane@example.com>",
    "",
    "..."
]

EXPECTED_CATEGORIES = [
    "feature", "feature", "feature",
    "bugfix", "bugfix",
    "maintenance", "maintenance", "maintenance",
    "integration", "integration", "integration",
    "content", "content", "content",
    "media", "media", "media",
    "localization", "localization", "localization",
    None, None, None, None  # Filtered out
]

def test_llm_classification():
    """Test LLM classification accuracy."""
    try:
        from gitflow_analytics.qualitative.classifiers.llm_commit_classifier import LLMCommitClassifier
        from gitflow_analytics.extractors.tickets import filter_git_artifacts
        
        # Initialize classifier (will use cached results if available)
        classifier = LLMCommitClassifier(
            api_key="test",  # Will use cache
            model="mistralai/mistral-7b-instruct",
            cache_dir=Path("/tmp/test_llm_cache")
        )
        
        correct = 0
        total = 0
        results = []
        
        for commit_msg, expected in zip(EWTN_TEST_COMMITS, EXPECTED_CATEGORIES):
            # Filter git artifacts
            filtered_msg = filter_git_artifacts(commit_msg)
            
            if not filtered_msg or filtered_msg.strip() == "":
                actual = None
            else:
                # Classify
                result = classifier.classify_commit(filtered_msg)
                actual = result['category']
            
            is_correct = actual == expected
            if expected is not None:  # Don't count filtered commits
                total += 1
                if is_correct:
                    correct += 1
            
            results.append({
                'commit': commit_msg[:50],
                'expected': expected,
                'actual': actual,
                'correct': is_correct
            })
            
            if not is_correct:
                print(f"❌ Mismatch: '{commit_msg[:50]}...' -> Expected: {expected}, Got: {actual}")
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print("\n" + "="*60)
        print(f"LLM Classification Test Results")
        print("="*60)
        print(f"✅ Correct: {correct}/{total}")
        print(f"📊 Accuracy: {accuracy:.1f}%")
        print(f"🎯 Target: >85%")
        print(f"{'✅ PASSED' if accuracy >= 85 else '❌ FAILED'}")
        
        # Cost estimation
        tokens_per_commit = 50  # Approximate
        total_tokens = len([c for c in EWTN_TEST_COMMITS if c]) * tokens_per_commit
        cost_per_million = 0.20  # Mistral pricing
        estimated_cost = (total_tokens / 1_000_000) * cost_per_million
        
        print(f"\n💰 Cost Analysis:")
        print(f"   Tokens used: ~{total_tokens:,}")
        print(f"   Estimated cost: ${estimated_cost:.4f}")
        print(f"   Cost per 1000 commits: ${estimated_cost / len(EWTN_TEST_COMMITS) * 1000:.2f}")
        
        # Save results
        output_file = Path("/tmp/ewtn_llm_test_results.json")
        with open(output_file, 'w') as f:
            json.dump({
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'results': results,
                'cost_estimate': {
                    'tokens': total_tokens,
                    'cost_usd': estimated_cost,
                    'per_1000_commits': estimated_cost / len(EWTN_TEST_COMMITS) * 1000
                }
            }, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: {output_file}")
        
        return accuracy >= 85
        
    except ImportError as e:
        print(f"⚠️  Cannot test - module not installed: {e}")
        print("Run: pipx install /Users/masa/Projects/managed/gitflow-analytics")
        return False

if __name__ == "__main__":
    import sys
    success = test_llm_classification()
    sys.exit(0 if success else 1)