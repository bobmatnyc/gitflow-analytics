#!/usr/bin/env python3
"""Debug and fix the line counting issue in git statistics."""

from pathlib import Path
import git
from git import Repo


def debug_diff_content():
    """Debug why diff.diff is returning empty content."""
    print("🔍 Debugging Diff Content Issue")
    print("=" * 50)
    
    repo_path = Path("/Users/masa/Projects/managed/gitflow-analytics")
    repo = Repo(repo_path)
    
    # Get a recent commit with known changes
    commits = list(repo.iter_commits('HEAD', max_count=5))
    
    for i, commit in enumerate(commits[:3], 1):
        print(f"\n#{i} Commit: {commit.hexsha[:8]}")
        print(f"   Message: {commit.message.strip()[:80]}...")
        
        # Get parent
        parent = commit.parents[0] if commit.parents else None
        
        # Test different methods of getting diff content
        try:
            print("   Testing diff methods:")
            
            # Method 1: commit.diff() (current problematic approach)
            diffs = commit.diff(parent)
            print(f"   - commit.diff(): {len(diffs)} diff objects")
            
            for j, diff in enumerate(diffs[:3]):  # Show first 3 files
                file_path = diff.b_path or diff.a_path or "unknown"
                print(f"     File {j+1}: {file_path}")
                print(f"     - diff.diff type: {type(diff.diff)}")
                print(f"     - diff.diff length: {len(diff.diff) if diff.diff else 'None'}")
                
                if diff.diff:
                    # Try to decode
                    try:
                        diff_text = diff.diff if isinstance(diff.diff, str) else diff.diff.decode("utf-8", errors="ignore")
                        lines = diff_text.split("\n")
                        plus_lines = [line for line in lines if line.startswith("+") and not line.startswith("+++")]
                        minus_lines = [line for line in lines if line.startswith("-") and not line.startswith("---")]
                        print(f"     - Manual count: +{len(plus_lines)}, -{len(minus_lines)}")
                        
                        # Show sample lines
                        if plus_lines:
                            print(f"     - Sample +: {plus_lines[0][:50]}...")
                        if minus_lines:
                            print(f"     - Sample -: {minus_lines[0][:50]}...")
                            
                    except Exception as e:
                        print(f"     - Decode error: {e}")
                else:
                    print("     - diff.diff is None/empty")
            
            # Method 2: Use git command directly
            try:
                if parent:
                    git_diff = repo.git.diff(parent.hexsha, commit.hexsha, '--numstat')
                    print(f"   - git diff --numstat output:")
                    for line in git_diff.split('\n')[:3]:  # Show first 3 lines
                        if line.strip():
                            parts = line.split('\t')
                            if len(parts) >= 3:
                                insertions, deletions, filename = parts
                                print(f"     {filename}: +{insertions}, -{deletions}")
            except Exception as e:
                print(f"   - git diff --numstat error: {e}")
            
            # Method 3: Check commit.stats again
            stats = commit.stats.total
            print(f"   - commit.stats.total: {stats}")
            
        except Exception as e:
            print(f"   Error processing commit: {e}")


def test_fixed_diff_parsing():
    """Test a corrected version of diff parsing."""
    print("\n\n🛠️ Testing Fixed Diff Parsing")
    print("=" * 50)
    
    def calculate_lines_correctly(commit, parent=None):
        """Correctly calculate insertions and deletions."""
        total_insertions = 0
        total_deletions = 0
        total_files = 0
        
        try:
            # Use git command for accurate line counts
            repo = commit.repo
            if parent:
                diff_output = repo.git.diff(parent.hexsha, commit.hexsha, '--numstat')
            else:
                # Initial commit
                diff_output = repo.git.show(commit.hexsha, '--numstat', '--format=')
            
            for line in diff_output.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 3:
                    try:
                        insertions = int(parts[0]) if parts[0] != '-' else 0
                        deletions = int(parts[1]) if parts[1] != '-' else 0
                        filename = parts[2]
                        
                        total_insertions += insertions
                        total_deletions += deletions
                        total_files += 1
                        
                    except ValueError:
                        # Skip binary files or malformed lines
                        continue
        
        except Exception as e:
            print(f"Error in git diff: {e}")
            # Fallback to commit.stats
            stats = commit.stats.total
            total_files = stats.get('files', 0) if isinstance(stats, dict) else getattr(stats, 'files', 0)
            total_insertions = stats.get('insertions', 0) if isinstance(stats, dict) else getattr(stats, 'insertions', 0)
            total_deletions = stats.get('deletions', 0) if isinstance(stats, dict) else getattr(stats, 'deletions', 0)
        
        return {
            'files': total_files,
            'insertions': total_insertions,
            'deletions': total_deletions
        }
    
    repo_path = Path("/Users/masa/Projects/managed/gitflow-analytics")
    repo = Repo(repo_path)
    
    commits = list(repo.iter_commits('HEAD', max_count=3))
    
    for i, commit in enumerate(commits, 1):
        print(f"\n#{i} Commit: {commit.hexsha[:8]}")
        
        parent = commit.parents[0] if commit.parents else None
        
        # Original stats
        stats = commit.stats.total
        original_files = stats.get('files', 0) if isinstance(stats, dict) else getattr(stats, 'files', 0)
        original_insertions = stats.get('insertions', 0) if isinstance(stats, dict) else getattr(stats, 'insertions', 0)  
        original_deletions = stats.get('deletions', 0) if isinstance(stats, dict) else getattr(stats, 'deletions', 0)
        
        # Fixed calculation
        fixed_stats = calculate_lines_correctly(commit, parent)
        
        print(f"   Original: Files={original_files}, +{original_insertions}, -{original_deletions}")
        print(f"   Fixed:    Files={fixed_stats['files']}, +{fixed_stats['insertions']}, -{fixed_stats['deletions']}")
        
        match = (original_files == fixed_stats['files'] and 
                original_insertions == fixed_stats['insertions'] and
                original_deletions == fixed_stats['deletions'])
        
        print(f"   Status: {'✅ MATCH' if match else '❌ DIFFER'}")


if __name__ == "__main__":
    debug_diff_content()
    test_fixed_diff_parsing()