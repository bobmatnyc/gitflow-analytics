"""GitFlow Analytics - Git repository productivity analysis tool."""

from ._version import __version__, __version_info__

__author__ = "Bob Matyas"
__email__ = "bobmatnyc@gmail.com"

# Heavy imports removed from package __init__ for CLI performance
# Import these directly when needed in your code:
#   from gitflow_analytics.core.analyzer import GitAnalyzer
#   from gitflow_analytics.core.cache import GitAnalysisCache
#   from gitflow_analytics.core.identity import DeveloperIdentityResolver
#   from gitflow_analytics.extractors.story_points import StoryPointExtractor
#   from gitflow_analytics.extractors.tickets import TicketExtractor
#   from gitflow_analytics.reports.csv_writer import CSVReportGenerator

__all__ = [
    "__version__",
    "__version_info__",
]
