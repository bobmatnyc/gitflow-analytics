"""GitFlow Analytics - Git repository productivity analysis tool."""

from ._version import __version__, __version_info__

__author__ = "Bob Matyas"
__email__ = "bobmatnyc@gmail.com"

from .core.analyzer import GitAnalyzer
from .core.cache import GitAnalysisCache
from .core.identity import DeveloperIdentityResolver
from .extractors.story_points import StoryPointExtractor
from .extractors.tickets import TicketExtractor
from .reports.csv_writer import CSVReportGenerator

__all__ = [
    "__version__",
    "__version_info__",
    "GitAnalyzer",
    "GitAnalysisCache",
    "DeveloperIdentityResolver",
    "StoryPointExtractor",
    "TicketExtractor",
    "CSVReportGenerator",
]
