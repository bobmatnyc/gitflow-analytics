"""CLI subpackage for GitFlow Analytics.

This package contains CLI-related modules including the installation wizard,
interactive launcher, and main menu system.
"""

from .install_wizard import InstallWizard
from .menu import show_main_menu
from .run_launcher import InteractiveLauncher, run_interactive_launcher

__all__ = [
    "InstallWizard",
    "InteractiveLauncher",
    "run_interactive_launcher",
    "show_main_menu",
]
