"""GitHub integration for code scanning (PR comments via deepteam[bot])."""

from .report import post_pr_comments

__all__ = ["post_pr_comments"]
