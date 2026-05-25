"""Helpers for running package scripts directly from the repository tree."""

from pathlib import Path
import sys


def ensure_repo_root(levels_up=1):
    repo_root = Path(__file__).resolve()
    for _ in range(levels_up + 1):
        repo_root = repo_root.parent
    repo_root = str(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
