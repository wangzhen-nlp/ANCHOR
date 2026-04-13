#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path


def ensure_repo_root(levels_to_root=1):
    repo_root = Path(__file__).resolve().parents[levels_to_root]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
