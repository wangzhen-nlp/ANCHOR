#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path


def ensure_repo_root(depth=1):
    current = Path(__file__).resolve()
    for _ in range(depth):
        current = current.parent
    repo_root = current.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

