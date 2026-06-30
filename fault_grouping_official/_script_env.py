#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path


def ensure_package_parent():
    package_parent = str(Path(__file__).resolve().parent.parent)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
