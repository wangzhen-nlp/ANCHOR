"""让 tools 下的脚本在直接执行时也能导入 official 包。"""

import sys

from pathlib import Path


def ensure_package_parent():
    package_parent = str(Path(__file__).resolve().parents[2])
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
