"""Compatibility alias for the historic isaaclab_nav_task package name."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("instinct_nav")
sys.modules[__name__] = _module
