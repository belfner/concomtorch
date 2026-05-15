"""
Correctness suite placeholder.

The full BKE correctness suite is deferred. This module exists so the gate collects a non-empty test set (pytest
exits 5 on an empty collection) until that suite lands here.
"""

from __future__ import annotations

import pytest


def test_correctness_suite_pending() -> None:
    """
    Hold a collectible slot until the deferred correctness suite is written.
    """
    pytest.skip("correctness suite deferred until its tests land here")
