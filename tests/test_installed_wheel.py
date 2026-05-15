"""
Provenance guard: the suite must exercise the installed repaired wheel, not
the editable source tree. This is the brief's first hard constraint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

concomtorch = pytest.importorskip("concomtorch")


def test_concomtorch_imported_from_site_packages() -> None:
    """
    Assert ``concomtorch`` resolves from an installed site-packages location.
    """
    mod_path = Path(concomtorch.__file__).resolve()
    parts = mod_path.parts
    assert "site-packages" in parts, f"concomtorch must import from the installed wheel; got {mod_path}"


def test_concomtorch_not_imported_from_source_tree() -> None:
    """
    Assert the import did not resolve to the in-repo ``package/src`` tree.
    """
    mod_path = Path(concomtorch.__file__).resolve()
    parts = mod_path.parts
    assert "src" not in parts, f"concomtorch resolved to the source tree, not the wheel: {mod_path}"
    assert "package" not in parts, f"concomtorch resolved to the source tree, not the wheel: {mod_path}"


def test_concomtorch_exposes_public_api() -> None:
    """
    Assert the documented public entry points are present on the wheel.
    """
    for name in (
        "connected_components",
        "create_labels_buffer",
        "get_unique_labels",
        "get_component_masks",
    ):
        assert hasattr(concomtorch, name), f"missing public API: {name}"
