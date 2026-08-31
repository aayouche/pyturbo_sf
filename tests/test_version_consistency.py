"""The version string lives in three places; keep them in sync.

docs/source/conf.py silently drifted to 1.0.7 while the package was at 1.0.8.
"""
import re
from pathlib import Path

import pytest

import pyturbo_sf

ROOT = Path(__file__).resolve().parents[1]

def test_pyproject_matches_package():
    path = ROOT / "pyproject.toml"
    if not path.exists():
        pytest.skip("pyproject.toml not present in this checkout")
    m = re.search(r'^version\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"), re.M)
    assert m, "no version found in pyproject.toml"
    assert m.group(1) == pyturbo_sf.__version__

@pytest.mark.parametrize("key", ["release", "version"])
def test_docs_conf_matches_package(key):
    conf = ROOT / "docs" / "source" / "conf.py"
    if not conf.exists():
        pytest.skip("docs not present in this checkout")
    m = re.search(rf'^{key}\s*=\s*["\']([^"\']+)["\']', conf.read_text(encoding="utf-8"), re.M)
    assert m, f"no {key} found in docs/source/conf.py"
    assert m.group(1) == pyturbo_sf.__version__

def test_changelog_documents_current_version():
    path = ROOT / "CHANGELOG.md"
    if not path.exists():
        pytest.skip("CHANGELOG.md not present in this checkout")
    assert f"[{pyturbo_sf.__version__}]" in path.read_text(encoding="utf-8")
