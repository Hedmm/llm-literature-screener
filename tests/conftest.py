"""Shared fixtures for the test suite."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"


@pytest.fixture
def fixtures() -> Path:
    """Directory with the small synthetic files used by the format tests."""
    return FIXTURES
