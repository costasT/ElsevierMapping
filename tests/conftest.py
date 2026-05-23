import pytest
from sdg_classifier import SDGClassifier


@pytest.fixture(scope="session")
def classifier():
    """Session-scoped SDGClassifier instance shared by tests."""
    return SDGClassifier()
