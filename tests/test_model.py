"""Model tests, mocked and CPU-only."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_pipeline_import():
    """Check that the pipeline imports without error."""
    from src.models.pipeline import AlpacaVisionPipeline
    from src.models.detector import AlpacaDetector
    from src.models.classifier import AnomalyClassifier
