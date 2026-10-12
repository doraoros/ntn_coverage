"""NTN / 6G coverage predictor (3GPP TR 38.811 / 38.821)."""

from ntn_coverage.coverage import CoverageGrid, PointPrediction, predict_coverage, predict_point
from ntn_coverage.scenarios import SCENARIOS, get_scenario

__version__ = "0.7.0"
__all__ = [
    "CoverageGrid",
    "PointPrediction",
    "SCENARIOS",
    "get_scenario",
    "predict_coverage",
    "predict_point",
]
