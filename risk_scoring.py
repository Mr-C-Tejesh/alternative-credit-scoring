"""Illustrative credit scoring and risk band transformation.

This module provides an educational, configurable probability-to-score mapping
using Points to Double the Odds (PDO) methodology and defines illustrative
model-estimated risk bands.

IMPORTANT DISCLAIMER:
This is an engineering demonstration of credit-scoring mathematics on synthetic data.
Outputs are NOT official credit bureau scores (e.g., CIBIL/FICO), regulator-approved
credit ratings, or validated underwriting decisions. They must not be used for real
credit approvals, pricing, or lending eligibility.
"""
from __future__ import annotations

import math
from typing import NamedTuple
import numpy as np


# Standard demonstration scoring defaults
DEFAULT_BASE_SCORE: float = 600.0
DEFAULT_BASE_ODDS: float = 50.0  # 50:1 Good-to-Bad odds
DEFAULT_PDO: float = 20.0       # 20 Points to Double the Odds
DEFAULT_MIN_SCORE: int = 300
DEFAULT_MAX_SCORE: int = 850
DEFAULT_EPSILON: float = 1e-12

# Illustrative model-estimated risk band defaults (from app.py demonstration)
DEFAULT_LOWER_RISK_CUTOFF: float = 0.10
DEFAULT_UPPER_RISK_CUTOFF: float = 0.20

LOWER_RISK_LABEL: str = "Lower model-estimated risk"
MIDDLE_RISK_LABEL: str = "Middle model-estimated risk"
HIGHER_RISK_LABEL: str = "Higher model-estimated risk"

BAND_LOWER: str = LOWER_RISK_LABEL
BAND_MIDDLE: str = MIDDLE_RISK_LABEL
BAND_HIGHER: str = HIGHER_RISK_LABEL


class ScoreResult(NamedTuple):
    """Container holding raw and clamped demonstration scores."""
    raw_score: float
    display_score: int
    good_to_bad_odds: float
    base_score: float
    base_odds: float
    pdo: float
    min_score: int
    max_score: int
    is_clamped: bool

    @property
    def default_probability(self) -> float:
        return 1.0 / (1.0 + self.good_to_bad_odds)

    @property
    def odds_good_to_bad(self) -> float:
        return self.good_to_bad_odds


def _validate_scoring_params(
    base_score: float,
    base_odds: float,
    pdo: float,
    min_score: int,
    max_score: int,
) -> None:
    """Validate mathematical parameters for the PDO score transformation."""
    if not np.isfinite(base_score):
        raise ValueError(f"base_score must be a finite number; got {base_score}.")
    if not np.isfinite(base_odds) or base_odds <= 0.0:
        raise ValueError(f"base_odds must be a finite positive number; got {base_odds}.")
    if not np.isfinite(pdo) or pdo <= 0.0:
        raise ValueError(f"pdo must be a finite positive number; got {pdo}.")
    if not isinstance(min_score, (int, np.integer)) or not isinstance(max_score, (int, np.integer)):
        raise ValueError(f"min_score and max_score must be integers; got ({min_score}, {max_score}).")
    if min_score >= max_score:
        raise ValueError(f"min_score ({min_score}) must be strictly less than max_score ({max_score}).")


def _validate_probability(p: float) -> float:
    """Validate that probability is finite and within [0.0, 1.0]."""
    if not np.isfinite(p):
        raise ValueError(f"Default probability must be a finite number; got {p}.")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Default probability must be in [0.0, 1.0]; got {p}.")
    return float(p)


def probability_to_raw_score(
    p: float,
    base_score: float = DEFAULT_BASE_SCORE,
    base_odds: float = DEFAULT_BASE_ODDS,
    pdo: float = DEFAULT_PDO,
    eps: float = DEFAULT_EPSILON,
) -> float:
    """Calculate the unclamped raw credit score from predicted default probability.

    Mathematical formulation:
        odds = (1 - p) / p  [good-to-bad odds]
        raw_score = base_score + (pdo / ln(2)) * ln(odds / base_odds)

    Sign convention:
        Higher predicted default probability (p) -> lower good-to-bad odds -> lower raw score.
        At p = 1 / (base_odds + 1), odds = base_odds, and raw_score = base_score.
        Doubling good-to-bad odds increases score by exactly pdo points.
        Doubling default odds (halving good-to-bad odds) decreases score by exactly pdo points.
    """
    _validate_probability(p)
    _validate_scoring_params(base_score, base_odds, pdo, DEFAULT_MIN_SCORE, DEFAULT_MAX_SCORE)

    # Numerical stabilization at probability boundaries 0 and 1
    p_eff = float(np.clip(p, eps, 1.0 - eps))
    good_to_bad_odds = (1.0 - p_eff) / p_eff

    factor = pdo / math.log(2.0)
    raw_score = base_score + factor * math.log(good_to_bad_odds / base_odds)
    return float(raw_score)


def probability_to_score(
    p: float | None = None,
    base_score: float = DEFAULT_BASE_SCORE,
    base_odds: float = DEFAULT_BASE_ODDS,
    pdo: float = DEFAULT_PDO,
    min_score: int = DEFAULT_MIN_SCORE,
    max_score: int = DEFAULT_MAX_SCORE,
    eps: float = DEFAULT_EPSILON,
    probability: float | None = None,
) -> ScoreResult:
    """Transform predicted default probability into an illustrative credit score.

    Returns a ScoreResult containing both the unclamped continuous raw score and
    the integer score clamped to [min_score, max_score].
    """
    if p is None:
        if probability is not None:
            p = probability
        else:
            raise TypeError("probability_to_score() missing required argument: 'p' or 'probability'")

    _validate_scoring_params(base_score, base_odds, pdo, min_score, max_score)
    p_clean = _validate_probability(p)

    raw = probability_to_raw_score(p_clean, base_score, base_odds, pdo, eps)

    p_eff = float(np.clip(p_clean, eps, 1.0 - eps))
    odds = (1.0 - p_eff) / p_eff

    # Clamping and rounding
    is_clamped = (raw < float(min_score) or raw > float(max_score))
    clamped = max(float(min_score), min(float(max_score), raw))
    display_score = int(round(clamped))

    return ScoreResult(
        raw_score=raw,
        display_score=display_score,
        good_to_bad_odds=odds,
        base_score=base_score,
        base_odds=base_odds,
        pdo=pdo,
        min_score=min_score,
        max_score=max_score,
        is_clamped=is_clamped,
    )


def map_probability_to_risk_band(
    p: float,
    lower_cutoff: float = DEFAULT_LOWER_RISK_CUTOFF,
    upper_cutoff: float = DEFAULT_UPPER_RISK_CUTOFF,
    middle_cutoff: float | None = None,
) -> str:
    """Map default probability to an illustrative model risk band.

    Cutoff rules:
        p < lower_cutoff:                   "Lower model-estimated risk"
        lower_cutoff <= p < upper_cutoff:   "Middle model-estimated risk"
        p >= upper_cutoff:                  "Higher model-estimated risk"

    Note:
        These risk bands are inherited demonstration settings from the prototype UI.
        They are NOT credit underwriting decision rules (e.g. approve/reject/tier)
        and do not represent calibrated lending loss cutoffs.
    """
    if middle_cutoff is not None:
        upper_cutoff = middle_cutoff

    _validate_probability(p)
    if not (np.isfinite(lower_cutoff) and np.isfinite(upper_cutoff)):
        raise ValueError("Risk band cutoffs must be finite numbers.")
    if not (0.0 <= lower_cutoff < upper_cutoff <= 1.0):
        raise ValueError(
            f"Cutoffs must satisfy 0.0 <= lower_cutoff < upper_cutoff <= 1.0; "
            f"got lower={lower_cutoff}, upper={upper_cutoff}."
        )

    if p < lower_cutoff:
        return LOWER_RISK_LABEL
    elif p < upper_cutoff:
        return MIDDLE_RISK_LABEL
    else:
        return HIGHER_RISK_LABEL
