"""Risk tier and percentile rules, shared by the API and build_metadata.py."""
import bisect

TIERS = ("Low", "Moderate", "High")


def assign_tier(score, cutoffs):
    """Low below the moderate cutoff, High at or above the high cutoff, Moderate in between."""
    if score >= cutoffs["high"]:
        return "High"
    if score >= cutoffs["moderate"]:
        return "Moderate"
    return "Low"


def percentile_rank(score, quantiles):
    """Approximate percentage (0-100) of test-set scores below `score`.

    `quantiles` holds the 0th..100th percentiles of the test-set scores.
    """
    return max(0, min(100, bisect.bisect_left(quantiles, score) - 1))
