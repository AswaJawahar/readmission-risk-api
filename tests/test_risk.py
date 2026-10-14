from fastapi.testclient import TestClient

from main import app, metadata
from risk import TIERS, assign_tier, percentile_rank

client = TestClient(app)
risk_tiers = metadata["risk_tiers"]
cutoffs = risk_tiers["cutoffs"]
quantiles = risk_tiers["score_quantiles"]


def test_tier_boundaries():
    assert assign_tier(cutoffs["moderate"] - 1e-9, cutoffs) == "Low"
    assert assign_tier(cutoffs["moderate"], cutoffs) == "Moderate"
    assert assign_tier(cutoffs["high"] - 1e-9, cutoffs) == "Moderate"
    assert assign_tier(cutoffs["high"], cutoffs) == "High"


def test_cutoffs_are_the_configured_percentiles():
    assert cutoffs["moderate"] == quantiles[risk_tiers["percentiles"]["moderate"]]
    assert cutoffs["high"] == quantiles[risk_tiers["percentiles"]["high"]]


def test_percentile_rank():
    assert percentile_rank(0.0, quantiles) == 0
    assert percentile_rank(1.0, quantiles) == 100
    assert percentile_rank(quantiles[50] + 1e-9, quantiles) == 50
    assert percentile_rank(quantiles[90] + 1e-9, quantiles) == 90


def test_tiers_cover_the_test_set_and_rise_with_risk():
    tiers = risk_tiers["tiers"]
    performance = metadata["performance"]
    assert [t["tier"] for t in tiers] == list(TIERS)
    assert sum(t["encounters"] for t in tiers) == performance["test_encounters"]
    assert sum(t["readmitted"] for t in tiers) == performance["readmitted"]
    rates = [t["readmission_rate"] for t in tiers]
    assert rates == sorted(rates)
    # The High tier should be clearly above the base rate
    assert rates[-1] >= 2 * performance["base_rate"]


def test_model_info_endpoint():
    body = client.get("/model-info").json()
    assert body["performance"]["auc"] == metadata["test_auc"]
    assert body["risk_tiers"]["cutoffs"] == cutoffs


def test_small_subgroups_report_counts_only():
    for row in metadata["subgroups"]:
        if row["readmitted"] < 100:
            assert "auc" not in row and "note" in row
        else:
            assert "auc" in row and len(row["auc_95_interval"]) == 2
