"""
01_feature_sets.py
==================
Defines the three strictly nested feature sets (T / T+O+W / T+O+W+S)
"""

# ── Topology-only (T) ─────────────────────────────────────────────────────────
TOPO_FEATURES = [
    "topo_src_degree",
    "topo_tgt_degree",
    "topo_src_betweenness",
    "topo_tgt_betweenness",
    "topo_src_closeness",
    "topo_tgt_closeness",
    "topo_src_clustering",
    "topo_tgt_clustering",
    "topo_src_eigenvector",
    "topo_tgt_eigenvector",
    "topo_edge_betweenness",
    "topo_common_neighbours",
]

# ── Operational + Weather additions (O+W) ─────────────────────────────────────
OPS_WEATHER_FEATURES = [
    "rides_planned",
    "stop_count",
    "disrupted_lag1",
    "disrupted_lag2",
    "disrupted_lag3",
    "disruption_count_lag1",
    "disruption_count_lag2",
    "disruption_count_lag3",
    "delay_freq_3m",
    "delay_freq_6m",
    # KNMI weather
    "DR", "RH", "SQ", "TG", "TN", "TX", "RHX",
]

# ── Socio-Economic additions (S) ──────────────────────────────────────────────
SES_FEATURES = [
    "source_SES_Score_Wealth_Avg",
    "source_SES_Score_Education_Avg",
    "source_TotalVandalism",
    "source_Remoteness_Index",
    "target_SES_Score_Wealth_Avg",
    "target_SES_Score_Education_Avg",
    "target_TotalVandalism",
    "target_Remoteness_Index",
    # Extended CBS indicators (include if present)
    "source_Income_Percentile",
    "source_CarOwnership_Rate",
    "source_Edu_High_Pct",
    "source_Edu_Low_Pct",
    "target_Income_Percentile",
    "target_CarOwnership_Rate",
    "target_Edu_High_Pct",
    "target_Edu_Low_Pct",
]

# ── Nested feature sets ───────────────────────────────────────────────────────
FEATURE_SETS = {
    "T":       TOPO_FEATURES,
    "T+O+W":   TOPO_FEATURES + OPS_WEATHER_FEATURES,
    "T+O+W+S": TOPO_FEATURES + OPS_WEATHER_FEATURES + SES_FEATURES,
}

TARGET = "Disrupted"


def resolve_features(feature_list, available_cols):
    """Return only those features that are present in the dataset."""
    return [f for f in feature_list if f in available_cols]