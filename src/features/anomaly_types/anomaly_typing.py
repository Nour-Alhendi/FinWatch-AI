# anomaly_typing.py
# Input:  feature DataFrame (one row = one ticker, one day)
# Output: same DataFrame + 5 new columns:
#           volume_anomaly_score      (0.0 - 1.0)
#           volatility_anomaly_score  (0.0 - 1.0)
#           price_anomaly_score       (0.0 - 1.0)
#           context_anomaly_score     (0.0 - 1.0)
#           anomaly_type              (dominant group as string)

import pandas as pd
import numpy as np

# Maps each anomaly group to the feature columns that define it
FEATURE_GROUPS = {
    "volume_flow": [
        "volume_zscore",
        "volume_trend",
        "obv_signal",
    ],
    "volatility_risk": [
        "vol_5",
        "vol_change",
        "volatility_ratio",
        "rolling_std",
    ],
    "price_momentum": [
        "returns",
        "return_lag_1",
        "return_lag_2",
        "momentum_5",
        "momentum_10",
        "rsi",
        "max_drawdown_30d",
        "trend_strength",
    ],
    "market_context": [
        "excess_return",
        "relative_return",
        "sector_relative",
        "beta",
        "corr_spx",
    ],
}

# Maps each group name to the output score column it produces
SCORE_COLUMNS = {
    "volume_flow": "volume_anomaly_score",
    "volatility_risk": "volatility_anomaly_score",
    "price_momentum": "price_anomaly_score",
    "market_context": "context_anomaly_score",
}


def _normalize_feature(series: pd.Series, window: int = 60) -> pd.Series:
    """Normalize a feature to [0, 1] via rolling z-score (window=60)."""
    # Compute rolling baseline over the lookback window
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()

    # Zero-std rows become NaN to avoid divide-by-zero
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)

    # Clip outliers to ±3σ then rescale to [0, 1]
    normalized = (z.clip(-3, 3) + 3) / 6

    # Fill missing (no history or flat series) with neutral 0.5
    return normalized.fillna(0.5)


def compute_group_scores(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Normalize each feature group and append one score column per group."""
    df = df.copy()

    for group, features in FEATURE_GROUPS.items():
        # Skip features not present in this DataFrame
        present = [f for f in features if f in df.columns]

        if not present:
            # No data for this group — mark score as NaN
            df[SCORE_COLUMNS[group]] = np.nan
            continue

        # Normalize each feature, then average across the group
        normalized = pd.concat(
            [_normalize_feature(df[f], window=window) for f in present],
            axis=1,
        )
        df[SCORE_COLUMNS[group]] = normalized.mean(axis=1)

    # Label each row with its dominant anomaly type
    df["anomaly_type"] = df.apply(get_anomaly_type, axis=1)
    return df


def get_anomaly_type(row: pd.Series) -> str:
    """Return the group name with the highest score, or 'unknown' if all NaN."""
    # Collect the four group scores for this row
    scores = {group: row.get(col, np.nan) for group, col in SCORE_COLUMNS.items()}

    # Drop NaN entries before comparing
    valid = {g: s for g, s in scores.items() if pd.notna(s)}

    if not valid:
        return "unknown"

    # Return the group with the maximum score
    return max(valid, key=lambda g: valid[g])
