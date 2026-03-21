# Central feature group definitions used across all models and pipelines.

PRICE_FEATURES = [
    "returns",
    "return_lag_1", "return_lag_2", "return_lag_3",
    "momentum_5", "momentum_10"
]

VOLATILITY_FEATURES = [
    "volatility",
    "vol_change",
    "rolling_std"
]

MARKET_FEATURES = [
    "spx_return",
    "relative_return",
    "sector_relative",
    "regime_encoded",
    "trend_strength",
    "volatility_ratio"
]

MOMENTUM_FEATURES = [
    "rsi",
    "beta",
    "corr_spx"
]

VOLUME_FEATURES = [
    "volume_zscore"
]

# All features combined — used as default input for models
ALL_FEATURES = (
    PRICE_FEATURES +
    VOLATILITY_FEATURES +
    MARKET_FEATURES +
    MOMENTUM_FEATURES +
    VOLUME_FEATURES
)

# Isolation Forest — static point features (tuned: ratio 4.22x)
IF_FEATURES = [
    "returns",
    "volatility",
    "relative_return",
    "rsi",
    "beta",
    "z_score"
]

# LSTM Autoencoder — temporal sequence features
LSTM_AE_FEATURES = [
    "returns", 
    "return_lag_1", 
    "return_lag_2", 
    "momentum_5",
    "volatility", 
    "rolling_mean",
    "rsi",
    "spx_return", 
    "corr_spx", 
    "beta",
    "volume_zscore"
]
