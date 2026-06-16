# LSTM Autoencoder for anomaly detection — dual model per group (low/high volatility regime).
# Trains separate models for calm and volatile periods.
# Columns: ae_error, ae_anomaly

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_METAL_DEVICE_ENABLE"] = "0"

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.features import LSTM_AE_FEATURES

ROOT       = Path(__file__).resolve().parents[2]
INPUT_DIR  = ROOT / "data/detection"
OUTPUT_DIR = ROOT / "data/detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 20
LSTM_UNITS      = 128
LATENT_DIM      = 16
DROPOUT         = 0.1
SMOOTH_WINDOW   = 5
N_FEATURES      = len(LSTM_AE_FEATURES)

GROUPS = {
    "Technology-Stable":   {"tickers": ["AAPL", "MSFT", "GOOG", "INTC", "IBM", "DELL"], "calm_q": 0.65, "percentile": 3},
    "Technology-Volatile": {"tickers": ["NVDA", "AMD", "NOK"],                           "calm_q": 0.60, "percentile": 4},
    "Semiconductors":      {"tickers": ["AVGO", "QCOM", "MU", "MRVL", "ASML"],          "calm_q": 0.65, "percentile": 3},
    "AI-Stable":           {"tickers": ["CRM", "SNOW"],                                  "calm_q": 0.65, "percentile": 5},
    "AI-Volatile":         {"tickers": ["PLTR", "META"],                                 "calm_q": 0.65, "percentile": 4},
    "Cybersecurity":       {"tickers": ["PANW", "CRWD", "NET"],                          "calm_q": 0.65, "percentile": 4},
    "Consumer-Volatile":   {"tickers": ["TSLA", "AMZN"],                                 "calm_q": 0.70, "percentile": 3},
    "Financials":          {"tickers": ["JPM", "BAC", "GS", "BLK", "V", "MA"],          "calm_q": 0.70, "percentile": 4},
    "Healthcare":          {"tickers": ["JNJ", "PFE", "UNH", "ABBV", "LLY", "AMGN"],   "calm_q": 0.75, "percentile": 3},
    "Consumer Staples":    {"tickers": ["PG", "KO", "COST", "WMT"],                     "calm_q": 0.70, "percentile": 3},
    "Energy":              {"tickers": ["XOM", "CVX", "COP", "EOG"],                     "calm_q": 0.70, "percentile": 3},
    "Industrials":         {"tickers": ["CAT", "HON", "BA", "GE", "RTX"],                "calm_q": 0.70, "percentile": 3},
    "Green Energy":        {"tickers": ["ENPH", "NEE", "FSLR"],                          "calm_q": 0.75, "percentile": 3},
    "Crypto-Volatile":     {"tickers": ["IREN", "MARA", "RIOT", "COIN", "APLD"],        "calm_q": 0.60, "percentile": 5},
    "SmallCap-Volatile":   {"tickers": ["NBIS", "AI"],                                   "calm_q": 0.55, "percentile": 5},
    "Quantum":             {"tickers": ["IONQ", "QBTS"],                                 "calm_q": 0.55, "percentile": 5},
}

# Crisis windows excluded from calm_training (scaler + threshold calibration)
ALL_CRISES = [
    ("2020-02-01", "2020-04-30"),  # COVID crash — all groups
    ("2022-01-01", "2022-10-31"),  # Fed shock — all groups
    ("2025-04-01", "2025-04-30"),  # Trump tariffs — all groups (post-2024, no effect until split_date moves)
    ("2026-03-01", "2026-03-31"),  # February fade — all groups (post-2024)
]

GROUP_CRISES = {
    "Energy":      [("2022-02-01", "2022-03-31"), ("2026-03-01", "2026-03-31")],  # Ukraine + Iran war
    "Industrials": [("2022-02-01", "2022-03-31"), ("2026-03-01", "2026-03-31")],  # Ukraine + Iran war
    "Financials":  [("2023-03-01", "2023-03-31")],                                 # SVB
}


def get_calm_mask(df: pd.DataFrame, group_name: str) -> pd.Series:
    """Returns bool Series — True = calm day (not in any crisis window)."""
    mask = pd.Series(True, index=df.index)
    for start, end in ALL_CRISES + GROUP_CRISES.get(group_name, []):
        crisis = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
        mask = mask & ~crisis
    return mask


def build_model():
    inputs = Input(shape=(SEQUENCE_LENGTH, N_FEATURES))
    x = LSTM(LSTM_UNITS, activation="relu", return_sequences=False)(inputs)
    x = Dropout(DROPOUT)(x)
    x = Dense(LATENT_DIM, activation="relu")(x)
    x = RepeatVector(SEQUENCE_LENGTH)(x)
    x = LSTM(LSTM_UNITS, activation="relu", return_sequences=True)(x)
    x = Dropout(DROPOUT)(x)
    output = TimeDistributed(Dense(N_FEATURES))(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="mse")
    return model


def build_sequences(data):
    return np.array([data[i-SEQUENCE_LENGTH:i] for i in range(SEQUENCE_LENGTH, len(data))])


def get_errors(model, X):
    X_pred = model.predict(X, verbose=0)
    errors = np.mean(np.abs(X - X_pred), axis=(1, 2))
    return pd.Series(errors).ewm(span=SMOOTH_WINDOW).mean().values


def run_autoencoder():
    tf.config.set_visible_devices([], 'GPU')
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    Path("models").mkdir(exist_ok=True)
    split_date = pd.Timestamp("2024-01-01")
    for group_name, params in GROUPS.items():
        tickers  = params["tickers"]
        calm_q   = params["calm_q"]
        perc     = params["percentile"]

        # Load data per ticker
        frames = []
        for ticker in tickers:
            file = INPUT_DIR / f"{ticker}.parquet"
            if not file.exists():
                print(f"  Skipping {ticker}: file not found")
                continue
            df = pd.read_parquet(file)
            df = df.dropna(subset=LSTM_AE_FEATURES).reset_index(drop=True)
            if df.empty:
                print(f"  Skipping {ticker}: no rows after dropna (missing features)")
                continue
            df["_is_train"] = df["Date"] < split_date
            df["_ticker"] = ticker
            frames.append(df)

        if not frames:
            print(f"Skipping group {group_name}: no data")
            continue

        # Split into low/high vol regime per ticker and scale separately
        low_frames  = []
        high_frames = []

        # Scaler fit on calm training data (crisis excluded, no regime split)
        # Full training data (all dates, low/high split) used for model weights
        for df in frames:
            train_df = df[df["_is_train"]]
            vol_threshold = train_df["volatility"].quantile(calm_q)
            df["_is_low_vol"]    = df["volatility"] <= vol_threshold
            df["_vol_threshold"] = vol_threshold

            # Calm training: exclude crisis periods, fit a single unified scaler
            calm_mask  = get_calm_mask(df, group_name)
            calm_train = df[df["_is_train"] & calm_mask].copy().reset_index(drop=True)

            if len(calm_train) < SEQUENCE_LENGTH:
                print(f"  Skipping {df['_ticker'].iloc[0]}: not enough calm training data ({len(calm_train)} rows)")
                continue

            scaler = MinMaxScaler()
            scaler.fit(calm_train[LSTM_AE_FEATURES])

            # Full training splits — scaled with calm scaler, crisis extremes clipped to [0, 1]
            low_df_train  = df[df["_is_low_vol"] &  df["_is_train"]].copy().reset_index(drop=True)
            high_df_train = df[~df["_is_low_vol"] & df["_is_train"]].copy().reset_index(drop=True)

            if len(low_df_train) > SEQUENCE_LENGTH:
                low_df_train[LSTM_AE_FEATURES]  = np.clip(scaler.transform(low_df_train[LSTM_AE_FEATURES]),  0, 1)
                high_df_train[LSTM_AE_FEATURES] = np.clip(scaler.transform(high_df_train[LSTM_AE_FEATURES]), 0, 1)
                low_frames.append((df, low_df_train, high_df_train, scaler, calm_mask))

        if not low_frames:
            print(f"Skipping group {group_name}: not enough low-vol data")
            continue

        # Build training sequences
        X_train_low  = []
        X_train_high = []

        for (df, low_df_train, high_df_train, scaler, calm_mask) in low_frames:
            if len(low_df_train) > SEQUENCE_LENGTH:
                X_train_low.append(build_sequences(low_df_train[LSTM_AE_FEATURES].values))
            if len(high_df_train) > SEQUENCE_LENGTH:
                X_train_high.append(build_sequences(high_df_train[LSTM_AE_FEATURES].values))

        X_train_low  = np.concatenate(X_train_low)  if X_train_low  else None
        X_train_high = np.concatenate(X_train_high) if X_train_high else None

        # Train dual models
        low_path  = Path(f"models/ae_{group_name}_low.keras")
        high_path = Path(f"models/ae_{group_name}_high.keras")

        if low_path.exists():
            from tensorflow.keras.models import load_model
            model_low = load_model(low_path)
            print(f"[{group_name}] low_vol_model loaded from disk")
        else:
            model_low = build_model()
            if X_train_low is not None:
                model_low.fit(X_train_low, X_train_low,
                              epochs=30, batch_size=32,
                              validation_split=0.1,
                              callbacks=[early_stop], verbose=0)
                model_low.save(low_path)
                print(f"[{group_name}] low_vol_model trained and saved")

        if high_path.exists():
            from tensorflow.keras.models import load_model
            model_high = load_model(high_path)
            print(f"[{group_name}] high_vol_model loaded from disk")
        else:
            model_high = build_model()
            if X_train_high is not None:
                model_high.fit(X_train_high, X_train_high,
                               epochs=30, batch_size=32,
                               validation_split=0.1,
                               callbacks=[early_stop], verbose=0)
                model_high.save(high_path)
                print(f"[{group_name}] high_vol_model trained and saved")

        # Predict and save per ticker
        for (df, low_df_train, high_df_train, scaler, calm_mask) in low_frames:
            ticker        = df["_ticker"].iloc[0]
            vol_threshold = df["_vol_threshold"].iloc[0]

            orig_df = pd.read_parquet(INPUT_DIR / f"{ticker}.parquet")
            feat_df = orig_df.dropna(subset=LSTM_AE_FEATURES).reset_index(drop=True)
            orig_df["ae_error"]   = np.nan
            orig_df["ae_anomaly"] = False
            feat_df["_is_low_vol"] = feat_df["volatility"] <= vol_threshold

            # Recompute calm mask on feat_df for threshold calculation
            calm_feat = get_calm_mask(feat_df, group_name)

            low_idx  = feat_df[feat_df["_is_low_vol"]].index
            high_idx = feat_df[~feat_df["_is_low_vol"]].index

            # Low vol regime prediction — scaler from calm, clip crisis values
            if len(low_idx) > SEQUENCE_LENGTH:
                low_scaled = np.clip(scaler.transform(feat_df.loc[low_idx, LSTM_AE_FEATURES]), 0, 1)
                X_low_all  = build_sequences(low_scaled)
                errors_low = get_errors(model_low, X_low_all)

                # Threshold from calm low-vol training days only
                calm_low_idx = feat_df[feat_df["_is_low_vol"] & (feat_df["Date"] < split_date) & calm_feat].index
                if len(calm_low_idx) > SEQUENCE_LENGTH:
                    calm_low_scaled  = np.clip(scaler.transform(feat_df.loc[calm_low_idx, LSTM_AE_FEATURES]), 0, 1)
                    calm_low_errors  = get_errors(model_low, build_sequences(calm_low_scaled))
                else:
                    # Fallback: all calm training data through model_low
                    calm_all_idx     = feat_df[(feat_df["Date"] < split_date) & calm_feat].index
                    calm_all_scaled  = np.clip(scaler.transform(feat_df.loc[calm_all_idx, LSTM_AE_FEATURES]), 0, 1)
                    calm_low_errors  = get_errors(model_low, build_sequences(calm_all_scaled))
                threshold_low = np.percentile(np.asarray(calm_low_errors, dtype=float), 100 - perc)

                target_dates = feat_df.loc[low_idx[SEQUENCE_LENGTH:SEQUENCE_LENGTH + len(errors_low)], "Date"].values
                mask = orig_df["Date"].isin(target_dates)
                orig_df.loc[mask, "ae_error"]   = errors_low
                orig_df.loc[mask, "ae_anomaly"] = errors_low > threshold_low

            # High vol regime prediction — same calm scaler, clip crisis values
            if len(high_idx) > SEQUENCE_LENGTH:
                high_scaled = np.clip(scaler.transform(feat_df.loc[high_idx, LSTM_AE_FEATURES]), 0, 1)
                X_high_all  = build_sequences(high_scaled)
                errors_high = get_errors(model_high, X_high_all)

                # Threshold from calm high-vol training days only
                calm_high_idx = feat_df[~feat_df["_is_low_vol"] & (feat_df["Date"] < split_date) & calm_feat].index
                if len(calm_high_idx) > SEQUENCE_LENGTH:
                    calm_high_scaled = np.clip(scaler.transform(feat_df.loc[calm_high_idx, LSTM_AE_FEATURES]), 0, 1)
                    calm_high_errors = get_errors(model_high, build_sequences(calm_high_scaled))
                else:
                    calm_all_idx     = feat_df[(feat_df["Date"] < split_date) & calm_feat].index
                    calm_all_scaled  = np.clip(scaler.transform(feat_df.loc[calm_all_idx, LSTM_AE_FEATURES]), 0, 1)
                    calm_high_errors = get_errors(model_high, build_sequences(calm_all_scaled))
                threshold_high = np.percentile(np.asarray(calm_high_errors, dtype=float), 100 - perc)

                target_dates = feat_df.loc[high_idx[SEQUENCE_LENGTH:SEQUENCE_LENGTH + len(errors_high)], "Date"].values
                mask = orig_df["Date"].isin(target_dates)
                orig_df.loc[mask, "ae_error"]   = errors_high
                orig_df.loc[mask, "ae_anomaly"] = errors_high > threshold_high

            orig_df.to_parquet(OUTPUT_DIR / f"{ticker}.parquet")
            print(f"  Saved: {ticker}.parquet  ({orig_df['ae_anomaly'].sum()} anomalies)")


if __name__ == "__main__":
    run_autoencoder()
