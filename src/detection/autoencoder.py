# LSTM Autoencoder for anomaly detection.
# Trains on calm periods (low volatility) and flags sequences with high reconstruction error.
# Columns: ae_error, ae_anomaly

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.features import LSTM_AE_FEATURES

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30
N_FEATURES = len(LSTM_AE_FEATURES)

def build_model():
    input_layer = Input(shape=(SEQUENCE_LENGTH, N_FEATURES))
    encoded = LSTM(32, activation="relu")(input_layer)
    repeated = RepeatVector(SEQUENCE_LENGTH)(encoded)
    decoded = LSTM(32, activation="relu", return_sequences=True)(repeated)
    output_layer = TimeDistributed(Dense(N_FEATURES))(decoded)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mse")
    return model

def run_autoencoder():
    for file in INPUT_DIR.glob("*.parquet"):
        if file.stem == "^SPX":
            continue

        df = pd.read_parquet(file)
        df = df[df["Date"] >= "2022-01-01"].reset_index(drop=True)
        df = df.dropna(subset=LSTM_AE_FEATURES).reset_index(drop=True)

        # Train only on calm periods
        normal_mask = df["volatility"] < df["volatility"].quantile(0.75)
        df_train = df[normal_mask].reset_index(drop=True)

        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(df_train[LSTM_AE_FEATURES])
        scaled_all = scaler.transform(df[LSTM_AE_FEATURES])

        X_train = np.array([scaled_train[i-SEQUENCE_LENGTH:i] for i in range(SEQUENCE_LENGTH, len(scaled_train))])
        X_all = np.array([scaled_all[i-SEQUENCE_LENGTH:i] for i in range(SEQUENCE_LENGTH, len(scaled_all))])

        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model = build_model()
        model.fit(X_train, X_train, epochs=50, batch_size=32,
                  validation_split=0.1, callbacks=[early_stop], verbose=0)

        X_pred = model.predict(X_all, verbose=0)
        errors = np.mean(np.abs(X_all - X_pred), axis=(1, 2))
        threshold = np.percentile(errors, 95)

        df["ae_error"] = np.nan
        df["ae_anomaly"] = False
        df.loc[SEQUENCE_LENGTH:len(df)-1, "ae_error"] = errors
        df.loc[SEQUENCE_LENGTH:len(df)-1, "ae_anomaly"] = errors > threshold

        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")

if __name__ == "__main__":
    run_autoencoder()
