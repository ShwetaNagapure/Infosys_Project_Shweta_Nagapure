import os
import io
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import math

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
SEQ_LEN = 30
SELECTED_FEATURES = [
    "op_setting_1","op_setting_2","op_setting_3",
    "sensor_2","sensor_3","sensor_4","sensor_7","sensor_8",
    "sensor_9","sensor_11","sensor_12","sensor_13","sensor_14","sensor_15"
]
RUL_CLIP = 125

# Default thresholds
DEFAULT_THRESHOLDS = {"warning": 50, "critical": 20}

# -------------------
# Model Definition
# -------------------
class SmallGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32,1)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last)

def load_model_and_scaler():
    model_path = r"D:\Infosys Internship\best_gru_fd001_fixed.pth"
    scaler_path = r"D:\Infosys Internship\scaler_fd001.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = SmallGRU(input_dim=len(SELECTED_FEATURES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

def build_last_windows_from_df(df, seq_len=SEQ_LEN):
    X = []
    units_sorted = sorted(df["unit"].unique())
    for uid in units_sorted:
        unit_df = df[df["unit"] == uid].reset_index(drop=True)
        if len(unit_df) < seq_len:
            pad_len = seq_len - len(unit_df)
            pad = np.repeat(unit_df[SELECTED_FEATURES].iloc[[0]].values, pad_len, axis=0)
            window = np.vstack([pad, unit_df[SELECTED_FEATURES].values])
        else:
            window = unit_df[SELECTED_FEATURES].values[-seq_len:]
        X.append(window.astype(np.float32))
    return np.array(X, dtype=np.float32), units_sorted

def generate_alerts(units, preds, true_rul, warning_thresh=50, critical_thresh=20):
    alerts = []
    for u,p,t in zip(units, preds, true_rul):
        if p <= critical_thresh:
            level = "Critical"
        elif p <= warning_thresh:
            level = "Warning"
        else:
            level = "Normal"
        error = abs(p-t)
        alerts.append({
            'unit': int(u),
            'predicted_rul': float(p),
            'true_rul': float(t),
            'level': level,
            'error': float(error)
        })
    return alerts

def run_fd001(unseen_content, true_rul_content, warning=DEFAULT_THRESHOLDS['warning'], critical=DEFAULT_THRESHOLDS['critical']):
    """Main function to run FD001 model prediction"""

    # Load model and scaler
    model, scaler = load_model_and_scaler()

    # Read unseen dataset
    cols_full = ["unit","cycle"] + [f"op_setting_{i}" for i in range(1,4)] + [f"sensor_{i}" for i in range(1,22)]
    unseen_df = pd.read_csv(io.StringIO(unseen_content.decode('utf-8')), sep='\s+', header=None).dropna(axis=1, how='all')
    unseen_df = unseen_df.iloc[:, :len(cols_full)]
    unseen_df.columns = cols_full
    unseen_df = unseen_df[["unit","cycle"] + SELECTED_FEATURES]
    unseen_df[SELECTED_FEATURES] = scaler.transform(unseen_df[SELECTED_FEATURES])

    # Read true RUL
    true_vals = np.loadtxt(io.StringIO(true_rul_content.decode('utf-8')))

    # Prepare sequence windows
    X_unseen, units = build_last_windows_from_df(unseen_df)
    X_t = torch.tensor(X_unseen, dtype=torch.float32).to(DEVICE)

    # Predict
    with torch.no_grad():
        preds = model(X_t).cpu().numpy().flatten().clip(max=RUL_CLIP)

    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(true_vals, preds))

    # Generate alerts
    alerts = generate_alerts(units, preds, true_vals, warning, critical)

    # Return results
    return {
        'units': units,
        'preds': preds.tolist(),
        'true_rul': true_vals.tolist(),
        'thresholds': {'warning': warning, 'critical': critical},
        'rmse': rmse,
        'alerts': alerts
    }
