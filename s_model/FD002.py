import os
import io
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request,redirect, url_for, flash, jsonify, send_file,render_template
import torch
import torch.nn as nn
import plotly
import plotly.graph_objs as go
from datetime import datetime
from io import BytesIO
import math
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ================= CONFIG =================
SEQ_LEN = 80
RUL_CLIP = 130
DEFAULT_THRESHOLDS = {"warning": 50, "critical": 20}
MODEL_PATH = r"model\best_model_fd002_hyper.pth"
SCALER_PATH = r"scaler\scaler_fd002_hyper.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SELECTED_FEATURES = [
    "op_setting_1","op_setting_2","op_setting_3",
    "sensor_2","sensor_3","sensor_4","sensor_7","sensor_8",
    "sensor_9","sensor_11","sensor_12","sensor_13","sensor_14","sensor_15"
]

# ================= MODEL =================
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)

class SmallModel(nn.Module):
    def __init__(self, input_dim, cnn_filters=64, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, cnn_filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            cnn_filters, lstm_hidden, lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.attn = Attention(2 * lstm_hidden)
        self.fc = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.attn(x)
        return self.fc(x)

def load_model_and_scaler(fd_model="FD002"):
    model_path = MODEL_PATH
    scaler_path = SCALER_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    model = SmallModel(len(SELECTED_FEATURES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, scaler
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f) 

app = Flask(__name__)
app.secret_key = "replace-this-with-a-secure-key"
app.config['LAST_RESULT'] = None

def build_last_windows(test_df, rul_df):
    X, y_true, units = [], [], []

    rul_values = rul_df.iloc[:, 0].values
    test_units = sorted(test_df["unit"].unique())
    rul_map = {u: min(int(r), RUL_CLIP) for u, r in zip(test_units, rul_values)}

    for uid in test_units:
        unit_df = test_df[test_df["unit"] == uid]
        feats = unit_df[SELECTED_FEATURES].values

        if len(feats) < SEQ_LEN:
            pad = np.repeat(feats[[0]], SEQ_LEN - len(feats), axis=0)
            window = np.vstack([pad, feats])
        else:
            window = feats[-SEQ_LEN:]

        X.append(window)
        y_true.append(rul_map[uid])
        units.append(uid)

    return np.array(X, dtype=np.float32), np.array(y_true, dtype=np.float32), units

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img

# ================= ALERTS =================
def generate_alerts(units, preds, true_rul, warning, critical):
    alerts = []
    for u, p, t in zip(units, preds, true_rul):
        if p <= critical:
            level, msg, pr = "Critical", "Immediate maintenance required!", 1
        elif p <= warning:
            level, msg, pr = "Warning", "Maintenance soon recommended", 2
        else:
            level, msg, pr = "Normal", "No immediate action", 3

        alerts.append({
            "unit": int(u),
            "predicted_rul": float(p),
            "true_rul": float(t),
            "level": level,
            "message": msg,
            "priority": pr,
            "error": float(abs(p - t))
        })

    return sorted(alerts, key=lambda x: (x["priority"], x["predicted_rul"]))

# ================= MAIN FD002 RUN =================
def run_fd002(unseen_file, true_file, warning_th, critical_th):
    try:
        model, scaler = load_model_and_scaler()
    except Exception as e:
        flash(str(e))
        return redirect(url_for('index'))
    try:
        cols_full = ["unit","cycle"] + [f"op_setting_{i}" for i in range(1,4)] + [f"sensor_{i}" for i in range(1,22)]
        unseen_df = pd.read_csv(io.StringIO(unseen_file.stream.read().decode('utf-8')), sep='\s+', header=None).dropna(axis=1, how='all')
        unseen_df = unseen_df.iloc[:, :len(cols_full)]
        unseen_df.columns = cols_full
        unseen_df = unseen_df[["unit","cycle"] + SELECTED_FEATURES]
        unseen_df[SELECTED_FEATURES] = scaler.transform(unseen_df[SELECTED_FEATURES])
    except Exception as e:
        flash(f"Error reading unseen file: {e}")
        return redirect(url_for('index'))
    try:
        true_vals = np.loadtxt(io.StringIO(true_file.stream.read().decode('utf-8')))
        true_df = pd.DataFrame(true_vals)
    except Exception as e:
        flash(f"Error reading true RUL file: {e}")
        return redirect(url_for('index'))
    X_unseen, true_vals, units = build_last_windows(unseen_df, true_df)
    X_t = torch.tensor(X_unseen, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy().flatten().clip(max=RUL_CLIP)
    rmse = math.sqrt(mean_squared_error(true_vals, preds))
    alerts = generate_alerts(units, preds, true_vals, warning_th, critical_th)

    return {
       'units': units,
        'preds': preds.tolist(),
        'true_rul': true_vals.tolist(),
         'rmse': float(rmse),
        'thresholds': {'warning': warning_th, 'critical': critical_th},
        'alerts': alerts,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
