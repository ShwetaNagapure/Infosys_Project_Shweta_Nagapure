import io, math, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from datetime import datetime

# ================= CONFIG =================
SEQ_LEN = 80
RUL_CAP = 125
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "model/best_model_fd004.pth"
SCALER_PATH = "scaler/scaler_fd004.pkl"

SELECTED_FEATURES = [
    "op_setting_1","op_setting_2","op_setting_3",
    "sensor_2","sensor_3","sensor_4","sensor_7","sensor_8",
    "sensor_9","sensor_11","sensor_12","sensor_13",
    "sensor_14","sensor_15"
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
    def __init__(self, input_dim):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        self.attn = Attention(256)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.cnn(x)
        x = x.permute(0,2,1)
        x,_ = self.lstm(x)
        x = self.attn(x)
        return self.fc(x)

# ================= LOAD =================
def load_fd004_model():
    model = SmallModel(len(SELECTED_FEATURES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

# ================= WINDOW =================
def build_last_windows(df):
    X, units = [], []
    for uid in sorted(df["unit"].unique()):
        feats = df[df["unit"] == uid][SELECTED_FEATURES].values
        if len(feats) < SEQ_LEN:
            pad = np.repeat(feats[[0]], SEQ_LEN - len(feats), axis=0)
            window = np.vstack([pad, feats])
        else:
            window = feats[-SEQ_LEN:]
        X.append(window.astype(np.float32))
        units.append(uid)
    return np.array(X), units

# ================= ALERTS =================
def generate_alerts(units, preds, true_rul, warning, critical):
    alerts = []
    for u,p,t in zip(units, preds, true_rul):
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

    return sorted(alerts, key=lambda x:(x["priority"], x["predicted_rul"]))

# ================= MAIN RUN =================
def run_fd004(unseen_file, true_file, warning_th, critical_th):
    model, scaler = load_fd004_model()

    cols = ["unit","cycle"] + \
           [f"op_setting_{i}" for i in range(1,4)] + \
           [f"sensor_{i}" for i in range(1,22)]

    unseen_df = pd.read_csv(
        io.StringIO(unseen_file.stream.read().decode()),
        sep=r"\s+", header=None
    ).iloc[:, :len(cols)]

    unseen_df.columns = cols
    unseen_df = unseen_df[["unit","cycle"] + SELECTED_FEATURES]
    unseen_df[SELECTED_FEATURES] = scaler.transform(unseen_df[SELECTED_FEATURES])

    true_vals = np.loadtxt(io.StringIO(true_file.stream.read().decode()))
    true_vals = np.minimum(true_vals, RUL_CAP)

    X, units = build_last_windows(unseen_df)

    with torch.no_grad():
        preds = model(torch.tensor(X).to(DEVICE)).cpu().numpy().flatten()

    rmse = math.sqrt(mean_squared_error(true_vals, preds))
    alerts = generate_alerts(units, preds, true_vals, warning_th, critical_th)

    return {
        "units": units,
        "preds": preds.tolist(),
        "true_rul": true_vals.tolist(),
        "alerts": alerts,
        "rmse": float(rmse),
        "thresholds": {"warning": warning_th, "critical": critical_th},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
