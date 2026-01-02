import io, math, pickle,os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from datetime import datetime
from flask import Flask
# Configuration
SEQ_LEN = 80
RUL_CLIP = 145
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLDS = {"warning": 50, "critical": 20}
MODEL_PATH = r"model\best_gru_fd003.pth"
SCALER_PATH = r"scaler\scaler_fd003_hyper.pkl"

SELECTED_FEATURES = [
    "op_setting_1","op_setting_2","op_setting_3",
    "sensor_2","sensor_3","sensor_4","sensor_7","sensor_8",
    "sensor_9","sensor_11","sensor_12","sensor_13",
    "sensor_14","sensor_15"
]

# =======================
# MODEL
# =======================
class SmallGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

app = Flask(__name__)
app.secret_key = "replace-this-with-a-secure-key"
app.config['LAST_RESULT'] = None
# Helper Functions
def load_model_and_scaler(fd_model="FD003"):
    model_path = MODEL_PATH
    scaler_path = SCALER_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    model = SmallGRU(len(SELECTED_FEATURES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler
def build_last_windows(df):
    X, units = [], sorted(df["unit"].unique())

    for uid in units:
        u_df = df[df["unit"] == uid].reset_index(drop=True)

        if len(u_df) < SEQ_LEN:
            pad = np.repeat(
                u_df[SELECTED_FEATURES].iloc[[0]].values,
                SEQ_LEN - len(u_df),
                axis=0
            )
            window = np.vstack([pad, u_df[SELECTED_FEATURES].values])
        else:
            window = u_df[SELECTED_FEATURES].values[-SEQ_LEN:]

        X.append(window.astype(np.float32))

    return np.array(X), units
def generate_alerts(units, preds, true_rul, warning_thresh=50, critical_thresh=25):
    """Generate maintenance alerts with priority scoring."""
    alerts = []
    for u, p, t in zip(units, preds, true_rul):
        if p <= critical_thresh:
            level = "Critical"
            msg = "Immediate maintenance required!"
            priority = 1
            color = "#dc3545"
        elif p <= warning_thresh:
            level = "Warning"
            msg = "Maintenance soon recommended"
            priority = 2
            color = "#ffc107"
        else:
            level = "Normal"
            msg = "No immediate action"
            priority = 3
            color = "#28a745"
        # Calculate accuracy
        error = abs(p - t)
        accuracy = max(0, 100 - (error / t * 100)) if t > 0 else 0
        alerts.append({
            'unit': int(u),
            'predicted_rul': float(p),
            'true_rul': float(t),
            'level': level,
            'message': msg,
            'priority': priority,
            'color': color,
            'error': float(error),
            'accuracy': float(accuracy)
        })
    # Sort by priority
    alerts.sort(key=lambda x: (x['priority'], x['predicted_rul']))
    return alerts
# ================= MAIN FD003 RUN =================
def run_fd003(unseen_file, true_file, warning_th, critical_th):
    model, scaler = load_model_and_scaler()

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

    true_df = np.loadtxt(io.StringIO(true_file.stream.read().decode()))

    X, units = build_last_windows(unseen_df)

    with torch.no_grad():
        preds = model(torch.tensor(X).to(DEVICE)) \
                    .cpu().numpy().flatten().clip(max=RUL_CLIP)

    rmse = math.sqrt(mean_squared_error(true_df, preds))
    alerts = generate_alerts(units, preds, true_df, warning_th, critical_th)

    return {
        "units": units,
        "preds": preds.tolist(),
        "true_rul": true_df.tolist(),
        "alerts": alerts,
        "rmse": float(rmse),
        "thresholds": {"warning": warning_th, "critical": critical_th},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
