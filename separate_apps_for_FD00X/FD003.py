## FD003 pipline 
import os
import io
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request,redirect, url_for, flash, jsonify, send_file
import torch
import torch.nn as nn
from flask import render_template
import plotly
import plotly.graph_objs as go
import json
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
from sklearn.metrics import mean_squared_error
import math

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
def convert_numpy_to_python(obj):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    else:
        return obj
def generate_pdf_report(data):
    """Generate a comprehensive PDF report."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    story = []
    styles = getSampleStyleSheet()
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=30,
        alignment=1  # Center
    )
    story.append(Paragraph("Predictive Maintenance Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                          styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    # Summary Statistics
    alerts = data['alerts']
    critical_count = sum(1 for a in alerts if a['level'] == 'Critical')
    warning_count = sum(1 for a in alerts if a['level'] == 'Warning')
    normal_count = sum(1 for a in alerts if a['level'] == 'Normal')
    summary_data = [
        ['Metric', 'Value'],
        ['Total Engines', str(len(alerts))],
        ['Critical Alerts', str(critical_count)],
        ['Warning Alerts', str(warning_count)],
        ['Normal Status', str(normal_count)],
        ['Warning Threshold', f"{data['thresholds']['warning']} cycles"],
        ['Critical Threshold', f"{data['thresholds']['critical']} cycles"]
    ]
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    story.append(summary_table)
    story.append(Spacer(1, 0.4*inch))
    # Detailed Alert Table
    story.append(Paragraph("Detailed Engine Status", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))  
    alert_data = [['Engine ID', 'Predicted RUL', 'True RUL', 'Error', 'Status', 'Action']]
    for alert in alerts:
        alert_data.append([
            f"Engine {alert['unit']}",
            f"{alert['predicted_rul']:.1f}",
            f"{alert['true_rul']:.1f}",
            f"{alert['error']:.1f}",
            alert['level'],
            alert['message']
        ])
    alert_table = Table(alert_data, colWidths=[1*inch, 1*inch, 1*inch, 0.8*inch, 1*inch, 1.7*inch])
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]
    # Color code rows by status
    for i, alert in enumerate(alerts, start=1):
        if alert['level'] == 'Critical':
            table_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#ffebee')))
        elif alert['level'] == 'Warning':
            table_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#fff3e0')))
        else:
            table_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#e8f5e9')))
    alert_table.setStyle(TableStyle(table_style))
    story.append(alert_table)
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer
# Routes
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/upload', methods=['POST'])
def upload_and_run():
    if 'unseen' not in request.files or 'true_rul' not in request.files:
        flash('Please upload both files')
        return redirect(url_for('index'))
    fd_model = request.form.get('fd_model', 'FD003')
    warning_th = float(request.form.get('warning', DEFAULT_THRESHOLDS['warning']))
    critical_th = float(request.form.get('critical', DEFAULT_THRESHOLDS['critical']))
    unseen_file = request.files['unseen']
    true_file = request.files['true_rul']
    try:
        model, scaler = load_model_and_scaler(fd_model)
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
    except Exception as e:
        flash(f"Error reading true RUL file: {e}")
        return redirect(url_for('index'))
    X_unseen, units = build_last_windows(unseen_df)
    X_t = torch.tensor(X_unseen, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy().flatten().clip(max=RUL_CLIP)
    rmse = math.sqrt(mean_squared_error(true_vals, preds))
    alerts = generate_alerts(units, preds, true_vals, warning_th, critical_th)
    app.config['LAST_RESULT'] = {
        'units': units,
        'preds': preds.tolist(),
        'true_rul': true_vals.tolist(),
        'thresholds': {'warning': warning_th, 'critical': critical_th},
        'alerts': alerts,
        'rmse': float(rmse),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    return redirect(url_for('dashboard'))
@app.route('/data')
def data_endpoint():
    payload = app.config.get('LAST_RESULT')
    if not payload:
        return jsonify({'error': 'No inference data available'}), 404
    payload = convert_numpy_to_python(payload)
    return jsonify(payload)
@app.route('/dashboard')
def dashboard():
    data = app.config.get('LAST_RESULT')

    if not data:
        flash('No inference data available.')
        return redirect(url_for('index'))

    data = convert_numpy_to_python(data)
    rmse = data.get('rmse',None)

    units = data['units']
    preds = data['preds']
    true_rul = data['true_rul']
    alerts = data['alerts']
    # Main RUL comparison chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=units, y=true_rul, 
        mode='lines+markers', 
        name='True RUL',
        line=dict(color='#2196F3', width=2),
        marker=dict(size=8)
    ))
    fig1.add_trace(go.Scatter(
        x=units, y=preds, 
        mode='lines+markers', 
        name='Predicted RUL',
        line=dict(color='#FF5722', width=2),
        marker=dict(size=8)
    ))
    
    fig1.update_layout(
        title='RUL Prediction vs Actual',
        xaxis_title='Engine Unit',
        yaxis_title='Remaining Useful Life (cycles)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    # Priority distribution chart
    priority_counts = {'Critical': 0, 'Warning': 0, 'Normal': 0}
    for alert in alerts:
        priority_counts[alert['level']] += 1
    fig2 = go.Figure(data=[go.Pie(
        labels=list(priority_counts.keys()),
        values=list(priority_counts.values()),
        marker=dict(colors=['#dc3545', '#ffc107', '#28a745']),
        hole=0.4
    )])
    fig2.update_layout(title='Alert Distribution', height=350)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("dashboard.html",
    data=data,
    alerts=alerts,
    rmse=rmse,
    graphJSON1=graphJSON1,
    graphJSON2=graphJSON2)
@app.route('/download-report')
def download_report():
    """Download comprehensive PDF report."""
    data = app.config.get('LAST_RESULT')
    if not data:
        flash('No data available for report generation.')
        return redirect(url_for('index'))
    try:
        pdf_buffer = generate_pdf_report(data)
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'maintenance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
    except Exception as e:
        flash(f'Error generating report: {str(e)}')
        return redirect(url_for('dashboard'))
@app.route('/download-csv')
def download_csv():
    """Download results as CSV."""
    data = app.config.get('LAST_RESULT')
    if not data:
        flash('No data available.')
        return redirect(url_for('index'))
    alerts = data['alerts']
    df = pd.DataFrame(alerts)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'rul_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
