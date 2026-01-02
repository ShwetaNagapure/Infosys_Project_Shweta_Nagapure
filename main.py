from flask import *
import io,os
from s_model.FD001 import run_fd001
from s_model.FD002 import run_fd002
from s_model.FD003 import run_fd003
from s_model.FD004 import run_fd004
import numpy as np
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
import plotly.graph_objects as go
import plotly, json
from datetime import datetime

app = Flask(__name__)
app.secret_key = "rul_secret"


app = Flask(__name__)
app.secret_key = "rul_secret"
app.config["LAST_RESULT"] = None
def convert_numpy_to_python(obj):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(obj, Response):
        return obj   

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
        alignment=1  
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
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    fd_model = request.form.get("fd_model")
    warning = float(request.form.get("warning"))
    critical = float(request.form.get("critical"))

    unseen_file = request.files.get("unseen")
    true_rul_file = request.files.get("true_rul")

    if not unseen_file or not true_rul_file:
        flash("Files missing")
        return redirect(url_for("index"))

    try:
        if fd_model == "FD001":
            result = run_fd001(
                unseen_file.read(),
                true_rul_file.read(),
                warning,
                critical
            )
        elif fd_model == "FD002":
            result = run_fd002(unseen_file, true_rul_file, warning, critical)
        elif fd_model == "FD003":
            result = run_fd003(unseen_file, true_rul_file, warning, critical)
        elif fd_model == "FD004":
            result = run_fd004(unseen_file, true_rul_file, warning, critical)
        else:
            flash("FD model not implemented yet")
            return redirect(url_for("index"))

    except Exception as e:
        flash(f"Inference error: {e}")
        return redirect(url_for("index"))

    if not isinstance(result, dict):
        flash("Inference failed. Invalid result.")
        return redirect(url_for("index"))

    app.config["LAST_RESULT"] = result
    return redirect(url_for("dashboard"))

@app.route('/data')
def data_endpoint():
    payload = app.config.get("LAST_RESULT")

    if payload is None:
        return jsonify({"error": "No data"}), 404

    return jsonify(convert_numpy_to_python(payload))
@app.route('/dashboard')
def dashboard():
    data = app.config.get("LAST_RESULT")

    if data is None:
        flash("No inference data available.")
        return redirect(url_for("index"))

    rmse = data.get("rmse")
    units = data["units"]
    preds = data["preds"]
    true_rul = data["true_rul"]
    alerts = data["alerts"]
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
    if isinstance(data, Response):
        data = data.get_json()
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
