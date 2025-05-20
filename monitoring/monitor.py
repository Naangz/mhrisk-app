import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# Load datasets
ref_data = pd.read_csv("monitoring/reference.csv")
cur_data = pd.read_csv("monitoring/current.csv")

# Generate monitoring report
monitor_report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])
monitor_report.run(reference_data=ref_data, current_data=cur_data)
monitor_report.save_html("monitoring/dashboard.html")

print("Monitoring report saved to monitoring/dashboard.html")
