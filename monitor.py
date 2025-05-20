import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset

# === Load Dataset ===
df = pd.read_csv("Data/mental_health_lite.csv")

# === Encode Categorical Columns ===
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
df_encoded = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# === Train/Test Split ===
target_column = "mental_health_risk"
X = df_encoded.drop(columns=[target_column])
y = df_encoded[target_column]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# === Data Drift Report ===
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=X_train, current_data=X_test)
drift_report.save_html("Results/data_drift_report.html")

# === Data Test Suite ===
test_suite = TestSuite(tests=[DataStabilityTestPreset()])
test_suite.run(reference_data=X_train, current_data=X_test)
test_suite.save_html("Results/data_tests.html")

print("Monitoring complete. Reports saved to Results/")
