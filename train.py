import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import skops.io as sio
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset

# === Load Dataset ===
df = pd.read_csv("Data/mental_health_lite.csv")

# === Select & reorder features ===
features = [
    "age", "gender", "employment_status", "work_environment", "mental_health_history",
    "seeks_treatment", "stress_level", "sleep_hours", "physical_activity_days",
    "depression_score", "anxiety_score", "social_support_score", "productivity_score"
]
target = "mental_health_risk"
df = df[features + [target]].dropna()

# === Split Train-Test ===
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Preprocessing ===
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(), cat_cols),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols)
])

# === Pipeline ===
pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipe.fit(X_train, y_train)

# === Evaluation ===
y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print(f"Accuracy: {accuracy:.2f}, F1: {f1:.2f}")

with open("Results/metrics.txt", "w") as f:
    f.write(f"Accuracy = {accuracy:.2f}, F1 Score = {f1:.2f}")

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred, labels=pipe.named_steps['model'].classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.named_steps['model'].classes_)
disp.plot()
plt.savefig("results/model_results.png", dpi=120)

# === Save Model ===
sio.dump(pipe, "model/mental_health_model.skops")

# === Evidently Report ===
report = Report(metrics=[ClassificationPreset()])
report.run(y_true=y_test, y_pred=y_pred, data=X_test)
report.save_html("results/evidently_report.html")

# === Save Reference Data for Monitoring ===
X_test_copy = X_test.copy()
X_test_copy["target"] = y_test
X_test_copy.to_csv("monitoring/reference.csv", index=False)