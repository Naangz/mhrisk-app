import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

# === Load Dataset ===
file_path = "data/mental_health_lite.csv"
df = pd.read_csv(file_path)

# === Encode Categorical Columns ===
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
label_encoders = {}
df_encoded = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# === Split Feature & Target ===
target_column = "mental_health_risk"
X = df_encoded.drop(columns=[target_column])
y = df_encoded[target_column]
target_encoder = label_encoders[target_column]
y_class_names = target_encoder.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === MLflow Config ===
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Use local tracking server
mlflow.set_experiment("Mental Health Risk Classification")

# === Model 1: Random Forest ===
with mlflow.start_run(run_name="RandomForest"):
    rf_model = RandomForestClassifier(random_state=42)
    rf_pipeline = Pipeline([("model", rf_model)])
    rf_pipeline.fit(X_train, y_train)

    y_pred_rf = rf_pipeline.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average="macro")

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", acc_rf)
    mlflow.log_metric("f1_score", f1_rf)
    mlflow.sklearn.log_model(rf_pipeline, "random_forest_model")

    print("Random Forest Metrics:")
    print(classification_report(y_test, y_pred_rf, target_names=y_class_names))

# === Model 2: XGBoost ===
with mlflow.start_run(run_name="XGBoost"):
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    xgb_pipeline = Pipeline([("model", xgb_model)])
    xgb_pipeline.fit(X_train, y_train)

    y_pred_xgb = xgb_pipeline.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb, average="macro")

    mlflow.log_param("model", "XGBoost")
    mlflow.log_metric("accuracy", acc_xgb)
    mlflow.log_metric("f1_score", f1_xgb)
    mlflow.sklearn.log_model(xgb_pipeline, "xgboost_model")

    print("XGBoost Metrics:")
    print(classification_report(y_test, y_pred_xgb, target_names=y_class_names))
