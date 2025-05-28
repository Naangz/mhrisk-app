import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import whylogs as why
from whylogs import ResultSet
from whylogs import log
from datetime import datetime
import shap
from skops.io import dump, load
import numpy as np

# 1. Load CSV
file_path = "data/mental_health_lite.csv"
df = pd.read_csv(file_path)

# 3. Encode target
le_target = LabelEncoder()
df["mental_health_risk"] = le_target.fit_transform(df["mental_health_risk"])

# 4. Split fitur dan target
X = df.drop(columns=["mental_health_risk"])
y = df["mental_health_risk"]

# 5. Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. WhyLogs - Logging data train
log(X_train).write(f"whylogs_profile_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# 7. Encode fitur kategorikal
label_encoders = {}
for col in X_train.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# 8. Pipeline Random Forest
pipeline_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42))
])
pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)

# 9. Pipeline XGBoost
pipeline_xgb = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])
pipeline_xgb.fit(X_train, y_train)
y_pred_xgb = pipeline_xgb.predict(X_test)

# 10. WhyLogs - Logging prediksi
pred_df = pd.DataFrame({
    "y_test": y_test,
    "y_pred_rf": y_pred_rf,
    "y_pred_xgb": y_pred_xgb
})
log(pred_df).write(f"whylogs_profile_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# 11. Evaluasi model
print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf, target_names=le_target.classes_))

print("\nXGBoost Report:")
print(classification_report(y_test, y_pred_xgb, target_names=le_target.classes_))

# Hitung akurasi masing-masing model
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

print(f"Akurasi Random Forest: {acc_rf:.4f}")
print(f"Akurasi XGBoost: {acc_xgb:.4f}")

# Pilih model dengan akurasi terbaik
if acc_rf >= acc_xgb:
    best_model = pipeline_rf
    best_model_name = "random_forest"
    print("Model terbaik: Random Forest")
else:
    best_model = pipeline_xgb
    best_model_name = "xgboost"
    print("Model terbaik: XGBoost")

# Simpan hanya model terbaik
from skops.io import dump
dump(best_model, f"model/model_best_{best_model_name}.skops")