import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import skops.io as sio
import shap
import pickle
import os
import whylogs as why
import xgboost as xgb
import lightgbm as lgb
import json
from datetime import datetime


def load_and_prepare_data():
    """Load dan prepare mental health dataset dengan WhyLogs monitoring"""
    
    # Try to load dataset dari folder data
    data_files = [
        "data/mental_health_lite.csv",
        "data/mental_health_life_cut.csv"
    ]
    
    df = None
    for file_path in data_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded from: {file_path}")
            break
    
    if df is None:
        raise FileNotFoundError("No dataset found in data/ folder")
    
    # Initialize WhyLogs dengan LOCAL session
    try:
        why.init(allow_local=True, allow_anonymous=False)
        print("✅ WhyLogs session initialized (LOCAL)")

        # Create WhyLogs profile
        profile = why.log(df)

        # Create directory if not exists
        os.makedirs("monitoring/whylogs_profiles", exist_ok=True)
        
        # Save profile
        profile_view = profile.view()
        profile_view.write("monitoring/whylogs_profiles/training_data_profile")
        print("✅ WhyLogs profile saved successfully")

    except Exception as e:
        print(f"⚠️ WhyLogs error: {e}")
        print("Continuing without WhyLogs profiling...")
        
        # Fallback: Simple data profiling
        print(f"📊 Dataset Info:")
        print(f"  - Shape: {df.shape}")
        print(f"  - Missing values: {df.isnull().sum().sum()}")
        print(f"  - Columns: {list(df.columns)}")
    
    # Encode categorical variables
    encoders = {}
    categorical_cols = ["gender", "employment_status", "work_environment", "mental_health_history", "seeks_treatment"]

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col])
            encoders[col] = le
            print(f"✅ Encoded {col}: {len(le.classes_)} classes")

    # Encode target variable
    target_col = None
    possible_targets = ['mental_health_risk', 'risk_level', 'target']
    
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    if target_col:
        le_risk = LabelEncoder()
        df['risk_encoded'] = le_risk.fit_transform(df[target_col])
        encoders['risk'] = le_risk
        print(f"✅ Encoded target ({target_col}): {le_risk.classes_}")
    else:
        print("⚠️ No target column found, using dummy target")
        df['risk_encoded'] = np.random.choice([0, 1, 2], size=len(df))
        encoders['risk'] = LabelEncoder().fit(['Low', 'Medium', 'High'])
    
    # Save encoders
    os.makedirs("model", exist_ok=True)
    with open("model/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    return df, encoders


def create_model_pipelines():
    """Create multiple ML pipelines"""

    models = {
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    xgb.XGBClassifier(
                        objective="multi:softprob",
                        learning_rate=0.1,
                        max_depth=6,
                        n_estimators=100,
                        colsample_bytree=0.8,
                        subsample=0.8,
                        random_state=42,
                        eval_metric="mlogloss",
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    lgb.LGBMClassifier(
                        objective="multiclass",
                        num_class=3,
                        learning_rate=0.1,
                        max_depth=6,
                        n_estimators=100,
                        feature_fraction=0.8,
                        bagging_fraction=0.8,
                        bagging_freq=5,
                        random_state=42,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
    }

    return models

def train_and_select_best_model():
    """Training multiple models dan pilih yang terbaik"""
    
    print("📊 Loading and preparing data...")
    df, encoders = load_and_prepare_data()
    
    # Identify numeric columns for features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target columns from features
    feature_columns = [col for col in numeric_cols 
                      if not col.endswith('_encoded') or col != 'risk_encoded']
    
    # Add encoded categorical features
    encoded_cols = [col for col in df.columns if col.endswith('_encoded') and col != 'risk_encoded']
    feature_columns.extend(encoded_cols)
    
    # Filter existing columns
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"📋 Available features: {len(available_features)}")
    print(f"Features: {available_features}")
    
    if len(available_features) == 0:
        raise ValueError("No feature columns found in dataset")
    
    X = df[available_features]
    y = df['risk_encoded']
    
    print(f"📊 Dataset shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create model pipelines
    models = create_model_pipelines()
    
    # Train and evaluate models
    model_scores = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("🔄 Training and evaluating models...")
    
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")

        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()

            model_scores[name] = {
                "cv_scores": cv_scores.tolist(),
                "mean_accuracy": mean_score,
                "std_accuracy": std_score,
                "model": model,
            }

            print(f"✅ {name} - CV Accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")

        except Exception as e:
            print(f"❌ {name} failed: {e}")
            model_scores[name] = {
                "cv_scores": [0.0],
                "mean_accuracy": 0.0,
                "std_accuracy": 0.0,
                "model": model,
                "error": str(e),
            }
    
    # Select best model
    valid_models = {k: v for k, v in model_scores.items() if 'error' not in v}
    
    if not valid_models:
        raise ValueError("No models trained successfully")
    
    best_model_name = max(valid_models.keys(), 
                         key=lambda x: valid_models[x]['mean_accuracy'])
    best_model = valid_models[best_model_name]['model']
    best_score = valid_models[best_model_name]['mean_accuracy']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"📊 CV Accuracy: {best_score:.4f}")
    
    # Train best model
    best_model.fit(X_train, y_train)
    
    # Final evaluation
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n📊 Final Test Results:")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Test F1 Score: {test_f1:.4f}")

    # Create results directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("explanations", exist_ok=True)
    
    # Save results
    comparison_results = {
        "timestamp": datetime.now().isoformat(),
        "best_model": best_model_name,
        "model_scores": {
            name: {"mean_accuracy": scores["mean_accuracy"], "std_accuracy": scores["std_accuracy"]}
            for name, scores in valid_models.items()
        },
        "final_test_accuracy": test_accuracy,
        "final_test_f1": test_f1,
    }
    
    with open("results/model_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    # Save metrics
    with open("results/metrics.txt", "w") as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"CV Accuracy: {best_score:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n\n")
        f.write("Model Comparison:\n")
        for name, scores in valid_models.items():
            f.write(f"{name}: {scores['mean_accuracy']:.4f} (+/- {scores['std_accuracy']*2:.4f})\n")
        f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Save model
    sio.dump(best_model, "model/mental_health_pipeline.skops")
    
    # Save metadata
    metadata = {
        "best_model_name": best_model_name,
        "feature_columns": available_features,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open("model/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    with open("model/feature_columns.pkl", "wb") as f:
        pickle.dump(available_features, f)

    print("✅ Model training completed successfully!")

    return best_model, best_model_name, test_accuracy, test_f1

if __name__ == "__main__":
    try:
        best_model, best_model_name, accuracy, f1 = train_and_select_best_model()
        print(f"\n🎉 Training completed successfully!")
        print(f"🏆 Best model: {best_model_name}")
        print(f"📊 Final accuracy: {accuracy:.4f}")
        print(f"📊 Final F1 score: {f1:.4f}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
