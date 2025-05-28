import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import skops.io as sio
import shap
import pickle
import os
import whylogs as why
from whylogs.core.constraints import ConstraintsBuilder
from whylogs.core.constraints.factories import greater_than_number, smaller_than_number
import xgboost as xgb
import lightgbm as lgb
import json
from datetime import datetime


def load_and_prepare_data():
    """Load dan prepare mental health dataset"""
    df = pd.read_csv("data/mental_health_lite.csv")

    # Log data dengan WhyLogs
    profile = why.log(df)
    profile.write("monitoring/whylogs_profiles/training_data_profile")

    # Encode categorical variables
    le_gender = LabelEncoder()
    le_employment = LabelEncoder()
    le_work_env = LabelEncoder()
    le_history = LabelEncoder()
    le_treatment = LabelEncoder()
    le_risk = LabelEncoder()

    df["gender_encoded"] = le_gender.fit_transform(df["gender"])
    df["employment_encoded"] = le_employment.fit_transform(df["employment_status"])
    df["work_env_encoded"] = le_work_env.fit_transform(df["work_environment"])
    df["history_encoded"] = le_history.fit_transform(df["mental_health_history"])
    df["treatment_encoded"] = le_treatment.fit_transform(df["seeks_treatment"])
    df["risk_encoded"] = le_risk.fit_transform(df["mental_health_risk"])

    # Save encoders
    encoders = {
        "gender": le_gender,
        "employment": le_employment,
        "work_env": le_work_env,
        "history": le_history,
        "treatment": le_treatment,
        "risk": le_risk,
    }

    with open("model/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    return df, encoders


def evaluate_models_with_cv(models, X, y, cv_folds=5):
    """Evaluate multiple models menggunakan cross-validation"""

    model_scores = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    print("Evaluating models dengan cross-validation...")
    print("=" * 60)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

        mean_score = cv_scores.mean()
        std_score = cv_scores.std()

        model_scores[name] = {
            "cv_scores": cv_scores.tolist(),
            "mean_accuracy": mean_score,
            "std_accuracy": std_score,
            "model": model,
        }

        print(f"{name} - CV Accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")

    return model_scores


def train_and_select_best_model():
    """Training multiple models dan pilih yang terbaik"""

    # Load data
    df, encoders = load_and_prepare_data()

    # Prepare features dan target
    feature_columns = [
        "age",
        "gender_encoded",
        "employment_encoded",
        "work_env_encoded",
        "history_encoded",
        "treatment_encoded",
        "stress_level",
        "sleep_hours",
        "physical_activity_days",
        "depression_score",
        "anxiety_score",
        "social_support_score",
        "productivity_score",
    ]

    X = df[feature_columns]
    y = df["risk_encoded"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create model pipelines
    models = create_model_pipelines()

    # Evaluate models dengan cross-validation
    model_scores = evaluate_models_with_cv(models, X_train, y_train)

    # Pilih model terbaik berdasarkan mean CV accuracy
    best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]["mean_accuracy"])
    best_model = model_scores[best_model_name]["model"]
    best_score = model_scores[best_model_name]["mean_accuracy"]

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"CV Accuracy: {best_score:.4f}")
    print(f"{'='*60}")

    # Train best model pada full training data
    best_model.fit(X_train, y_train)

    # Final evaluation pada test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)

    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nFinal Test Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    # Save model comparison results
    comparison_results = {
        "timestamp": datetime.now().isoformat(),
        "best_model": best_model_name,
        "model_scores": {
            name: {"mean_accuracy": scores["mean_accuracy"], "std_accuracy": scores["std_accuracy"]}
            for name, scores in model_scores.items()
        },
        "final_test_accuracy": test_accuracy,
        "final_test_f1": test_f1,
    }

    with open("results/model_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)

    # Save detailed metrics
    with open("results/metrics.txt", "w") as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"CV Accuracy: {best_score:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n\n")
        f.write("Model Comparison:\n")
        for name, scores in model_scores.items():
            f.write(f"{name}: {scores['mean_accuracy']:.4f} (+/- {scores['std_accuracy']*2:.4f})\n")
        f.write(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # Create comparison visualization
    create_model_comparison_plot(model_scores, best_model_name)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"])
    disp.plot()
    plt.title(f"Mental Health Risk Prediction - {best_model_name}\nTest Accuracy: {test_accuracy:.3f}")
    plt.savefig("results/model_results.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Generate SHAP explanations untuk best model
    generate_shap_explanations(best_model, X_train, X_test, feature_columns, best_model_name)

    # Save best model
    sio.dump(best_model, "model/mental_health_pipeline.skops")

    # Save metadata
    metadata = {
        "best_model_name": best_model_name,
        "feature_columns": feature_columns,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "timestamp": datetime.now().isoformat(),
    }

    with open("model/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open("model/feature_columns.pkl", "wb") as f:
        pickle.dump(feature_columns, f)

    return best_model, best_model_name, test_accuracy, test_f1


def create_model_comparison_plot(model_scores, best_model_name):
    """Create visualization untuk model comparison"""

    models = list(model_scores.keys())
    accuracies = [model_scores[model]["mean_accuracy"] for model in models]
    std_devs = [model_scores[model]["std_accuracy"] for model in models]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["gold" if model == best_model_name else "lightblue" for model in models]
    bars = ax.bar(models, accuracies, yerr=std_devs, capsize=5, color=colors, edgecolor="black", linewidth=1)

    # Highlight best model
    for i, (model, bar) in enumerate(zip(models, bars)):
        if model == best_model_name:
            bar.set_color("gold")
            bar.set_edgecolor("red")
            bar.set_linewidth(2)

        # Add value labels on bars
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std_devs[i],
            f"{accuracies[i]:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Cross-Validation Accuracy")
    ax.set_title("Model Performance Comparison\n(Error bars show ±1 std dev)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Add legend
    ax.text(
        0.02,
        0.98,
        f"Best Model: {best_model_name}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="gold", alpha=0.8),
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()


def generate_shap_explanations(model, X_train, X_test, feature_columns, model_name):
    """Generate SHAP explanations untuk best model"""

    try:
        # Create SHAP explainer berdasarkan model type
        if "XGBoost" in model_name:
            explainer = shap.TreeExplainer(model.named_steps["classifier"])
            shap_values = explainer.shap_values(X_test.sample(200))
        elif "LightGBM" in model_name:
            explainer = shap.TreeExplainer(model.named_steps["classifier"])
            shap_values = explainer.shap_values(X_test.sample(200))
        else:  # RandomForest
            explainer = shap.Explainer(model, X_train.sample(100))
            shap_values = explainer(X_test.sample(200))

        # Save SHAP values
        with open("explanations/shap_values.pkl", "wb") as f:
            pickle.dump(shap_values, f)

        # Generate SHAP plots
        sample_data = X_test.sample(200)

        # Summary plot
        plt.figure(figsize=(12, 8))
        if "XGBoost" in model_name or "LightGBM" in model_name:
            # For tree models, use class 0 (or average across classes)
            if len(shap_values) > 1:  # Multi-class
                shap.summary_plot(shap_values[1], sample_data, feature_names=feature_columns, show=False)
            else:
                shap.summary_plot(shap_values, sample_data, feature_names=feature_columns, show=False)
        else:
            shap.summary_plot(shap_values, sample_data, feature_names=feature_columns, show=False)

        plt.title(f"SHAP Summary Plot - {model_name}\nMental Health Risk Factors")
        plt.tight_layout()
        plt.savefig("results/shap_summary.png", dpi=120, bbox_inches="tight")
        plt.close()

        # Feature importance plot
        plt.figure(figsize=(10, 6))
        if "XGBoost" in model_name or "LightGBM" in model_name:
            if len(shap_values) > 1:
                shap.summary_plot(shap_values[1], sample_data, plot_type="bar", feature_names=feature_columns, show=False)
            else:
                shap.summary_plot(shap_values, sample_data, plot_type="bar", feature_names=feature_columns, show=False)
        else:
            shap.summary_plot(shap_values, sample_data, plot_type="bar", feature_names=feature_columns, show=False)

        plt.title(f"SHAP Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig("results/shap_importance.png", dpi=120, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")
        # Create fallback feature importance plot
        create_fallback_feature_importance(model, feature_columns, model_name)


def create_fallback_feature_importance(model, feature_columns, model_name):
    """Create fallback feature importance jika SHAP gagal"""

    try:
        # Get feature importance dari model
        if hasattr(model.named_steps["classifier"], "feature_importances_"):
            importances = model.named_steps["classifier"].feature_importances_

            # Create feature importance plot
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_columns[i] for i in indices], rotation=45)
            plt.title(f"Feature Importance - {model_name}")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.savefig("results/feature_importance.png", dpi=120, bbox_inches="tight")
            plt.close()

    except Exception as e:
        print(f"Error creating fallback feature importance: {e}")


if __name__ == "__main__":
    best_model, best_model_name, accuracy, f1 = train_and_select_best_model()
    print(f"\nTraining completed!")
    print(f"Best model: {best_model_name}")
    print(f"Final accuracy: {accuracy:.4f}")
