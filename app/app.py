import gradio as gr
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import tempfile
import warnings
from io import BytesIO

import skops.io as sio
from skops.io import get_untrusted_types, load

warnings.filterwarnings("ignore")


# ---------------------------------------------------
# Core predictor
# ---------------------------------------------------
class MentalHealthPredictor:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.feature_columns = None
        self.model_metadata = None
        self._load_model_artifacts()

    # ---------- model / artefact I/O ----------
    def _load_model_artifacts(self):
        try:
            model_path = "model/mental_health_pipeline.skops"
            if os.path.exists(model_path):
                untrusted_types = get_untrusted_types(file=model_path)
                self.model = load(model_path, trusted=untrusted_types)
                print("✅  Model loaded")
            else:
                print(f"❌  Model file not found: {model_path}")

            enc_path = "model/encoders.pkl"
            if os.path.exists(enc_path):
                with open(enc_path, "rb") as f:
                    self.encoders = pickle.load(f)
                print("✅  Encoders loaded")

            feat_path = "model/feature_columns.pkl"
            if os.path.exists(feat_path):
                with open(feat_path, "rb") as f:
                    self.feature_columns = pickle.load(f)
                print("✅  Feature columns loaded")

            meta_path = "model/model_metadata.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    self.model_metadata = json.load(f)
                print("✅  Metadata loaded")

        except Exception as e:
            print(f"❌  Error loading artefacts: {e}")

    # ---------- helpers ----------
    def _get_model_classes(self):
        """Return class labels in the exact order the model was trained with."""
        if hasattr(self.model, "classes_"):
            return list(self.model.classes_)
        # Model is a Pipeline
        try:
            return list(self.model.named_steps["classifier"].classes_)
        except Exception:
            return ["Low", "Medium", "High"]  # safe fallback

    @staticmethod
    def _normalise_class_name(name: str) -> str:
        """Map any variant to canonical ‘Low’, ‘Medium’, or ‘High’."""
        name = str(name).strip().lower()
        if name.startswith("low"):
            return "Low"
        if name.startswith("med"):
            return "Medium"
        if name.startswith("high"):
            return "High"
        # default
        return "Medium"

    # ---------- input prep ----------
    def _prepare_input(
        self,
        age,
        gender,
        employment,
        work_env,
        mental_history,
        seeks_treatment,
        stress_level,
        sleep_hours,
        physical_activity,
        depression_score,
        anxiety_score,
        social_support,
        productivity,
    ):
        """Build a single-row DataFrame ready for model.predict[_proba]."""
        try:
            row = {
                "age": float(age),
                "stress_level": float(stress_level),
                "sleep_hours": float(sleep_hours),
                "physical_activity_days": float(physical_activity),
                "depression_score": float(depression_score),
                "anxiety_score": float(anxiety_score),
                "social_support_score": float(social_support),
                "productivity_score": float(productivity),
                "gender": gender,
                "employment_status": employment,
                "work_environment": work_env,
                "mental_health_history": mental_history,
                "seeks_treatment": seeks_treatment,
            }
            df = pd.DataFrame([row])

            # encode categoricals
            if self.encoders:
                cat_cols = [
                    "gender",
                    "employment_status",
                    "work_environment",
                    "mental_health_history",
                    "seeks_treatment",
                ]
                for col in cat_cols:
                    if col in self.encoders and col in df.columns:
                        enc = self.encoders[col]
                        val = df[col].iloc[0]
                        df[f"{col}_encoded"] = (
                            enc.transform(df[col])
                            if val in enc.classes_
                            else 0
                        )

            # keep only expected columns
            if self.feature_columns:
                df = df[[c for c in self.feature_columns if c in df.columns]]
            else:
                df = df.select_dtypes(include=[np.number])

            return df

        except Exception as e:
            print(f"❌  Input preparation failed: {e}")
            return None

    # ---------- human-friendly texts ----------
    @staticmethod
    def _risk_dictionary():
        return {
            "Low": {
                "emoji": "🟢",
                "color": "#28a745",
                "description": "Your responses suggest a lower likelihood of mental-health concerns.",
                "recommendations": [
                    "Maintain healthy lifestyle habits",
                    "Keep regular physical activity and sleep routine",
                    "Stay socially connected",
                    "Use preventive stress-management techniques",
                ],
            },
            "Medium": {
                "emoji": "🟡",
                "color": "#ffc107",
                "description": "Some aspects may need attention.",
                "recommendations": [
                    "Consider speaking with a professional",
                    "Reduce stress via meditation or hobbies",
                    "Focus on sleep quality and balanced activity",
                    "Monitor emotional well-being regularly",
                ],
            },
            "High": {
                "emoji": "🔴",
                "color": "#dc3545",
                "description": "You may benefit from professional mental-health support.",
                "recommendations": [
                    "🚨  Contact a mental-health professional soon",
                    "Seek support from friends or crisis services",
                    "Practise immediate stress-relief techniques",
                    "Consider medical evaluation",
                ],
            },
        }

    # ---------- SHAP-style simple bar plot ----------
    @staticmethod
    def _make_bar_plot(df_row):
        try:
            names = df_row.columns.tolist()
            vals = df_row.iloc[0].values
            plt.figure(figsize=(10, 7))
            ypos = np.arange(len(names))
            colors = ["#ff7f7f" if v > np.median(vals) else "#7fbfff" for v in vals]
            plt.barh(ypos, vals, color=colors, edgecolor="black", alpha=0.7)
            plt.yticks(
                ypos,
                [n.replace("_encoded", "").replace("_", " ").title() for n in names],
            )
            plt.xlabel("Feature values")
            plt.title("Feature contribution overview")
            plt.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(tmp.name, dpi=140, bbox_inches="tight")
            return tmp.name
        except Exception as e:
            print(f"⚠️  Plot error: {e}")
            return None

    # ---------- main predict ----------
    def predict(
        self,
        age,
        gender,
        employment,
        work_env,
        mental_history,
        seeks_treatment,
        stress_level,
        sleep_hours,
        physical_activity,
        depression_score,
        anxiety_score,
        social_support,
        productivity,
    ):
        if self.model is None:
            return self._fallback()

        features = self._prepare_input(
            age,
            gender,
            employment,
            work_env,
            mental_history,
            seeks_treatment,
            stress_level,
            sleep_hours,
            physical_activity,
            depression_score,
            anxiety_score,
            social_support,
            productivity,
        )
        if features is None:
            return self._fallback()

        # raw prediction
        y_proba = self.model.predict_proba(features)[0]
        y_pred_idx = int(np.argmax(y_proba))
        model_classes = self._get_model_classes()

        # decode
        raw_class = model_classes[y_pred_idx]
        risk_class = self._normalise_class_name(raw_class)

        # probability dict → ordered list Low-Med-High
        canonical_order = ["Low", "Medium", "High"]
        proba_map = {self._normalise_class_name(c): p for c, p in zip(model_classes, y_proba)}
        ordered_proba = [proba_map.get(lvl, 0.0) for lvl in canonical_order]

        info = self._risk_dictionary()[risk_class]

        # ------------------ HTML result block ------------------
        result_html = f"""
        <div style="text-align:center; padding:20px; border-radius:10px;
             background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white;">
            <h2 style="margin:0">{info['emoji']} Mental Health Risk Assessment</h2>
            <h3 style="margin:10px 0; background:white; color:{info['color']};
                padding:8px 12px; border-radius:6px; display:inline-block">
                Risk Level: {risk_class}
            </h3>
        </div>
        <div style="background:#f8f9fa; padding:22px; border-radius:10px; margin-top:18px;">
            <h3 style="color:#2d3a4a; margin-top:0">📊 Probability Scores</h3>
        """

        for lvl, prob in zip(canonical_order, ordered_proba):
            emoji = "🟢" if lvl == "Low" else "🟡" if lvl == "Medium" else "🔴"
            bar = "#28a745" if lvl == "Low" else "#ffc107" if lvl == "Medium" else "#dc3545"
            width = prob * 100
            text_col = "white" if lvl in ("Low", "High") else "#222"
            result_html += f"""
            <div style="margin:10px 0;">
                <strong>{emoji} {lvl}:</strong> {prob:.1%}
                <div style="height:24px; background:#e9ecef; border-radius:10px; margin-top:4px;">
                    <div style="height:100%; width:{width}%; background:{bar};
                                border-radius:10px; display:flex; align-items:center;
                                justify-content:center; color:{text_col}; font-weight:bold;">
                        {prob:.1%}
                    </div>
                </div>
            </div>"""

        result_html += f"""
        </div>
        <div style="background:#fff8e1; padding:20px; border-radius:10px; margin-top:18px;
                    border-left:6px solid {info['color']}">
            <h3>🧠 Interpretation</h3>
            <p style="margin:0">{info['description']}</p>
        </div>"""

        # metadata (optional)
        if self.model_metadata:
            result_html += f"""
            <div style="background:#f8f9fa; padding:15px; border-radius:8px; margin-top:18px;
                        border-left:5px solid #007bff;">
                <h4 style="margin-top:0">🤖 Model Info</h4>
                <ul style="margin:6px 0 0 18px; padding:0; color:#222">
                    <li><strong>Model:</strong> {self.model_metadata.get('best_model_name','-')}</li>
                    <li><strong>Accuracy:</strong> {self.model_metadata.get('test_accuracy',0):.1%}</li>
                    <li><strong>Last updated:</strong> {self.model_metadata.get('timestamp','-')[:10]}</li>
                </ul>
            </div>"""

        # SHAP-like quick plot
        shap_img = self._make_bar_plot(features)

        # personalised recommendations
        rec_html = self._recommendations_block(info)

        return result_html, shap_img, rec_html

    # ---------- blocks for fallback / rec ----------
    def _fallback(self):
        msg = """
        <div style="background:#ffc107; padding:20px; border-radius:10px; text-align:center;">
            <h2>⚠️  Assessment temporarily unavailable</h2>
            <p>Please try again later or consult a healthcare professional.</p>
        </div>
        """
        return msg, None, self._general_recommendations()

    @staticmethod
    def _general_recommendations():
        return """
        <div style="background:white; padding:25px; border-radius:10px;">
            <h2>💡 General Mental-Health Tips</h2>
            <ul style="line-height:1.8;">
                <li>Maintain a regular sleep schedule (7-9 h per night)</li>
                <li>Exercise regularly (min. 30 min, 3-4 × week)</li>
                <li>Practise stress-management techniques</li>
                <li>Stay connected with friends and family</li>
                <li>Seek professional help when needed</li>
            </ul>
        </div>"""

    @staticmethod
    def _recommendations_block(info):
        rec_items = "".join(f"<li>{r}</li>" for r in info["recommendations"])
        return f"""
        <div style="background:#f8f9fa; padding:25px; border-radius:10px;">
            <h2 style="color:{info['color']}">💡 Personalised Recommendations</h2>
            <ul style="line-height:1.8; margin:12px 0;">{rec_items}</ul>
            <div style="background:#fff3cd; padding:14px; border-radius:8px;
                        border:1px solid #ffe8a1; color:#856404;">
                <em>⚠️  This assessment is informational only and does not replace professional advice.</em>
            </div>
        </div>"""


# ---------------------------------------------------
# Gradio UI
# ---------------------------------------------------
predictor = MentalHealthPredictor()

def _interface(
    age,
    gender,
    employment,
    work_env,
    mental_history,
    seeks_treatment,
    stress_level,
    sleep_hours,
    physical_activity,
    depression_score,
    anxiety_score,
    social_support,
    productivity,
):
    return predictor.predict(
        age,
        gender,
        employment,
        work_env,
        mental_history,
        seeks_treatment,
        stress_level,
        sleep_hours,
        physical_activity,
        depression_score,
        anxiety_score,
        social_support,
        productivity,
    )


with gr.Blocks(
    title="Mental Health Risk Identifier",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container {max-width:1200px;margin:auto;}
        .main-header {text-align:center;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                      color:white;padding:2rem;border-radius:15px;margin-bottom:2rem;}
    """,
) as demo:
    gr.HTML(
        """
    <div class="main-header">
        <h1 style="margin:0;font-size:2.5em">🧠 Mental Health Risk Identifier</h1>
        <p style="margin:1rem 0 0;font-size:1.2em;opacity:0.9">
            AI-powered mental-health risk assessment with explainable predictions
        </p>
    </div>"""
    )

    gr.Markdown("### 📝 Fill in the form to receive your personalised assessment")

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age", value=30, minimum=18, maximum=80, precision=0)
            gender = gr.Dropdown(
                ["Male", "Female", "Non-binary", "Prefer not to say"], label="Gender"
            )
            employment = gr.Dropdown(
                ["Employed", "Unemployed", "Student", "Self-employed"],
                label="Employment",
            )
            work_env = gr.Dropdown(
                ["On-site", "Remote", "Hybrid"], label="Work Environment"
            )
            mental_history = gr.Dropdown(
                ["Yes", "No"], label="Mental-Health History"
            )
            seeks_treatment = gr.Dropdown(
                ["Yes", "No"], label="Currently Seeking Treatment"
            )

        with gr.Column():
            stress_level = gr.Slider(1, 10, value=5, label="Stress Level")
            sleep_hours = gr.Slider(3, 12, value=7, step=0.5, label="Sleep Hours")
            physical_activity = gr.Slider(
                0, 7, value=3, step=1, label="Physical-Activity Days"
            )
            depression_score = gr.Slider(0, 50, value=10, label="Depression Score")
            anxiety_score = gr.Slider(0, 50, value=10, label="Anxiety Score")
            social_support = gr.Slider(
                0, 100, value=70, step=5, label="Social-Support Score"
            )
            productivity = gr.Slider(
                0, 100, value=70, step=5, label="Productivity Score"
            )

    with gr.Row():
        assess_btn = gr.Button("🎯 Assess Risk", variant="primary")
        clear_btn = gr.Button("🔄 Clear", variant="secondary")

    with gr.Row():
        with gr.Column(scale=2):
            result_html = gr.HTML()
            rec_html = gr.HTML()
        with gr.Column(scale=1):
            shap_img = gr.Image(type="filepath", height=400)

    assess_btn.click(
        fn=_interface,
        inputs=[
            age,
            gender,
            employment,
            work_env,
            mental_history,
            seeks_treatment,
            stress_level,
            sleep_hours,
            physical_activity,
            depression_score,
            anxiety_score,
            social_support,
            productivity,
        ],
        outputs=[result_html, shap_img, rec_html],
    )

    clear_btn.click(
        fn=lambda: [
            30,
            "Male",
            "Employed",
            "On-site",
            "No",
            "No",
            5,
            7,
            3,
            10,
            10,
            70,
            70,
        ],
        outputs=[
            age,
            gender,
            employment,
            work_env,
            mental_history,
            seeks_treatment,
            stress_level,
            sleep_hours,
            physical_activity,
            depression_score,
            anxiety_score,
            social_support,
            productivity,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
