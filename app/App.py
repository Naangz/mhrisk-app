import gradio as gr
import pandas as pd
import numpy as np
import skops.io as sio
import pickle
import shap
import matplotlib.pyplot as plt
import io
import base64
import json

# Load model dan metadata
model = sio.load("../model/mental_health_pipeline.skops", trusted=True)

with open("../model/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("../model/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Load model metadata
with open("../model/model_metadata.json", "r") as f:
    model_metadata = json.load(f)

# Load model comparison results
try:
    with open("../Results/model_comparison.json", "r") as f:
        comparison_results = json.load(f)
except:
    comparison_results = None

def predict_mental_health_risk(age, gender, employment_status, work_environment, 
                             mental_health_history, seeks_treatment, stress_level,
                             sleep_hours, physical_activity_days, depression_score,
                             anxiety_score, social_support_score, productivity_score):
    
    # Encode categorical variables
    gender_encoded = encoders['gender'].transform([gender])[0]
    employment_encoded = encoders['employment'].transform([employment_status])[0]
    work_env_encoded = encoders['work_env'].transform([work_environment])[0]
    history_encoded = encoders['history'].transform([mental_health_history])[0]
    treatment_encoded = encoders['treatment'].transform([seeks_treatment])[0]
    
    # Create feature array
    features = np.array([[
        age, gender_encoded, employment_encoded, work_env_encoded,
        history_encoded, treatment_encoded, stress_level, sleep_hours,
        physical_activity_days, depression_score, anxiety_score,
        social_support_score, productivity_score
    ]])
    
    # Predict
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]
    
    # Decode prediction
    risk_levels = ['Low', 'Medium', 'High']
    predicted_risk = risk_levels[prediction]
    
    # Create probability dictionary
    prob_dict = {risk_levels[i]: float(prediction_proba[i]) for i in range(len(risk_levels))}
    
    # Generate SHAP explanation
    shap_explanation = generate_individual_shap_explanation(features)
    
    # Model info
    model_info = f"Model: {model_metadata['best_model_name']} | Accuracy: {model_metadata['test_accuracy']:.3f}"
    
    return predicted_risk, prob_dict, shap_explanation, model_info

def generate_individual_shap_explanation(features):
    """Generate SHAP explanation untuk prediksi individual"""
    try:
        # Create explainer dengan sample data
        sample_data = pd.DataFrame(np.random.randn(100, len(feature_columns)), columns=feature_columns)
        explainer = shap.Explainer(model, sample_data)
        
        # Get SHAP values
        shap_vals = explainer(features)
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap_vals[0], show=False)
        plt.title(f'Feature Contributions - {model_metadata["best_model_name"]}')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        return f"Error generating SHAP explanation: {str(e)}"

def show_model_comparison():
    """Show model comparison results"""
    if comparison_results:
        comparison_text = f"**Model Training Results**\n\n"
        comparison_text += f"**Best Model:** {comparison_results['best_model']}\n\n"
        comparison_text += f"**Model Comparison:**\n"
        
        for model_name, scores in comparison_results['model_scores'].items():
            comparison_text += f"- {model_name}: {scores['mean_accuracy']:.4f} (±{scores['std_accuracy']:.4f})\n"
        
        comparison_text += f"\n**Final Test Results:**\n"
        comparison_text += f"- Accuracy: {comparison_results['final_test_accuracy']:.4f}\n"
        comparison_text += f"- F1 Score: {comparison_results['final_test_f1']:.4f}\n"
        
        return comparison_text
    else:
        return "Model comparison data not available"

# Create Gradio interface
with gr.Blocks(title="Mental Health Risk Prediction - Multi-Model MLOps") as demo:
    gr.Markdown("# Mental Health Risk Prediction with Multi-Model Selection")
    gr.Markdown("Sistem MLOps yang menggunakan multiple algorithms (RandomForest, XGBoost, LightGBM) dan memilih model terbaik")
    
    with gr.Tab("Prediction"):
        with gr.Row():
            with gr.Column():
                age = gr.Slider(18, 80, value=30, label="Age")
                gender = gr.Dropdown(["Male", "Female", "Non-binary"], label="Gender")
                employment_status = gr.Dropdown(["Employed", "Unemployed", "Student", "Self-employed"], label="Employment Status")
                work_environment = gr.Dropdown(["On-site", "Remote", "Hybrid"], label="Work Environment")
                mental_health_history = gr.Dropdown(["Yes", "No"], label="Mental Health History")
                seeks_treatment = gr.Dropdown(["Yes", "No"], label="Seeks Treatment")
                stress_level = gr.Slider(1, 10, value=5, label="Stress Level")
                
            with gr.Column():
                sleep_hours = gr.Slider(3, 12, value=7, step=0.1, label="Sleep Hours")
                physical_activity_days = gr.Slider(0, 7, value=3, label="Physical Activity Days")
                depression_score = gr.Slider(0, 50, value=15, label="Depression Score")
                anxiety_score = gr.Slider(0, 50, value=10, label="Anxiety Score")
                social_support_score = gr.Slider(0, 100, value=70, label="Social Support Score")
                productivity_score = gr.Slider(0, 100, value=75, label="Productivity Score")
        
        predict_btn = gr.Button("Predict Mental Health Risk", variant="primary")
        
        with gr.Row():
            with gr.Column():
                prediction_output = gr.Textbox(label="Predicted Risk Level")
                probability_output = gr.JSON(label="Risk Probabilities")
                model_info_output = gr.Textbox(label="Model Information")
            
            with gr.Column():
                shap_output = gr.Image(label="SHAP Explanation")
        
        predict_btn.click(
            predict_mental_health_risk,
            inputs=[age, gender, employment_status, work_environment, mental_health_history,
                    seeks_treatment, stress_level, sleep_hours, physical_activity_days,
                    depression_score, anxiety_score, social_support_score, productivity_score],
            outputs=[prediction_output, probability_output, shap_output, model_info_output]
        )
    
    with gr.Tab("Model Comparison"):
        gr.Markdown("## Model Performance Comparison")
        comparison_output = gr.Markdown(show_model_comparison())
        
        # Show comparison plot if available
        try:
            comparison_plot = gr.Image("../Results/model_comparison.png", label="Model Comparison Plot")
        except:
            gr.Markdown("Model comparison plot not available")

if __name__ == "__main__":
    demo.launch()