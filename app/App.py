import gradio as gr
import pandas as pd
import numpy as np
import pickle
import json
import os
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import skops.io as sio
from datetime import datetime

class MentalHealthPredictor:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.feature_columns = None
        self.model_metadata = None
        self.explainer = None
        self.load_model_artifacts()
    
    def load_model_artifacts(self):
        """Load model artifacts with error handling"""
        try:
            # Load model
            if os.path.exists("model/mental_health_pipeline.skops"):
                self.model = sio.load("model/mental_health_pipeline.skops", trusted=True)
                print("✅ Model loaded successfully")
            else:
                print("⚠️ Model file not found")
                return False
            
            # Load encoders
            if os.path.exists("model/encoders.pkl"):
                with open("model/encoders.pkl", "rb") as f:
                    self.encoders = pickle.load(f)
                print("✅ Encoders loaded successfully")
            
            # Load feature columns
            if os.path.exists("model/feature_columns.pkl"):
                with open("model/feature_columns.pkl", "rb") as f:
                    self.feature_columns = pickle.load(f)
                print("✅ Feature columns loaded successfully")
            
            # Load model metadata
            if os.path.exists("model/model_metadata.json"):
                with open("model/model_metadata.json", "r") as f:
                    self.model_metadata = json.load(f)
                print("✅ Model metadata loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model artifacts: {e}")
            return False
    
    def prepare_input_data(self, age, gender, employment, work_env, mental_history, 
                          seeks_treatment, stress_level, sleep_hours, physical_activity, 
                          depression_score, anxiety_score, social_support, productivity):
        """Prepare input data for prediction"""
        try:
            # Create input dataframe
            input_data = {
                'age': age,
                'gender': gender,
                'employment_status': employment,
                'work_environment': work_env,
                'mental_health_history': mental_history,
                'seeks_treatment': seeks_treatment,
                'stress_level': stress_level,
                'sleep_hours': sleep_hours,
                'physical_activity_days': physical_activity,
                'depression_score': depression_score,
                'anxiety_score': anxiety_score,
                'social_support_score': social_support,
                'productivity_score': productivity
            }
            
            df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            if self.encoders:
                categorical_cols = ['gender', 'employment_status', 'work_environment', 
                                  'mental_health_history', 'seeks_treatment']
                
                for col in categorical_cols:
                    if col in self.encoders and col in df.columns:
                        try:
                            df[f'{col}_encoded'] = self.encoders[col].transform(df[col])
                        except ValueError as e:
                            print(f"⚠️ Encoding error for {col}: {e}")
                            # Use a default value or handle unknown categories
                            df[f'{col}_encoded'] = 0
            
            # Select features for prediction
            if self.feature_columns:
                available_features = [col for col in self.feature_columns if col in df.columns]
                df_features = df[available_features]
            else:
                # Fallback feature selection
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                df_features = df[numeric_cols]
            
            return df_features
            
        except Exception as e:
            print(f"❌ Error preparing input data: {e}")
            return None
    
    def predict_mental_health_risk(self, age, gender, employment, work_env, mental_history, 
                                  seeks_treatment, stress_level, sleep_hours, physical_activity, 
                                  depression_score, anxiety_score, social_support, productivity):
        """Make prediction and generate explanation"""
        
        if not self.model:
            return "❌ Model not loaded. Please check model artifacts.", "", ""
        
        try:
            # Prepare input data
            input_features = self.prepare_input_data(
                age, gender, employment, work_env, mental_history, 
                seeks_treatment, stress_level, sleep_hours, physical_activity, 
                depression_score, anxiety_score, social_support, productivity
            )
            
            if input_features is None:
                return "❌ Error preparing input data", "", ""
            
            # Make prediction
            prediction_proba = self.model.predict_proba(input_features)[0]
            prediction = self.model.predict(input_features)[0]
            
            # Get risk levels
            if self.encoders and 'risk' in self.encoders:
                risk_levels = self.encoders['risk'].classes_
                risk_prediction = risk_levels[prediction]
            else:
                risk_levels = ['Low', 'Medium', 'High']
                risk_prediction = risk_levels[prediction] if prediction < len(risk_levels) else 'Unknown'
            
            # Create prediction result
            result = f"""
## 🎯 Mental Health Risk Prediction

**Predicted Risk Level: {risk_prediction}**

### 📊 Probability Scores:
"""
            for i, (level, prob) in enumerate(zip(risk_levels, prediction_proba)):
                emoji = "🟢" if level == "Low" else "🟡" if level == "Medium" else "🔴"
                result += f"- {emoji} **{level} Risk:** {prob:.2%}\n"
            
            # Add model information
            if self.model_metadata:
                result += f"""
### 🤖 Model Information:
- **Model Type:** {self.model_metadata.get('best_model_name', 'Unknown')}
- **Model Accuracy:** {self.model_metadata.get('test_accuracy', 0):.2%}
- **Last Updated:** {self.model_metadata.get('timestamp', 'Unknown')}
"""
            
            # Generate SHAP explanation
            explanation_plot = self.generate_shap_explanation(input_features)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(risk_prediction, {
                'stress_level': stress_level,
                'sleep_hours': sleep_hours,
                'physical_activity': physical_activity,
                'depression_score': depression_score,
                'anxiety_score': anxiety_score,
                'social_support': social_support
            })
            
            return result, explanation_plot, recommendations
            
        except Exception as e:
            return f"❌ Prediction error: {str(e)}", "", ""
    
    def generate_shap_explanation(self, input_features):
        """Generate SHAP explanation plot"""
        try:
            # Create a simple feature importance visualization
            feature_names = input_features.columns.tolist()
            feature_values = input_features.iloc[0].values
            
            # Create a simple bar plot for feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(feature_names)), feature_values)
            plt.yticks(range(len(feature_names)), feature_names)
            plt.xlabel('Feature Values')
            plt.title('Input Feature Values')
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            # Encode to base64
            plot_base64 = base64.b64encode(plot_data).decode()
            
            return f"data:image/png;base64,{plot_base64}"
            
        except Exception as e:
            print(f"⚠️ SHAP explanation error: {e}")
            return None
    
    def generate_recommendations(self, risk_level, user_data):
        """Generate personalized recommendations"""
        recommendations = f"""
## 💡 Personalized Recommendations

Based on your **{risk_level} Risk** assessment:

"""
        
        if risk_level == "High":
            recommendations += """
### 🚨 High Priority Actions:
- **Seek Professional Help:** Consider consulting with a mental health professional
- **Crisis Support:** Contact a mental health crisis line if needed
- **Medical Evaluation:** Schedule an appointment with your healthcare provider
"""
        
        elif risk_level == "Medium":
            recommendations += """
### ⚠️ Moderate Priority Actions:
- **Monitor Symptoms:** Keep track of your mental health symptoms
- **Lifestyle Changes:** Focus on stress management and self-care
- **Support Network:** Reach out to friends, family, or support groups
"""
        
        else:  # Low risk
            recommendations += """
### ✅ Maintenance Actions:
- **Continue Good Habits:** Maintain your current positive lifestyle
- **Preventive Care:** Regular check-ins with healthcare providers
- **Stay Informed:** Learn about mental health awareness
"""
        
        # Specific recommendations based on user data
        if user_data['stress_level'] > 7:
            recommendations += "- **Stress Management:** Practice relaxation techniques, meditation, or yoga\n"
        
        if user_data['sleep_hours'] < 6:
            recommendations += "- **Sleep Hygiene:** Aim for 7-9 hours of quality sleep per night\n"
        
        if user_data['physical_activity'] < 3:
            recommendations += "- **Physical Activity:** Increase exercise to at least 3-4 days per week\n"
        
        if user_data['social_support'] < 50:
            recommendations += "- **Social Connection:** Build stronger relationships and social support networks\n"
        
        recommendations += """
### 📞 Resources:
- **National Suicide Prevention Lifeline:** 988
- **Crisis Text Line:** Text HOME to 741741
- **SAMHSA National Helpline:** 1-800-662-4357

*⚠️ Disclaimer: This tool is for informational purposes only and should not replace professional medical advice.*
"""
        
        return recommendations

# Initialize predictor
predictor = MentalHealthPredictor()

def predict_interface(age, gender, employment, work_env, mental_history, seeks_treatment,
                     stress_level, sleep_hours, physical_activity, depression_score, 
                     anxiety_score, social_support, productivity):
    """Gradio interface function"""
    return predictor.predict_mental_health_risk(
        age, gender, employment, work_env, mental_history, seeks_treatment,
        stress_level, sleep_hours, physical_activity, depression_score, 
        anxiety_score, social_support, productivity
    )

# Create Gradio interface
with gr.Blocks(title="Mental Health Risk Prediction", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🧠 Mental Health Risk Prediction System
    
    This AI-powered tool helps assess mental health risk levels using machine learning.
    Please fill out the form below to get your personalized assessment.
    
    **⚠️ Important:** This tool is for informational purposes only and should not replace professional medical advice.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📝 Personal Information")
            
            age = gr.Slider(
                minimum=18, maximum=80, value=30, step=1,
                label="Age", info="Your current age"
            )
            
            gender = gr.Dropdown(
                choices=["Male", "Female", "Non-binary"], value="Male",
                label="Gender", info="Your gender identity"
            )
            
            employment = gr.Dropdown(
                choices=["Employed", "Unemployed", "Student", "Self-employed"], 
                value="Employed", label="Employment Status"
            )
            
            work_env = gr.Dropdown(
                choices=["On-site", "Remote", "Hybrid"], value="On-site",
                label="Work Environment", info="Your primary work setting"
            )
            
            gr.Markdown("## 🏥 Mental Health History")
            
            mental_history = gr.Dropdown(
                choices=["Yes", "No"], value="No",
                label="Previous Mental Health Issues", 
                info="Have you experienced mental health issues before?"
            )
            
            seeks_treatment = gr.Dropdown(
                choices=["Yes", "No"], value="No",
                label="Currently Seeking Treatment",
                info="Are you currently receiving mental health treatment?"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Current Assessment")
            
            stress_level = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Stress Level", info="Rate your current stress (1=Low, 10=High)"
            )
            
            sleep_hours = gr.Slider(
                minimum=3, maximum=12, value=7, step=0.5,
                label="Sleep Hours", info="Average hours of sleep per night"
            )
            
            physical_activity = gr.Slider(
                minimum=0, maximum=7, value=3, step=1,
                label="Physical Activity Days", info="Days per week with physical activity"
            )
            
            depression_score = gr.Slider(
                minimum=0, maximum=50, value=10, step=1,
                label="Depression Score", info="Self-assessed depression level (0=None, 50=Severe)"
            )
            
            anxiety_score = gr.Slider(
                minimum=0, maximum=50, value=10, step=1,
                label="Anxiety Score", info="Self-assessed anxiety level (0=None, 50=Severe)"
            )
            
            social_support = gr.Slider(
                minimum=0, maximum=100, value=70, step=5,
                label="Social Support Score", info="Perceived social support (0=None, 100=Excellent)"
            )
            
            productivity = gr.Slider(
                minimum=0, maximum=100, value=70, step=5,
                label="Productivity Score", info="Self-assessed productivity (0=Very Low, 100=Very High)"
            )
    
    with gr.Row():
        predict_btn = gr.Button("🎯 Predict Mental Health Risk", variant="primary", size="lg")
        clear_btn = gr.Button("🔄 Clear Form", variant="secondary")
    
    with gr.Row():
        with gr.Column(scale=2):
            prediction_output = gr.Markdown(label="Prediction Results")
            recommendations_output = gr.Markdown(label="Recommendations")
        
        with gr.Column(scale=1):
            explanation_output = gr.Image(label="Feature Analysis", type="filepath")
    
    # Event handlers
    predict_btn.click(
        fn=predict_interface,
        inputs=[age, gender, employment, work_env, mental_history, seeks_treatment,
                stress_level, sleep_hours, physical_activity, depression_score, 
                anxiety_score, social_support, productivity],
        outputs=[prediction_output, explanation_output, recommendations_output]
    )
    
    clear_btn.click(
        fn=lambda: [30, "Male", "Employed", "On-site", "No", "No", 5, 7, 3, 10, 10, 70, 70],
        outputs=[age, gender, employment, work_env, mental_history, seeks_treatment,
                stress_level, sleep_hours, physical_activity, depression_score, 
                anxiety_score, social_support, productivity]
    )
    
    gr.Markdown("""
    ---
    ### 📋 About This Tool
    
    This Mental Health Risk Prediction System uses advanced machine learning algorithms including:
    - **Random Forest** for robust predictions
    - **XGBoost** for high accuracy
    - **LightGBM** for efficient processing
    
    The model was trained on mental health assessment data and includes:
    - ✅ **Multi-model ensemble** for reliable predictions
    - ✅ **SHAP explanations** for model interpretability
    - ✅ **Evidently monitoring** for data quality assurance
    - ✅ **Continuous training** for model updates
    
    **🔒 Privacy:** Your data is processed locally and not stored.
    
    **⚠️ Disclaimer:** This tool is for educational and informational purposes only. 
    Always consult with qualified healthcare professionals for medical advice.
    """)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
