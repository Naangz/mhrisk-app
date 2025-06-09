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
import warnings
warnings.filterwarnings('ignore')

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
            model_path = "model/mental_health_pipeline.skops"
            if os.path.exists(model_path):
                self.model = sio.load(model_path, trusted=True)
                print("✅ Model loaded successfully")
            else:
                print("⚠️ Model file not found, using fallback")
                return False
            
            # Load encoders
            encoders_path = "model/encoders.pkl"
            if os.path.exists(encoders_path):
                with open(encoders_path, "rb") as f:
                    self.encoders = pickle.load(f)
                print("✅ Encoders loaded successfully")
            
            # Load feature columns
            feature_path = "model/feature_columns.pkl"
            if os.path.exists(feature_path):
                with open(feature_path, "rb") as f:
                    self.feature_columns = pickle.load(f)
                print("✅ Feature columns loaded successfully")
            
            # Load model metadata
            metadata_path = "model/model_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
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
                'age': float(age),
                'stress_level': float(stress_level),
                'sleep_hours': float(sleep_hours),
                'physical_activity_days': float(physical_activity),
                'depression_score': float(depression_score),
                'anxiety_score': float(anxiety_score),
                'social_support_score': float(social_support),
                'productivity_score': float(productivity),
                'gender': gender,
                'employment_status': employment,
                'work_environment': work_env,
                'mental_health_history': mental_history,
                'seeks_treatment': seeks_treatment
            }
            
            df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            if self.encoders:
                categorical_cols = ['gender', 'employment_status', 'work_environment', 
                                  'mental_health_history', 'seeks_treatment']
                
                for col in categorical_cols:
                    if col in self.encoders and col in df.columns:
                        try:
                            # Handle unknown categories
                            encoder = self.encoders[col]
                            value = df[col].iloc[0]
                            
                            if value in encoder.classes_:
                                df[f'{col}_encoded'] = encoder.transform(df[col])
                            else:
                                # Use most frequent class for unknown values
                                df[f'{col}_encoded'] = 0
                                print(f"⚠️ Unknown value '{value}' for {col}, using default")
                        except Exception as e:
                            print(f"⚠️ Encoding error for {col}: {e}")
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
    
    def get_risk_interpretation(self, risk_level, probabilities):
        """Get detailed risk interpretation"""
        interpretations = {
            "Low": {
                "emoji": "🟢",
                "color": "#28a745",
                "description": "Your responses suggest a lower likelihood of mental health concerns.",
                "recommendations": [
                    "Continue maintaining your current positive lifestyle habits",
                    "Keep up with regular physical activity and good sleep hygiene",
                    "Stay connected with your support network",
                    "Practice stress management techniques preventively"
                ]
            },
            "Medium": {
                "emoji": "🟡", 
                "color": "#ffc107",
                "description": "Your responses indicate some areas that may benefit from attention.",
                "recommendations": [
                    "Consider speaking with a mental health professional for guidance",
                    "Focus on stress reduction techniques like meditation or yoga",
                    "Prioritize sleep hygiene and aim for 7-9 hours per night",
                    "Increase physical activity and social connections",
                    "Monitor your mental health symptoms regularly"
                ]
            },
            "High": {
                "emoji": "🔴",
                "color": "#dc3545", 
                "description": "Your responses suggest you may benefit from professional mental health support.",
                "recommendations": [
                    "🚨 Consider reaching out to a mental health professional soon",
                    "Contact your healthcare provider for a comprehensive evaluation",
                    "Reach out to trusted friends, family, or support groups",
                    "Practice immediate stress relief techniques",
                    "Consider crisis resources if you're having thoughts of self-harm"
                ]
            }
        }
        
        return interpretations.get(risk_level, interpretations["Medium"])
    
    def generate_shap_explanation(self, input_features):
        """Generate SHAP explanation visualization"""
        try:
            # Create feature importance visualization
            feature_names = input_features.columns.tolist()
            feature_values = input_features.iloc[0].values
            
            # Create horizontal bar plot
            plt.figure(figsize=(10, 8))
            y_pos = np.arange(len(feature_names))
            
            # Color bars based on values
            colors = ['#ff7f7f' if val > np.median(feature_values) else '#7fbfff' for val in feature_values]
            
            bars = plt.barh(y_pos, feature_values, color=colors, alpha=0.7, edgecolor='black')
            
            plt.yticks(y_pos, [name.replace('_encoded', '').replace('_', ' ').title() for name in feature_names])
            plt.xlabel('Feature Values (Normalized)')
            plt.title('Input Feature Analysis\n(Higher values may indicate increased risk factors)', 
                     fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, feature_values)):
                plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{val:.2f}', va='center', fontsize=10)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(plot_data).decode()
            
        except Exception as e:
            print(f"⚠️ SHAP explanation error: {e}")
            return None
    
    def predict_mental_health_risk(self, age, gender, employment, work_env, mental_history, 
                                  seeks_treatment, stress_level, sleep_hours, physical_activity, 
                                  depression_score, anxiety_score, social_support, productivity):
        """Make prediction and generate explanation"""
        
        if not self.model:
            return self.create_fallback_response()
        
        try:
            # Prepare input data
            input_features = self.prepare_input_data(
                age, gender, employment, work_env, mental_history, 
                seeks_treatment, stress_level, sleep_hours, physical_activity, 
                depression_score, anxiety_score, social_support, productivity
            )
            
            if input_features is None:
                return self.create_fallback_response()
            
            # Make prediction
            prediction_proba = self.model.predict_proba(input_features)[0]
            prediction = self.model.predict(input_features)[0]
            
            # Get risk levels
            if self.encoders and 'risk' in self.encoders:
                risk_levels = self.encoders['risk'].classes_
                risk_prediction = risk_levels[prediction]
            else:
                risk_levels = ['Low', 'Medium', 'High']
                risk_prediction = risk_levels[prediction] if prediction < len(risk_levels) else 'Medium'
            
            # Get risk interpretation
            risk_info = self.get_risk_interpretation(risk_prediction, prediction_proba)
            
            # Create main result
            result_html = f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin: 20px 0;">
                <h2 style="margin: 0; font-size: 2em;">{risk_info['emoji']} Mental Health Risk Assessment</h2>
                <h3 style="margin: 10px 0; color: {risk_info['color']}; background: white; padding: 10px; border-radius: 5px;">
                    Risk Level: {risk_prediction}
                </h3>
            </div>
            
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3>📊 Probability Scores:</h3>
                <div style="margin: 15px 0;">
            """
            
            for i, (level, prob) in enumerate(zip(risk_levels, prediction_proba)):
                emoji = "🟢" if level == "Low" else "🟡" if level == "Medium" else "🔴"
                width = prob * 100
                result_html += f"""
                    <div style="margin: 10px 0;">
                        <strong>{emoji} {level} Risk:</strong> {prob:.1%}
                        <div style="background: #e9ecef; border-radius: 10px; height: 25px; margin: 5px 0;">
                            <div style="background: {'#28a745' if level == 'Low' else '#ffc107' if level == 'Medium' else '#dc3545'}; 
                                        height: 100%; width: {width}%; border-radius: 10px; 
                                        display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                                {prob:.1%}
                            </div>
                        </div>
                    </div>
                """
            
            result_html += "</div></div>"
            
            # Add interpretation
            result_html += f"""
            <div style="background: #e8f4f8; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid {risk_info['color']};">
                <h3>🧠 Interpretation:</h3>
                <p style="font-size: 1.1em; line-height: 1.6;">{risk_info['description']}</p>
            </div>
            """
            
            # Add model information
            if self.model_metadata:
                result_html += f"""
                <div style="background: #f1f3f4; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <h4>🤖 Model Information:</h4>
                    <ul style="margin: 10px 0;">
                        <li><strong>Model:</strong> {self.model_metadata.get('best_model_name', 'Unknown')}</li>
                        <li><strong>Accuracy:</strong> {self.model_metadata.get('test_accuracy', 0):.1%}</li>
                        <li><strong>Last Updated:</strong> {self.model_metadata.get('timestamp', 'Unknown')[:10]}</li>
                    </ul>
                </div>
                """
            
            # Generate SHAP explanation
            explanation_plot = self.generate_shap_explanation(input_features)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(risk_info)
            
            return result_html, explanation_plot, recommendations
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self.create_fallback_response()
    
    def create_fallback_response(self):
        """Create fallback response when model fails"""
        result = """
        <div style="text-align: center; padding: 20px; border-radius: 10px; background: #ffc107; color: #212529;">
            <h2>⚠️ Assessment Temporarily Unavailable</h2>
            <p>We're experiencing technical difficulties. Please try again later or consult a healthcare professional.</p>
        </div>
        """
        return result, None, self.get_general_recommendations()
    
    def generate_recommendations(self, risk_info):
        """Generate personalized recommendations"""
        recommendations_html = f"""
        <div style="background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h2 style="color: {risk_info['color']}; margin-bottom: 20px;">💡 Personalized Recommendations</h2>
            
            <div style="background: {risk_info['color']}15; padding: 20px; border-radius: 8px; margin: 15px 0;">
                <h3 style="color: {risk_info['color']}; margin-top: 0;">Based on your {risk_info['emoji']} risk assessment:</h3>
                <ul style="line-height: 1.8; font-size: 1.1em;">
        """
        
        for rec in risk_info['recommendations']:
            recommendations_html += f"<li>{rec}</li>"
        
        recommendations_html += """
                </ul>
            </div>
            
            <div style="background: #e8f5e8; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #155724; margin-top: 0;">📞 Crisis Resources:</h3>
                <ul style="line-height: 1.8;">
                    <li><strong>National Suicide Prevention Lifeline:</strong> 988</li>
                    <li><strong>Crisis Text Line:</strong> Text HOME to 741741</li>
                    <li><strong>SAMHSA National Helpline:</strong> 1-800-662-4357</li>
                    <li><strong>Emergency Services:</strong> 911</li>
                </ul>
            </div>
            
            <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border: 1px solid #ffeaa7;">
                <p style="margin: 0; font-style: italic; text-align: center;">
                    <strong>⚠️ Important:</strong> This assessment is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment.
                </p>
            </div>
        </div>
        """
        
        return recommendations_html
    
    def get_general_recommendations(self):
        """Get general mental health recommendations"""
        return """
        <div style="background: white; padding: 25px; border-radius: 10px;">
            <h2>💡 General Mental Health Tips</h2>
            <ul style="line-height: 1.8;">
                <li>Maintain regular sleep schedule (7-9 hours per night)</li>
                <li>Exercise regularly (at least 30 minutes, 3-4 times per week)</li>
                <li>Practice stress management techniques</li>
                <li>Stay connected with friends and family</li>
                <li>Seek professional help when needed</li>
            </ul>
        </div>
        """

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
with gr.Blocks(
    title="Mental Health Risk Identifier", 
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    """
) as app:
    
    gr.HTML("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5em;">🧠 Mental Health Risk Identifier</h1>
        <p style="margin: 1rem 0 0 0; font-size: 1.2em; opacity: 0.9;">
            AI-powered mental health risk assessment with explainable predictions
        </p>
    </div>
    """)
    
    gr.Markdown("""
    ### 📝 Please fill out this assessment form to get your personalized mental health risk evaluation.
    
    **⚠️ Important Disclaimer:** This tool is for educational and informational purposes only. 
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 👤 Personal Information")
            
            age = gr.Slider(
                minimum=18, maximum=80, value=30, step=1,
                label="Age", info="Your current age"
            )
            
            gender = gr.Dropdown(
                choices=["Male", "Female", "Non-binary", "Prefer not to say"], 
                value="Male",
                label="Gender", info="Your gender identity"
            )
            
            employment = gr.Dropdown(
                choices=["Employed", "Unemployed", "Student", "Self-employed"], 
                value="Employed", 
                label="Employment Status"
            )
            
            work_env = gr.Dropdown(
                choices=["On-site", "Remote", "Hybrid"], 
                value="On-site",
                label="Work Environment", 
                info="Your primary work setting"
            )
            
            gr.Markdown("## 🏥 Mental Health History")
            
            mental_history = gr.Dropdown(
                choices=["Yes", "No"], 
                value="No",
                label="Previous Mental Health Issues", 
                info="Have you experienced mental health issues before?"
            )
            
            seeks_treatment = gr.Dropdown(
                choices=["Yes", "No"], 
                value="No",
                label="Currently Seeking Treatment",
                info="Are you currently receiving mental health treatment?"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Current Assessment")
            
            stress_level = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Stress Level", 
                info="Rate your current stress (1=Very Low, 10=Very High)"
            )
            
            sleep_hours = gr.Slider(
                minimum=3, maximum=12, value=7, step=0.5,
                label="Sleep Hours", 
                info="Average hours of sleep per night"
            )
            
            physical_activity = gr.Slider(
                minimum=0, maximum=7, value=3, step=1,
                label="Physical Activity Days", 
                info="Days per week with physical activity (30+ minutes)"
            )
            
            depression_score = gr.Slider(
                minimum=0, maximum=50, value=10, step=1,
                label="Depression Score", 
                info="Self-assessed depression level (0=None, 50=Severe)"
            )
            
            anxiety_score = gr.Slider(
                minimum=0, maximum=50, value=10, step=1,
                label="Anxiety Score", 
                info="Self-assessed anxiety level (0=None, 50=Severe)"
            )
            
            social_support = gr.Slider(
                minimum=0, maximum=100, value=70, step=5,
                label="Social Support Score", 
                info="Perceived social support (0=None, 100=Excellent)"
            )
            
            productivity = gr.Slider(
                minimum=0, maximum=100, value=70, step=5,
                label="Productivity Score", 
                info="Self-assessed productivity (0=Very Low, 100=Very High)"
            )
    
    with gr.Row():
        predict_btn = gr.Button(
            "🎯 Assess Mental Health Risk", 
            variant="primary", 
            size="lg",
            scale=2
        )
        clear_btn = gr.Button(
            "🔄 Clear Form", 
            variant="secondary",
            scale=1
        )
    
    with gr.Row():
        with gr.Column(scale=2):
            prediction_output = gr.HTML(label="Risk Assessment Results")
            recommendations_output = gr.HTML(label="Personalized Recommendations")
        
        with gr.Column(scale=1):
            explanation_output = gr.Image(
                label="Feature Analysis", 
                type="filepath",
                height=400
            )
    
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
    
    gr.HTML("""
    <div style="margin-top: 3rem; padding: 2rem; background: #f8f9fa; border-radius: 10px;">
        <h3>📋 About This Tool</h3>
        <p>This Mental Health Risk Identifier uses advanced machine learning algorithms to assess mental health risk factors based on your responses. The system includes:</p>
        <ul>
            <li>✅ <strong>Multi-model ensemble</strong> for reliable predictions</li>
            <li>✅ <strong>Feature analysis</strong> for model interpretability</li>
            <li>✅ <strong>Evidently monitoring</strong> for data quality assurance</li>
            <li>✅ <strong>Continuous training</strong> for model updates</li>
        </ul>
        
        <h4>🔒 Privacy & Security</h4>
        <p>Your data is processed locally and not stored. All computations happen in real-time without saving personal information.</p>
        
        <h4>⚠️ Important Disclaimers</h4>
        <ul>
            <li>This tool is for <strong>educational and informational purposes only</strong></li>
            <li>It should <strong>not replace professional medical advice</strong></li>
            <li>Always consult qualified healthcare professionals for medical concerns</li>
            <li>If you're experiencing crisis, contact emergency services immediately</li>
        </ul>
        
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: white; border-radius: 8px;">
            <p style="margin: 0; font-weight: bold; color: #666;">
                Built with ❤️ for mental health awareness • Powered by MLOps best practices
            </p>
        </div>
    </div>
    """)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,
        show_error=True
    )
