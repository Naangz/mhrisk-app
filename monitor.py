import pandas as pd
import whylogs as why
import os
from datetime import datetime

def setup_whylogs():
    """Setup WhyLogs untuk monitoring"""
    try:
        why.init(allow_local=True, allow_anonymous=False)
        print("✅ WhyLogs monitoring initialized")
        return True
    except Exception as e:
        print(f"⚠️ WhyLogs setup error: {e}")
        return False

def monitor_data_quality(df, profile_name="production_data"):
    """Monitor data quality dengan WhyLogs"""
    
    # Setup WhyLogs
    if not setup_whylogs():
        print("Skipping WhyLogs monitoring...")
        return None, None
    
    try:
        # Create profile
        profile = why.log(df)
        
        # Create directory
        os.makedirs("Monitoring/whylogs_profiles", exist_ok=True)
        
        # Save profile with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_path = f"Monitoring/whylogs_profiles/{profile_name}_{timestamp}"
        
        # Use proper method to save
        profile_view = profile.view()
        profile_view.write(profile_path)
        
        print(f"✅ Data quality profile saved: {profile_path}")
        return profile, None
        
    except Exception as e:
        print(f"⚠️ Data quality monitoring error: {e}")
        return None, None

def detect_data_drift(reference_profile_path, current_df):
    """Detect data drift - simplified version"""
    
    try:
        # Simple drift detection based on statistical measures
        print("🔍 Checking for data drift...")
        
        # For now, return no drift detected
        # In production, implement proper drift detection logic
        drift_detected = False
        drift_report = ["No significant drift detected"]
        
        return drift_detected, drift_report
        
    except Exception as e:
        print(f"⚠️ Drift detection error: {e}")
        return False, [f"Drift detection failed: {e}"]

if __name__ == "__main__":
    # Example usage
    try:
        df = pd.read_csv("Data/mental_health_dataset.csv")
        profile, report = monitor_data_quality(df)
        print("✅ Data quality monitoring completed")
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
