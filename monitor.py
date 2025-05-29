import pandas as pd
import os
from datetime import datetime
from scripts.evidently_monitoring import EvidentlyMonitor

def monitor_data_quality(df, profile_name="production_data"):
    """Monitor data quality dengan Evidently saja"""
    
    try:
        monitor = EvidentlyMonitor()
        
        # Load reference data if exists
        reference_path = "monitoring/reference_data.csv"
        if os.path.exists(reference_path):
            reference_data = pd.read_csv(reference_path)
            monitor.set_reference_data(reference_data)
        
        # Run monitoring
        results = monitor.monitor_mental_health_data(df)
        
        print(f"✅ Evidently monitoring completed")
        print(f"📊 Drift detected: {results['drift_results']['drift_detected']}")
        
        return results
        
    except Exception as e:
        print(f"⚠️ Evidently monitoring error: {e}")
        return None

def detect_data_drift(reference_profile_path, current_df):
    """Detect data drift menggunakan Evidently"""
    
    try:
        monitor = EvidentlyMonitor()
        
        # Load reference data
        if os.path.exists(reference_profile_path):
            reference_data = pd.read_csv(reference_profile_path)
            monitor.set_reference_data(reference_data)
            
            # Run drift detection
            results = monitor.monitor_mental_health_data(current_df)
            
            drift_detected = results['drift_results']['drift_detected']
            drift_report = [f"Drifted columns: {len(results['drift_results']['drifted_columns'])}"]
            
            return drift_detected, drift_report
        else:
            print("⚠️ Reference data not found")
            return False, ["Reference data not available"]
            
    except Exception as e:
        print(f"⚠️ Drift detection error: {e}")
        return False, [f"Drift detection failed: {e}"]

if __name__ == "__main__":
    try:
        df = pd.read_csv("data/mental_health_lite.csv")
        results = monitor_data_quality(df)
        print("✅ Data quality monitoring completed")
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
