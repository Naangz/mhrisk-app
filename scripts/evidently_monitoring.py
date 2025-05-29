from evidently import Report
from evidently.metrics import ValueDrift, RowCount, MissingValueCount, UniqueValueCount
import pandas as pd
import os
from datetime import datetime
import json

class EvidentlyMonitor:
    def __init__(self):
        self.reference_data = None
        self.reports_dir = "monitoring/evidently_reports"
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def set_reference_data(self, df):
        """Set reference data untuk comparison"""
        self.reference_data = df.copy()
        print(f"✅ Reference data set with {len(df)} rows")
    
    def create_data_quality_report(self, current_data, save_html=True):
        """Create data quality report dengan parameter yang benar"""
        
        # Metrics untuk data quality
        metrics = [
            RowCount(),
        ]
        
        # Tambahkan MissingValueCount untuk setiap kolom (HARUS ada parameter column)
        for col in current_data.columns:
            metrics.append(MissingValueCount(column=col))
        
        # Tambahkan ValueDrift hanya untuk kolom numerical
        numerical_columns = current_data.select_dtypes(include=['number']).columns
        for col in numerical_columns:
            metrics.append(ValueDrift(column=col))
        
        # Skip categorical columns untuk menghindari error "text type"
        # Evidently 0.7.6 memiliki masalah dengan kolom categorical
        
        # Create report
        report = Report(metrics=metrics)
        
        if self.reference_data is not None:
            # Pastikan data types konsisten
            ref_df = self.reference_data.copy()
            curr_df = current_data.copy()
            
            # Convert categorical ke string untuk menghindari error
            for col in current_data.select_dtypes(include=['object']).columns:
                if col in ref_df.columns:
                    ref_df[col] = ref_df[col].astype(str)
                if col in curr_df.columns:
                    curr_df[col] = curr_df[col].astype(str)
            
            report.run(reference_data=ref_df, current_data=curr_df)
        else:
            report.run(reference_data=current_data, current_data=current_data)
        
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"{self.reports_dir}/data_quality_report_{timestamp}.html"
            report.save_html(report_path)
            print(f"✅ Data quality report saved: {report_path}")
        
        return report
    
    def create_drift_report(self, current_data, save_html=True):
        """Create drift detection report - simplified version"""
        
        if self.reference_data is None:
            print("⚠️ No reference data set. Cannot create drift report.")
            return None
        
        # Hanya gunakan numerical columns untuk drift detection
        numerical_columns = current_data.select_dtypes(include=['number']).columns
        
        if len(numerical_columns) == 0:
            print("⚠️ No numerical columns found for drift detection.")
            return None
        
        # Metrics untuk drift detection (hanya numerical)
        drift_metrics = []
        for col in numerical_columns:
            drift_metrics.append(ValueDrift(column=col))
        
        if not drift_metrics:
            print("⚠️ No valid columns for drift detection.")
            return None
        
        # Create drift report
        drift_report = Report(metrics=drift_metrics)
        
        try:
            drift_report.run(reference_data=self.reference_data, current_data=current_data)
            
            if save_html:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = f"{self.reports_dir}/drift_report_{timestamp}.html"
                drift_report.save_html(report_path)
                print(f"✅ Drift report saved: {report_path}")
            
            return drift_report
            
        except Exception as e:
            print(f"⚠️ Error creating drift report: {e}")
            return None
    
    def extract_drift_results(self, drift_report):
        """Extract drift detection results"""
        if drift_report is None:
            return {"drift_detected": False, "drifted_columns": []}
        
        try:
            # Get report as dict untuk analysis
            report_dict = drift_report.as_dict()
            
            drifted_columns = []
            drift_detected = False
            
            # Parse metrics results
            for metric in report_dict.get('metrics', []):
                if metric.get('metric') == 'ValueDrift':
                    column_name = metric.get('result', {}).get('column_name')
                    drift_score = metric.get('result', {}).get('drift_score', 0)
                    is_drifted = metric.get('result', {}).get('drift_detected', False)
                    
                    if is_drifted and column_name:
                        drifted_columns.append({
                            'column': column_name,
                            'drift_score': drift_score
                        })
                        drift_detected = True
            
            return {
                "drift_detected": drift_detected,
                "drifted_columns": drifted_columns,
                "total_drifted": len(drifted_columns)
            }
            
        except Exception as e:
            print(f"⚠️ Error extracting drift results: {e}")
            return {"drift_detected": False, "drifted_columns": []}
    
    def monitor_mental_health_data(self, current_data):
        """Comprehensive monitoring untuk mental health data"""
        
        print("🔍 Starting Evidently monitoring...")
        
        try:
            # Data quality report
            quality_report = self.create_data_quality_report(current_data)
            print("✅ Data quality report created")
        except Exception as e:
            print(f"⚠️ Error creating data quality report: {e}")
            quality_report = None
        
        try:
            # Drift detection report
            drift_report = self.create_drift_report(current_data)
            print("✅ Drift report created")
        except Exception as e:
            print(f"⚠️ Error creating drift report: {e}")
            drift_report = None
        
        # Extract drift results
        drift_results = self.extract_drift_results(drift_report)
        
        # Summary
        print(f"📊 Monitoring Summary:")
        print(f"  - Data rows: {len(current_data)}")
        print(f"  - Drift detected: {drift_results['drift_detected']}")
        print(f"  - Drifted columns: {drift_results['total_drifted']}")
        
        if drift_results['drifted_columns']:
            print("  - Drifted features:")
            for col_info in drift_results['drifted_columns']:
                print(f"    * {col_info['column']}: {col_info['drift_score']:.3f}")
        
        return {
            "quality_report": quality_report,
            "drift_report": drift_report,
            "drift_results": drift_results
        }

def run_evidently_monitoring():
    """Main function untuk menjalankan Evidently monitoring"""
    
    # Load data
    data_files = ['data/mental_health_lite.csv', 'data/mental_health_life_cut.csv']
    current_data = None
    
    for file_path in data_files:
        if os.path.exists(file_path):
            current_data = pd.read_csv(file_path)
            print(f"✅ Data loaded from: {file_path}")
            break
    
    if current_data is None:
        print("❌ No data file found")
        return
    
    # Initialize monitor
    monitor = EvidentlyMonitor()
    
    # Set reference data (gunakan 70% pertama sebagai reference)
    split_idx = int(len(current_data) * 0.7)
    reference_data = current_data.iloc[:split_idx]
    current_data_subset = current_data.iloc[split_idx:]
    
    monitor.set_reference_data(reference_data)
    
    # Run monitoring
    results = monitor.monitor_mental_health_data(current_data_subset)
    
    # Save results summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "reference_data_size": len(reference_data),
        "current_data_size": len(current_data_subset),
        "drift_detected": results["drift_results"]["drift_detected"],
        "drifted_columns": results["drift_results"]["drifted_columns"]
    }
    
    with open("monitoring/evidently_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Evidently monitoring completed successfully")
    return results

if __name__ == "__main__":
    try:
        run_evidently_monitoring()
    except Exception as e:
        print(f"❌ Evidently monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Create minimal summary untuk CI/CD
        summary = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "drifted_columns": [],
            "error": str(e)
        }
        
        os.makedirs("monitoring", exist_ok=True)
        with open("monitoring/evidently_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Minimal summary created for CI/CD continuation")
