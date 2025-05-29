import json
import os

def add_evidently_summary_to_report():
    """Add Evidently summary to CML report"""
    
    with open('report.md', 'a') as f:
        f.write("\n## 🔍 Evidently Data Monitoring Results\n\n")
        
        if os.path.exists('monitoring/evidently_summary.json'):
            with open('monitoring/evidently_summary.json', 'r') as summary_file:
                summary = json.load(summary_file)
            
            f.write(f"**Monitoring Timestamp:** {summary.get('timestamp', 'Unknown')}\n")
            f.write(f"**Reference Data Size:** {summary.get('reference_data_size', 'Unknown')} rows\n")
            f.write(f"**Current Data Size:** {summary.get('current_data_size', 'Unknown')} rows\n")
            
            drift_status = "🚨 Yes" if summary.get('drift_detected', False) else "✅ No"
            f.write(f"**Data Drift Detected:** {drift_status}\n")
            
            if summary.get('drifted_columns'):
                f.write("\n**Drifted Columns:**\n")
                for col_info in summary['drifted_columns']:
                    f.write(f"- {col_info['column']}: drift score {col_info['drift_score']:.3f}\n")
            else:
                f.write("**Drifted Columns:** None detected\n")
        else:
            f.write("**Evidently Summary:** Not available\n")
        
        f.write("\n")
        
        # Add reports availability
        if os.path.exists('monitoring/evidently_reports') and os.listdir('monitoring/evidently_reports'):
            f.write("## 📋 Evidently Reports\n")
            f.write("Interactive Evidently reports have been generated and are available in the artifacts.\n\n")

if __name__ == "__main__":
    add_evidently_summary_to_report()
