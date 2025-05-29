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
        # Create summary dengan default values
        summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_data_size": 0,
            "current_data_size": 0,
            "drift_detected": False,
            "drifted_columns": [],
            "error": "No data file found"
        }
        
        os.makedirs("monitoring", exist_ok=True)
        with open("monitoring/evidently_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return
    
    # Initialize monitor
    monitor = EvidentlyMonitor()
    
    # Set reference data (gunakan 70% pertama sebagai reference)
    split_idx = int(len(current_data) * 0.7)
    reference_data = current_data.iloc[:split_idx]
    current_data_subset = current_data.iloc[split_idx:]
    
    print(f"📊 Data split: Reference={len(reference_data)}, Current={len(current_data_subset)}")
    
    monitor.set_reference_data(reference_data)
    
    try:
        # Run monitoring
        results = monitor.monitor_mental_health_data(current_data_subset)
        
        # Save results summary dengan explicit data sizes
        summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_data_size": int(len(reference_data)),  # ✅ Explicit conversion
            "current_data_size": int(len(current_data_subset)),  # ✅ Explicit conversion
            "drift_detected": bool(results["drift_results"]["drift_detected"]),
            "drifted_columns": results["drift_results"]["drifted_columns"],
            "total_features": int(len(current_data.columns)),
            "monitoring_status": "success"
        }
        
        print(f"✅ Summary created: Ref={summary['reference_data_size']}, Curr={summary['current_data_size']}")
        
    except Exception as e:
        print(f"❌ Evidently monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Create fallback summary dengan actual data sizes
        summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_data_size": int(len(reference_data)) if 'reference_data' in locals() else 0,
            "current_data_size": int(len(current_data_subset)) if 'current_data_subset' in locals() else 0,
            "drift_detected": False,
            "drifted_columns": [],
            "error": str(e),
            "monitoring_status": "failed"
        }
    
    # Ensure monitoring directory exists
    os.makedirs("monitoring", exist_ok=True)
    
    # Save summary dengan error handling
    try:
        with open("monitoring/evidently_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Summary saved: {summary}")
    except Exception as e:
        print(f"❌ Failed to save summary: {e}")
    
    return summary
