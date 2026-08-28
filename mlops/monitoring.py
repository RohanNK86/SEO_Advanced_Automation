"""
Evidently Monitoring Script
FlyRank Content Intelligence Platform
"""
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("Evidently is not installed. Please install it using: pip install evidently")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
REFERENCE_DATA_PATH = os.path.join(DATA_DIR, "features.csv") 
# In a real scenario, current_data would be live incoming data
CURRENT_DATA_PATH = os.path.join(DATA_DIR, "features.csv") 

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "mlops", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

def generate_drift_report(reference_path: str = REFERENCE_DATA_PATH, current_path: str = CURRENT_DATA_PATH):
    if not EVIDENTLY_AVAILABLE:
        return None
        
    if not os.path.exists(reference_path) or not os.path.exists(current_path):
        print("Data files not found. Please run feature pipeline first.")
        return None
        
    print("Loading data for drift monitoring...")
    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_path)
    
    # For demo purposes, we compare the data against a sample of itself if current is same as ref
    if reference_path == current_path:
        ref_df = ref_df.sample(frac=0.5, random_state=42)
        cur_df = cur_df.drop(ref_df.index)
        
    print("Generating Data Drift Report...")
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    data_drift_report.run(reference_data=ref_df, current_data=cur_df)
    
    report_path = os.path.join(REPORT_DIR, "data_drift_report.html")
    data_drift_report.save_html(report_path)
    print(f"Drift report saved to: {report_path}")
    return report_path

if __name__ == "__main__":
    generate_drift_report()
