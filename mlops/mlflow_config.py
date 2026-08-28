"""
MLflow Configuration
FlyRank Content Intelligence Platform
"""
import os

def setup_mlflow():
    """Setup MLflow tracking URI and experiment."""
    import mlflow
    
    # Store MLflow data in the project root / mlruns
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tracking_uri = f"sqlite:///{os.path.join(base_dir, 'mlflow.db')}"
    
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "FlyRank-ContentRefresh-Score"
    
    # Create experiment if it doesn't exist
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
    mlflow.set_experiment(experiment_name)
    return experiment_id, tracking_uri

if __name__ == "__main__":
    exp_id, uri = setup_mlflow()
    print(f"MLflow configured.\nTracking URI: {uri}\nExperiment ID: {exp_id}")
