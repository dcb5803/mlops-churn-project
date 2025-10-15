import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from data import load_and_split_data
import os

# Set a remote or local tracking URI (e.g., set to a local 'mlruns' directory for simplicity)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Churn_Prediction_Project")

def train_and_log_model(n_estimators, max_depth):
    """Trains, evaluates, and logs a RandomForest model with MLflow."""
    X_train, X_test, y_train, y_test = load_and_split_data()

    with mlflow.start_run() as run:
        # 1. Train Model
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        # 2. Log Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        print(f"Logged Metrics: Accuracy={accuracy}, Precision={precision}")

        # 3. Log Parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # 4. Log Model and Register
        model_name = "Churn_RF_Model"
        
        # Log the model artifact and register it in the Model Registry
        mlflow.sklearn.log_model(
            sk_model=rf, 
            artifact_path="model", 
            registered_model_name=model_name
        )
        
        # NOTE: In a real project, you would transition the model to 'Staging'
        # based on metric checks, but for this basic CI, we're just registering.

if __name__ == "__main__":
    # Example runs for comparison
    train_and_log_model(n_estimators=100, max_depth=5)
    train_and_log_model(n_estimators=200, max_depth=10) # Best model will be chosen/promoted
