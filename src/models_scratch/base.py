from abc import ABC, abstractmethod
import numpy as np
import pickle

class BaseModel(ABC):
    """
    The Contract: All models (Linear, Logistic, Trees) must follow this.
    """
    
    def __init__(self):
        self.weights = None
        self.bias = None
        self.history = {}  # To store cost/loss over time

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, lr: float = 0.01) -> None:
        """
        Train the model using Gradient Descent (NumPy only).
        :param X: Feature matrix (Rows = samples, Cols = features)
        :param y: Target vector
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        """
        pass

    def save(self, filepath: str):
        """Standard save method so Airflow can archive the model."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load a saved model from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def load_from_minio(cls, model_name: str):
        """Load model from MinIO if it exists, otherwise return None."""
        from minio import Minio
        import os
        from io import BytesIO
        
        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        client = Minio(
            endpoint,
            access_key=os.getenv("MINIO_ACCESS_KEY", "minio_admin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minio_password"),
            secure=False
        )
        
        try:
            response = client.get_object("models", f"{model_name}_latest.pkl")
            model = pickle.loads(response.read())
            response.close()
            response.release_conn()
            print(f"   ♻️  Loaded existing model from MinIO: {model_name}")
            return model
        except Exception as e:
            print(f"   🆕 No existing model found in MinIO: {model_name} (will train from scratch)")
            return None

    def save_to_minio(self, model_name: str):
        """Save model to MinIO models bucket."""
        from minio import Minio
        import os
        from io import BytesIO
        
        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        client = Minio(
            endpoint,
            access_key=os.getenv("MINIO_ACCESS_KEY", "minio_admin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minio_password"),
            secure=False
        )
        
        # Create bucket if it doesn't exist
        if not client.bucket_exists("models"):
            client.make_bucket("models")
            print(f"   Created MinIO bucket: models")
        
        # Serialize and upload
        model_bytes = pickle.dumps(self)
        client.put_object(
            "models",
            f"{model_name}_latest.pkl",
            BytesIO(model_bytes),
            len(model_bytes)
        )
        print(f"   💾 Saved to MinIO: models/{model_name}_latest.pkl ({len(model_bytes):,} bytes)")

    def _add_bias(self, X):
        """Helper to add a column of 1s for the bias term (x0)."""
        return np.c_[np.ones(X.shape[0]), X]