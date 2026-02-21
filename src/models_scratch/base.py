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

        # Optional (used by benchmark + preprocessing)
        self.feature_names = None
        self.nan_mean = None
        self.mu = None
        self.std = None

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        lr: float = 0.01,
        feature_names=None,
        warm_start: bool = False,
        **kwargs
    ) -> None:
        """
        Train the model (NumPy only).
        benchmark may pass: feature_names, warm_start
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
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
        except Exception:
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

        if not client.bucket_exists("models"):
            client.make_bucket("models")
            print(f"   Created MinIO bucket: models")

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

    # =========================
    # Extra helpers (so NO utils.py needed)
    # =========================
    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def _impute_nan_fit(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        self.nan_mean = np.nanmean(X, axis=0)
        X2 = X.copy()
        inds = np.where(np.isnan(X2))
        if inds[0].size > 0:
            X2[inds] = np.take(self.nan_mean, inds[1])
        return X2

    def _impute_nan_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.nan_mean is None:
            return np.nan_to_num(X, nan=0.0)
        X2 = X.copy()
        inds = np.where(np.isnan(X2))
        if inds[0].size > 0:
            X2[inds] = np.take(self.nan_mean, inds[1])
        return X2

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        self.mu = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0
        return (X - self.mu) / self.std

    def _standardize_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.mu is None or self.std is None:
            return X
        return (X - self.mu) / self.std

    def _prep_fit(self, X: np.ndarray, standardize: bool = True) -> np.ndarray:
        X = self._impute_nan_fit(X)
        if standardize:
            X = self._standardize_fit(X)
        return X

    def _prep_transform(self, X: np.ndarray, standardize: bool = True) -> np.ndarray:
        X = self._impute_nan_transform(X)
        if standardize:
            X = self._standardize_transform(X)
        return X
