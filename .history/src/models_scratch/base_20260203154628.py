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

    def _add_bias(self, X):
        """Helper to add a column of 1s for the bias term (x0)."""
        return np.c_[np.ones(X.shape[0]), X]