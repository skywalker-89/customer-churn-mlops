import os
import sys
from minio import Minio
from io import BytesIO
import pandas as pd
import pickle
import json

# Add project root to sys.path to allow unpickling custom classes
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Try to import custom classes to ensure they are available for unpickling
try:
    from src.models_scratch.logistic_regression import (
        LogisticRegressionScratch as LogisticRegression,
    )
    from src.models_scratch.decision_tree import DecisionTreeScratch as DecisionTree
    from src.models_scratch.random_forest import RandomForestScratch as RandomForest
    from src.models_scratch.svm import SVMScratch as SVM
    from src.models_scratch.random_forest_pca import (
        RandomForestPCAScratch as RandomForestPCA,
    )
    from src.models_scratch.svm_pca import SVMPCAScratch as SVMPCA
    from src.models_scratch.kmeans_clustering import KMeansScratch as KMeansClustering
    from src.models_scratch.agglomerative_clustering import (
        AgglomerativeClusteringScratch as AgglomerativeClustering,
    )
    from src.models_scratch.perceptron import PerceptronScratch as Perceptron
    from src.models_scratch.mlp import MLPScratch as MLP
    from src.models_scratch.custom_model import CustomModelScratch as CustomModel
    from src.models_scratch.linear_regression import (
        LinearRegressionScratch as LinearRegression,
    )
    from src.models_scratch.multiple_regression import (
        MultipleRegressionScratch as MultipleRegression,
    )
    from src.models_scratch.polynomial_regression import (
        PolynomialRegressionScratch as PolynomialRegression,
    )
    from src.models_scratch.xgboost import XGBoostScratch as XGBoost
except ImportError as e:
    print(f"Warning: Could not import some custom model classes: {e}")


class MinIOClient:
    def __init__(self):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minio_admin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minio_password")
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False,
        )

    def get_model(self, model_name: str, bucket_name: str = "models"):
        """Load a model from MinIO."""
        try:
            # Try loading as is first (if it has extension), then with _latest.pkl, then with .pkl
            exceptions = []
            response = None

            candidates = [f"{model_name}_latest.pkl", f"{model_name}.pkl", model_name]

            for candidate in candidates:
                try:
                    response = self.client.get_object(bucket_name, candidate)
                    break
                except Exception as e:
                    exceptions.append(e)
                    continue

            if response is None:
                print(f"Could not load model {model_name}. Tried: {candidates}")
                return None

            model_bytes = response.read()
            model = pickle.loads(model_bytes)
            response.close()
            response.release_conn()
            return model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None

    def get_data(self, object_name: str, bucket_name: str = "processed-data"):
        """Load data from MinIO."""
        try:
            response = self.client.get_object(bucket_name, object_name)
            data = pd.read_parquet(BytesIO(response.read()))
            response.close()
            response.release_conn()
            return data
        except Exception as e:
            print(f"Error loading data {object_name}: {e}")
            # Fallback to local file
            try:
                local_path = f"/Users/jul/Desktop/uni/customer-churn-mlops/processed-data/{object_name}"
                if os.path.exists(local_path):
                    return pd.read_parquet(local_path)
            except:
                pass
            return None

    def list_models(self, bucket_name: str = "models"):
        """List all models in the bucket."""
        try:
            objects = self.client.list_objects(bucket_name)
            models = [
                obj.object_name for obj in objects if obj.object_name.endswith(".pkl")
            ]
            return models
        except Exception as e:
            print(f"Error listing models: {e}")
            return []


# Create global instance
minio_client = MinIOClient()
