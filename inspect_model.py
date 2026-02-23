
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from core.minio_client import minio_client

def inspect_model(model_name):
    print(f"--- Inspecting Model: {model_name} ---")
    try:
        model = minio_client.get_model(model_name)
        if model is None:
            print(f"Model {model_name} not found in MinIO.")
            return

        print(f"Model type: {type(model)}")
        
        if hasattr(model, "feature_names"):
            print(f"Has feature_names: Yes ({len(model.feature_names)} features)")
            print(f"Features: {model.feature_names}")
        else:
            print("Has feature_names: No")

        if hasattr(model, "feature_names_in_"):
            print(f"Has feature_names_in_: Yes ({len(model.feature_names_in_)} features)")
            print(f"Features: {model.feature_names_in_}")
        else:
            print("Has feature_names_in_: No")

        if hasattr(model, "n_features_in_"):
            print(f"Has n_features_in_: Yes ({model.n_features_in_})")
        else:
            print("Has n_features_in_: No")
            
        # Check if it's an sklearn pipeline or similar
        if hasattr(model, "steps"):
            print("Model is a Pipeline")
            for name, step in model.steps:
                print(f"Step: {name}, Type: {type(step)}")
                if hasattr(step, "feature_names_in_"):
                    print(f"  Step features: {step.feature_names_in_}")

    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    inspect_model("xgboost_regression")
    inspect_model("logisticregression_classification")
