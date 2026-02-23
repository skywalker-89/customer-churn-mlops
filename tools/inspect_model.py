import sys
import os
import pickle

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock custom model imports if needed, but minio_client does it.
# We need to make sure we can import them too if pickle needs them.
# minio_client.py imports them, so importing minio_client should be enough?
# No, unpickling requires the classes to be available in sys.modules.
# minio_client.py does `from src.models_scratch...` which registers them.
# So importing minio_client is crucial.

try:
    from backend.core.minio_client import MinIOClient
except ImportError:
    # If backend module not found, try adding src
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from backend.core.minio_client import MinIOClient

def inspect_model(model_name):
    client = MinIOClient()
    print(f"\n--- Inspecting {model_name} ---")
    
    # Try to list models first to see what's available
    try:
        models = client.list_models()
        # print(f"Available models in MinIO: {models}")
    except Exception as e:
        print(f"Error listing models: {e}")
        models = []
    
    # Clean up model name for searching
    clean_name = model_name.replace(".pkl", "").replace("_latest", "")
    
    # Find best match in models list
    target_filename = None
    for m in models:
        if clean_name in m:
            target_filename = m
            break
    
    if target_filename:
        print(f"Found candidate file: {target_filename}")
        load_name = target_filename # Pass the full filename, get_model handles it if it's in candidates?
        # validation: get_model tries [name_latest.pkl, name.pkl, name]
        # if target_filename is "foo.pkl", passing "foo.pkl" works as 3rd candidate.
    else:
        print(f"Model {model_name} not found in list. Trying direct load of {clean_name}...")
        load_name = clean_name

    try:
        model = client.get_model(load_name)
        if model:
            print(f"Model loaded successfully: {type(model)}")
            
            features = None
            if hasattr(model, "feature_names_in_"):
                features = model.feature_names_in_
                print(f"Feature names found ({len(features)}):")
                print(features)
            elif hasattr(model, "n_features_in_"):
                print(f"Number of features: {model.n_features_in_}")
                print("feature_names_in_ not found.")
            else:
                print("No standard feature info found on model object.")
                
            # Check for coefficient logic if linear model
            if hasattr(model, "coef_"):
                print(f"Model has coefficients (shape: {model.coef_.shape})")
                if features is not None and len(features) == model.coef_.shape[-1]:
                     # specific check for engagement score
                     for i, f in enumerate(features):
                         if "engagement" in f or "app_usage" in f or "social" in f:
                             print(f"Feature '{f}' coefficient: {model.coef_[0][i]}")
            
            # Check for feature importances
            if hasattr(model, "feature_importances_"):
                print("Model has feature importances.")
                if features is not None:
                     # specific check for engagement score
                     indices = [i for i, f in enumerate(features) if "engagement" in f or "app_usage" in f or "social" in f]
                     for i in indices:
                         print(f"Feature '{features[i]}' importance: {model.feature_importances_[i]}")

    except Exception as e:
        print(f"Error loading/inspecting model: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            inspect_model(name)
    else:
        inspect_model("logisticregression_classification")
        inspect_model("randomforest_classification")
        inspect_model("xgboost_regression")
