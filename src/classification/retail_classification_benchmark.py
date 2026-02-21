"""
Retail Classification Benchmark
Compares MULTIPLE from-scratch classification models vs 1 sklearn model
Target: churned (predicting customer churn: 0 = retained, 1 = churned)

INSTRUCTIONS FOR ML ENGINEER:
================================
1. Build each model from scratch in src/models_scratch/ folder
2. Each model MUST inherit from BaseModel (see src/models_scratch/base.py)
3. Import your models in the section marked: 🔧 IMPORT YOUR MODELS HERE
4. Configure hyperparameters in MODEL_CONFIG section
5. Add model instances to the models list in train_scratch_models()
6. The benchmark will automatically:
   - Train all models
   - Save to MinIO
   - Load from MinIO for warm starts
   - Compare with sklearn baseline
   - Log metrics to MLflow

REQUIRED MODELS (Build from Scratch):
======================================
✓ Logistic Regression
✓ Decision Tree
✓ Random Forest (Ensemble Learning)
✓ Support Vector Machine (SVM)
✓ Dimensionality Reduction + Random Forest
✓ Dimensionality Reduction + SVM
✓ K-Means Clustering (Unsupervised)
✓ Agglomerative Clustering (Unsupervised)
✓ Perceptron / Single-Layer Perceptron (SLP)
✓ Multi-Layer Perceptron (MLP)
✓ [YOUR CHOICE] - One custom model from outside classroom
train iterate ให้เท่ากัน

SKLEARN COMPARISON:
===================
Pick ONE model from sklearn to benchmark against (e.g., RandomForestClassifier, XGBoost)
"""

import os
import numpy as np
import pandas as pd
import mlflow
from io import BytesIO
from minio import Minio


# ============================================================
# 🔧 IMPORT YOUR FROM-SCRATCH MODELS HERE
# ============================================================
from src.models_scratch.logistic_regression import LogisticRegressionScratch
from src.models_scratch.decision_tree import DecisionTreeScratch
from src.models_scratch.random_forest import RandomForestScratch
from src.models_scratch.svm import SVMScratch
from src.models_scratch.random_forest_pca import RandomForestPCAScratch
from src.models_scratch.svm_pca import SVMPCAScratch
from src.models_scratch.kmeans_clustering import KMeansScratch
from src.models_scratch.agglomerative_clustering import AgglomerativeClusteringScratch
from src.models_scratch.perceptron import PerceptronScratch
from src.models_scratch.mlp import MLPScratch
from src.models_scratch.custom_model import CustomModelScratch


# ============================================================
# 📊 IMPORT SKLEARN FOR COMPARISON
# ============================================================
# TODO: Pick ONE sklearn model to compare against your from-scratch implementations
# Example:
# from sklearn.ensemble import RandomForestClassifier
# OR
# from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split


# ============================================================
# ⚙️ Model Hyperparameters Configuration
# ============================================================
# 100 epochs
MODEL_CONFIG = {
    "LogisticRegression": {"epochs": 0, "lr": 0.01, "warm_start_epochs": 20},
    "DecisionTree": {
        "epochs": 0,
        "max_depth": 10,
        "min_samples_split": 20,
        "warm_start_epochs": 0,
    },
    "RandomForest": {
        "epochs": 0,
        "n_estimators": 50,
        "max_depth": 10,
        "min_samples_split": 20,
        "warm_start_epochs": 0,
    },
    "SVM": {
        "epochs": 0,
        "lr": 0.001,
        "C": 1.0,
        "kernel": "linear",
        "warm_start_epochs": 20,
    },
    "RandomForestPCA": {
        "epochs": 0,
        "n_components": 10,
        "n_estimators": 50,
        "max_depth": 10,
        "warm_start_epochs": 0,
    },
    "SVMPCA": {
        "n_components": 10,
        "epochs": 0,
        "lr": 0.001,
        "C": 1.0,
        "kernel": "linear",
        "warm_start_epochs": 20,
    },
    "KMeans": {
        "epochs": 0,
        "n_clusters": 2,
        "max_iters": 100,
        "warm_start_epochs": 0,
    },
    "AgglomerativeClustering": {
        "epochs": 0,
        "n_clusters": 2,
        "linkage": "ward",
        "warm_start_epochs": 0,
    },
    "Perceptron": {"epochs": 0, "lr": 0.01, "warm_start_epochs": 20},
    "MLP": {
        "epochs": 0,
        "lr": 0.001,
        "hidden_layers": [64, 32],
        "activation": "relu",
        "warm_start_epochs": 20,
    },
    "CustomModel": {"epochs": 0, "lr": 0.01, "warm_start_epochs": 20},
}


# ============================================================
# Utility Functions (Metrics + Train/Test Split)
# ============================================================
def train_test_split_np(X, y, test_size=0.2, random_state=42):
    """NumPy-based train/test split"""
    rng = np.random.RandomState(random_state)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(np.floor(n * test_size))
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


def accuracy(y_true, y_pred):
    """Calculate accuracy"""
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    """Calculate precision (TP / (TP + FP))"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    """Calculate recall (TP / (TP + FN))"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    """Calculate F1 score (harmonic mean of precision and recall)"""
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix"""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


print("=" * 60)
print("RETAIL CLASSIFICATION BENCHMARK - Customer Churn Prediction")
print("=" * 60)


# ============================================================
# Data Loader
# ============================================================
def load_retail_data(sample_size=None):
    """
    Load retail training data from MinIO.

    Args:
        sample_size: Number of rows to sample (default None = use full dataset)
    """
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minio_admin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minio_password")
    bucket = os.getenv("SOURCE_BUCKET", "processed-data")

    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    print(f"\n📥 Loading from s3://{bucket}/training_data.parquet...")
    response = client.get_object(bucket, "training_data.parquet")
    df = pd.read_parquet(BytesIO(response.read()))
    response.close()
    response.release_conn()

    print(f"   Loaded {len(df):,} rows with {len(df.columns)} columns")

    if sample_size and len(df) > sample_size:
        print(f"   📊 Sampling {sample_size:,} rows to reduce memory usage...")
        df = df.sample(n=sample_size, random_state=42)
        print(f"   Using {len(df):,} rows for training")

    drop_cols = ["total_sales", "churned", "clv_per_year"]
    X_df = df.drop(columns=drop_cols, errors="ignore")
    X_df = X_df.select_dtypes(include=[np.number])

    y = df["churned"].values.astype(np.int32)
    feature_names = list(X_df.columns)

    print(f"\n   📊 Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"      Class {cls}: {count:,} ({count/len(y)*100:.2f}%)")

    print(f"\n   Features: {len(feature_names)}")
    print(f"   Target (churned): {len(y):,} samples")

    return X_df.values.astype(np.float64), y, feature_names


# ============================================================
# 🔨 Train From-Scratch Models
# ============================================================
def train_scratch_models(X_train, y_train, X_test, y_test, feature_names):
    """
    Train ALL from-scratch models with incremental learning.

    Each tuple: (name, model_instance, epochs, lr)
    """
    print("\n" + "=" * 60)
    print("🔨 TRAINING FROM-SCRATCH MODELS (Incremental)")
    print("=" * 60)

    models = [
        (
            "LogisticRegression",
            LogisticRegressionScratch(),
            MODEL_CONFIG["LogisticRegression"]["epochs"],
            MODEL_CONFIG["LogisticRegression"]["lr"],
        ),
        (
            "DecisionTree",
            DecisionTreeScratch(
                max_depth=MODEL_CONFIG["DecisionTree"]["max_depth"],
                min_samples_split=MODEL_CONFIG["DecisionTree"]["min_samples_split"],
            ),
            MODEL_CONFIG["DecisionTree"]["epochs"],
            0.0,
        ),
        (
            "RandomForest",
            RandomForestScratch(
                n_estimators=MODEL_CONFIG["RandomForest"]["n_estimators"],
                max_depth=MODEL_CONFIG["RandomForest"]["max_depth"],
                min_samples_split=MODEL_CONFIG["RandomForest"]["min_samples_split"],
            ),
            MODEL_CONFIG["RandomForest"]["epochs"],
            0.0,
        ),
        (
            "SVM",
            SVMScratch(
                C=MODEL_CONFIG["SVM"]["C"], kernel=MODEL_CONFIG["SVM"]["kernel"]
            ),
            MODEL_CONFIG["SVM"]["epochs"],
            MODEL_CONFIG["SVM"]["lr"],
        ),
        (
            "RandomForestPCA",
            RandomForestPCAScratch(
                n_components=MODEL_CONFIG["RandomForestPCA"]["n_components"],
                n_estimators=MODEL_CONFIG["RandomForestPCA"]["n_estimators"],
                max_depth=MODEL_CONFIG["RandomForestPCA"]["max_depth"],
            ),
            MODEL_CONFIG["RandomForestPCA"]["epochs"],
            0.0,
        ),
        (
            "SVMPCA",
            SVMPCAScratch(
                n_components=MODEL_CONFIG["SVMPCA"]["n_components"],
                C=MODEL_CONFIG["SVMPCA"]["C"],
                kernel=MODEL_CONFIG["SVMPCA"]["kernel"],
            ),
            MODEL_CONFIG["SVMPCA"]["epochs"],
            MODEL_CONFIG["SVMPCA"]["lr"],
        ),
        (
            "KMeans",
            KMeansScratch(
                n_clusters=MODEL_CONFIG["KMeans"]["n_clusters"],
                max_iters=MODEL_CONFIG["KMeans"]["max_iters"],
            ),
            MODEL_CONFIG["KMeans"]["epochs"],
            0.0,
        ),
        (
            "AgglomerativeClustering",
            AgglomerativeClusteringScratch(
                n_clusters=MODEL_CONFIG["AgglomerativeClustering"]["n_clusters"],
                linkage=MODEL_CONFIG["AgglomerativeClustering"]["linkage"],
                sample_size=300,
            ),
            MODEL_CONFIG["AgglomerativeClustering"]["epochs"],
            0.0,
        ),
        (
            "Perceptron",
            PerceptronScratch(),
            MODEL_CONFIG["Perceptron"]["epochs"],
            MODEL_CONFIG["Perceptron"]["lr"],
        ),
        (
            "MLP",
            MLPScratch(
                hidden_layers=MODEL_CONFIG["MLP"]["hidden_layers"],
                activation=MODEL_CONFIG["MLP"]["activation"],
            ),
            MODEL_CONFIG["MLP"]["epochs"],
            MODEL_CONFIG["MLP"]["lr"],
        ),
        (
            "CustomModel",
            CustomModelScratch(),
            MODEL_CONFIG["CustomModel"]["epochs"],
            MODEL_CONFIG["CustomModel"]["lr"],
        ),
    ]

    results = []

    for name, model, epochs, lr in models:
        print(f"\n{'─' * 60}")
        print(f"📊 {name} Classification")
        print(f"{'─' * 60}")

        model_name = f"{name.lower()}_classification"

        # Warm start load (BUT keep epochs the same)
        try:
            loaded_model = model.load_from_minio(model_name)
            if loaded_model:
                model = loaded_model
                print(f"   ✅ Loaded existing model from MinIO: {model_name}")
                warm_start = True
            else:
                print(
                    f"   🆕 No existing model found: {model_name} (training from scratch)"
                )
                warm_start = False
        except Exception as e:
            print(f"   🆕 Error loading model: {e} (training from scratch)")
            warm_start = False

        # ✅ TRAIN ITERATE ให้เท่ากัน: always use epochs from config
        actual_epochs = int(epochs)

        model.fit(
            X_train,
            y_train,
            epochs=actual_epochs,
            lr=lr,
            feature_names=feature_names,
            warm_start=warm_start,
        )

        model.save_to_minio(model_name)

        y_pred = model.predict(X_test)
        y_pred = (
            (y_pred > 0.5).astype(np.int32) if y_pred.dtype == np.float64 else y_pred
        )

        _accuracy = accuracy(y_test, y_pred)
        _precision = precision(y_test, y_pred)
        _recall = recall(y_test, y_pred)
        _f1 = f1_score(y_test, y_pred)
        _cm = confusion_matrix(y_test, y_pred)

        print(f"\n   📈 Results:")
        print(f"      Accuracy:  {_accuracy:.4f}")
        print(f"      Precision: {_precision:.4f}")
        print(f"      Recall:    {_recall:.4f}")
        print(f"      F1 Score:  {_f1:.4f}")
        print(
            f"      Confusion Matrix: TN={_cm['TN']}, FP={_cm['FP']}, FN={_cm['FN']}, TP={_cm['TP']}"
        )

        results.append(
            {
                "model": f"{name} (scratch)",
                "accuracy": _accuracy,
                "precision": _precision,
                "recall": _recall,
                "f1_score": _f1,
                "confusion_matrix": _cm,
            }
        )

    return results


# ============================================================
# 📊 Train Sklearn Model (for comparison)
# ============================================================
def train_sklearn_model(X_train, y_train, X_test, y_test):
    """
    Train ONE sklearn model as a benchmark.
    """
    print("\n" + "=" * 60)
    print("🚀 TRAINING SKLEARN MODEL (Comparison)")
    print("=" * 60)

    # ============================================================
    # 🔧 CONFIGURE YOUR SKLEARN MODEL HERE
    # ============================================================
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=3
    )
    model_name = "random_forest_sklearn"

    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minio_admin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minio_password")
    bucket_name = "models"
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    import pickle

    try:
        response = client.get_object(bucket_name, f"{model_name}_latest.pkl")
        model = pickle.loads(response.read())
        print(f"   ✅ Loaded existing Sklearn model from MinIO: {model_name}")
        response.close()
        response.release_conn()
    except Exception:
        print(f"   🆕 No existing model found: {model_name} (training from scratch)")
        model.fit(X_train, y_train)

        try:
            model_bytes = pickle.dumps(model)
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)

            client.put_object(
                bucket_name,
                f"{model_name}_latest.pkl",
                BytesIO(model_bytes),
                len(model_bytes),
            )
            print(f"   💾 Saved to MinIO: {bucket_name}/{model_name}_latest.pkl")
        except Exception as save_err:
            print(f"   ⚠️ Failed to save model to MinIO: {save_err}")

    y_pred = model.predict(X_test)

    _accuracy = accuracy(y_test, y_pred)
    _precision = precision(y_test, y_pred)
    _recall = recall(y_test, y_pred)
    _f1 = f1_score(y_test, y_pred)
    _cm = confusion_matrix(y_test, y_pred)

    print(f"\n   📈 Results:")
    print(f"      Accuracy:  {_accuracy:.4f}")
    print(f"      Precision: {_precision:.4f}")
    print(f"      Recall:    {_recall:.4f}")
    print(f"      F1 Score:  {_f1:.4f}")
    print(
        f"      Confusion Matrix: TN={_cm['TN']}, FP={_cm['FP']}, FN={_cm['FN']}, TP={_cm['TP']}"
    )

    return [
        {
            "model": f"{model_name.replace('_', ' ').title()} (sklearn)",
            "accuracy": _accuracy,
            "precision": _precision,
            "recall": _recall,
            "f1_score": _f1,
            "confusion_matrix": _cm,
        }
    ]


# ============================================================
# Main
# ============================================================
def main():
    mlflow.set_experiment("retail_classification_benchmark")

    with mlflow.start_run(run_name="retail_customer_churn_prediction"):
        X, y, feature_names = load_retail_data()

        X_train, X_test, y_train, y_test = train_test_split_np(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\n🔀 Train/Test Split:")
        print(f"   Train: {len(y_train):,} samples")
        print(f"   Test:  {len(y_test):,} samples")

        scratch_results = train_scratch_models(
            X_train, y_train, X_test, y_test, feature_names
        )
        sklearn_results = train_sklearn_model(X_train, y_train, X_test, y_test)

        all_results = scratch_results + sklearn_results

        if not all_results:
            print("\n⚠️  No models trained yet. Please configure your models first.")
            return

        print("\n" + "=" * 60)
        print("📊 FINAL COMPARISON")
        print("=" * 60)
        df_results = pd.DataFrame(
            [
                {k: v for k, v in r.items() if k != "confusion_matrix"}
                for r in all_results
            ]
        )
        print(df_results.to_string(index=False))

        for result in all_results:
            clean_name = (
                result["model"]
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .lower()
            )
            metric_prefix = f"{clean_name}"

            mlflow.log_metrics(
                {
                    f"{metric_prefix}_accuracy": result["accuracy"],
                    f"{metric_prefix}_precision": result["precision"],
                    f"{metric_prefix}_recall": result["recall"],
                    f"{metric_prefix}_f1_score": result["f1_score"],
                }
            )

        print("\n✅ Benchmark complete!")
        print(
            f"   Best F1 Score: {df_results['f1_score'].max():.4f} ({df_results.loc[df_results['f1_score'].idxmax(), 'model']})"
        )


if __name__ == "__main__":
    main()
