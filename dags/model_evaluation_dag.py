"""
Model Evaluation DAG — Score ALL Trained Models & Compare

This DAG loads already-trained models from MinIO, evaluates them on the
holdout test set, computes metrics, and generates a comparison report.

Classification Models (11 scratch + 1 sklearn):
    Logistic Regression, Decision Tree, Random Forest, SVM,
    Random Forest + PCA, SVM + PCA, K-Means, Agglomerative Clustering,
    Perceptron, MLP, Custom Model, Sklearn Random Forest

Regression Models (4 scratch + 1 sklearn):
    Linear, Multiple, Polynomial, XGBoost (scratch), XGBoost (sklearn)

Metrics:
    Classification → Accuracy, Precision, Recall, F1, Confusion Matrix
    Regression     → RMSE, MAE, R², MAPE
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
project_root = Path("/opt/airflow")
sys.path.insert(0, str(project_root))


# ============================================================
# Metric helpers (pure numpy, no sklearn dependency)
# ============================================================
def _accuracy(y_true, y_pred):
    import numpy as np
    return float(np.mean(y_true == y_pred))

def _precision(y_true, y_pred):
    import numpy as np
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

def _recall(y_true, y_pred):
    import numpy as np
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

def _f1_score(y_true, y_pred):
    p = _precision(y_true, y_pred)
    r = _recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

def _confusion_matrix(y_true, y_pred):
    import numpy as np
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}

def _rmse(y_true, y_pred):
    import numpy as np
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mae(y_true, y_pred):
    import numpy as np
    return float(np.mean(np.abs(y_true - y_pred)))

def _r2(y_true, y_pred):
    import numpy as np
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

def _mape(y_true, y_pred):
    import numpy as np
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


# ============================================================
# Task 1: Validate test data
# ============================================================
def validate_test_data():
    """Load training_data.parquet from MinIO and validate."""
    from minio import Minio
    from io import BytesIO
    import pandas as pd
    import os

    print("🔍 Validating test data for model evaluation...")

    client = Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key="minio_admin",
        secret_key="minio_password",
        secure=False,
    )

    response = client.get_object("processed-data", "training_data.parquet")
    df = pd.read_parquet(BytesIO(response.read()))
    response.close()
    response.release_conn()

    print(f"✅ Training data loaded: {len(df):,} rows, {len(df.columns)} columns")

    required_cols = ["total_sales", "churned"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"📊 Target distributions:")
    print(f"   - total_sales: mean=${df['total_sales'].mean():.2f}, std=${df['total_sales'].std():.2f}")
    print(f"   - churned: {df['churned'].value_counts().to_dict()}")

    return {"rows": len(df), "columns": len(df.columns), "targets_valid": True}


# ============================================================
# Task 2: Evaluate ALL classification models
# ============================================================
def evaluate_classification_models():
    """Load all trained classification models from MinIO and evaluate."""
    import os
    import numpy as np
    import pandas as pd
    import pickle
    from io import BytesIO
    from minio import Minio

    os.environ["MINIO_ENDPOINT"] = "minio:9000"
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"

    print("=" * 70)
    print("🎯 CLASSIFICATION MODEL EVALUATION")
    print("=" * 70)

    # --- Load data (same split as benchmark) ---
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    client = Minio(endpoint, access_key="minio_admin", secret_key="minio_password", secure=False)

    response = client.get_object("processed-data", "training_data.parquet")
    df = pd.read_parquet(BytesIO(response.read()))
    response.close()
    response.release_conn()

    drop_cols = ["total_sales", "churned", "clv_per_year"]
    X_df = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
    y = df["churned"].values.astype(np.int32)

    # Same split as benchmarks (seed=42, test=20%)
    rng = np.random.RandomState(42)
    n = len(X_df)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(np.floor(n * 0.2))
    X_test = X_df.values.astype(np.float64)[idx[:n_test]]
    y_test = y[idx[:n_test]]

    print(f"📊 Test set: {len(y_test):,} samples")

    # --- Define models to load ---
    scratch_models = [
        ("Logistic Regression", "logisticregression_classification"),
        ("Decision Tree", "decisiontree_classification"),
        ("Random Forest", "randomforest_classification"),
        ("SVM", "svm_classification"),
        ("Random Forest + PCA", "randomforestpca_classification"),
        ("SVM + PCA", "svmpca_classification"),
        ("K-Means Clustering", "kmeans_classification"),
        ("Agglomerative Clustering", "agglomerativeclustering_classification"),
        ("Perceptron", "perceptron_classification"),
        ("MLP", "mlp_classification"),
        ("Custom Model", "custommodel_classification"),
    ]

    results = []

    # --- Evaluate scratch models ---
    for display_name, minio_key in scratch_models:
        print(f"\n{'─' * 60}")
        print(f"📊 Evaluating: {display_name} (scratch)")
        print(f"{'─' * 60}")

        try:
            response = client.get_object("models", f"{minio_key}_latest.pkl")
            model = pickle.loads(response.read())
            response.close()
            response.release_conn()
            print(f"   ✅ Loaded model from MinIO: {minio_key}")
        except Exception as e:
            print(f"   ❌ Could not load model '{minio_key}': {e}")
            results.append({
                "model": f"{display_name} (scratch)",
                "accuracy": None, "precision": None, "recall": None,
                "f1_score": None, "confusion_matrix": None,
                "status": "NOT FOUND",
            })
            continue

        try:
            y_pred = model.predict(X_test)
            y_pred = (y_pred > 0.5).astype(np.int32) if y_pred.dtype == np.float64 else y_pred

            acc  = _accuracy(y_test, y_pred)
            prec = _precision(y_test, y_pred)
            rec  = _recall(y_test, y_pred)
            f1   = _f1_score(y_test, y_pred)
            cm   = _confusion_matrix(y_test, y_pred)

            print(f"   Accuracy:  {acc:.4f}")
            print(f"   Precision: {prec:.4f}")
            print(f"   Recall:    {rec:.4f}")
            print(f"   F1 Score:  {f1:.4f}")
            print(f"   Confusion: TN={cm['TN']}  FP={cm['FP']}  FN={cm['FN']}  TP={cm['TP']}")

            results.append({
                "model": f"{display_name} (scratch)",
                "accuracy": acc, "precision": prec, "recall": rec,
                "f1_score": f1, "confusion_matrix": cm,
                "status": "OK",
            })
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            results.append({
                "model": f"{display_name} (scratch)",
                "accuracy": None, "precision": None, "recall": None,
                "f1_score": None, "confusion_matrix": None,
                "status": f"ERROR: {e}",
            })

    # --- Evaluate sklearn model ---
    print(f"\n{'─' * 60}")
    print(f"📊 Evaluating: Random Forest (sklearn)")
    print(f"{'─' * 60}")
    try:
        response = client.get_object("models", "random_forest_sklearn_latest.pkl")
        sklearn_model = pickle.loads(response.read())
        response.close()
        response.release_conn()
        print("   ✅ Loaded sklearn model from MinIO")

        y_pred = sklearn_model.predict(X_test)
        acc  = _accuracy(y_test, y_pred)
        prec = _precision(y_test, y_pred)
        rec  = _recall(y_test, y_pred)
        f1   = _f1_score(y_test, y_pred)
        cm   = _confusion_matrix(y_test, y_pred)

        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   Confusion: TN={cm['TN']}  FP={cm['FP']}  FN={cm['FN']}  TP={cm['TP']}")

        results.append({
            "model": "Random Forest (sklearn)",
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1_score": f1, "confusion_matrix": cm,
            "status": "OK",
        })
    except Exception as e:
        print(f"   ❌ Could not load/evaluate sklearn model: {e}")
        results.append({
            "model": "Random Forest (sklearn)",
            "accuracy": None, "precision": None, "recall": None,
            "f1_score": None, "confusion_matrix": None,
            "status": f"ERROR: {e}",
        })

    print(f"\n✅ Classification evaluation complete: {len(results)} models evaluated")
    return results


# ============================================================
# Task 3: Evaluate ALL regression models
# ============================================================
def evaluate_regression_models():
    """Load all trained regression models from MinIO and evaluate."""
    import os
    import numpy as np
    import pandas as pd
    import pickle
    from io import BytesIO
    from minio import Minio

    os.environ["MINIO_ENDPOINT"] = "minio:9000"
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"

    print("=" * 70)
    print("📈 REGRESSION MODEL EVALUATION")
    print("=" * 70)

    # --- Load data (same split as benchmark) ---
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    client = Minio(endpoint, access_key="minio_admin", secret_key="minio_password", secure=False)

    response = client.get_object("processed-data", "training_data.parquet")
    df = pd.read_parquet(BytesIO(response.read()))
    response.close()
    response.release_conn()

    drop_cols = ["total_sales", "churned", "clv_per_year"]
    X_df = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
    y = df["total_sales"].values.astype(np.float64)

    # Same split as benchmarks (seed=42, test=20%)
    rng = np.random.RandomState(42)
    n = len(X_df)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(np.floor(n * 0.2))
    X_test = X_df.values.astype(np.float64)[idx[:n_test]]
    y_test = y[idx[:n_test]]

    print(f"📊 Test set: {len(y_test):,} samples")
    print(f"   Target (total_sales): mean=${y_test.mean():.2f}, std=${y_test.std():.2f}")

    # --- Define models to load ---
    scratch_models = [
        ("Linear Regression", "linear_regression"),
        ("Multiple Regression", "multiple_regression"),
        ("Polynomial Regression", "polynomial_regression"),
        ("XGBoost", "xgboost_regression"),
    ]

    results = []

    # --- Evaluate scratch models ---
    for display_name, minio_key in scratch_models:
        print(f"\n{'─' * 60}")
        print(f"📊 Evaluating: {display_name} (scratch)")
        print(f"{'─' * 60}")

        try:
            response = client.get_object("models", f"{minio_key}_latest.pkl")
            model = pickle.loads(response.read())
            response.close()
            response.release_conn()
            print(f"   ✅ Loaded model from MinIO: {minio_key}")
        except Exception as e:
            print(f"   ❌ Could not load model '{minio_key}': {e}")
            results.append({
                "model": f"{display_name} (scratch)",
                "rmse": None, "mae": None, "r2": None, "mape": None,
                "status": "NOT FOUND",
            })
            continue

        try:
            y_pred = model.predict(X_test)

            r = _rmse(y_test, y_pred)
            m = _mae(y_test, y_pred)
            r2 = _r2(y_test, y_pred)
            mp = _mape(y_test, y_pred)

            print(f"   RMSE:  ${r:,.2f}")
            print(f"   MAE:   ${m:,.2f}")
            print(f"   R²:    {r2:.4f}")
            print(f"   MAPE:  {mp:.2f}%")

            results.append({
                "model": f"{display_name} (scratch)",
                "rmse": r, "mae": m, "r2": r2, "mape": mp,
                "status": "OK",
            })
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            results.append({
                "model": f"{display_name} (scratch)",
                "rmse": None, "mae": None, "r2": None, "mape": None,
                "status": f"ERROR: {e}",
            })

    # --- Evaluate sklearn XGBoost ---
    print(f"\n{'─' * 60}")
    print(f"📊 Evaluating: XGBoost (sklearn)")
    print(f"{'─' * 60}")
    try:
        response = client.get_object("models", "xgboost_sklearn_latest.pkl")
        sklearn_model = pickle.loads(response.read())
        response.close()
        response.release_conn()
        print("   ✅ Loaded sklearn XGBoost model from MinIO")

        y_pred = sklearn_model.predict(X_test)

        r = _rmse(y_test, y_pred)
        m = _mae(y_test, y_pred)
        r2_val = _r2(y_test, y_pred)
        mp = _mape(y_test, y_pred)

        print(f"   RMSE:  ${r:,.2f}")
        print(f"   MAE:   ${m:,.2f}")
        print(f"   R²:    {r2_val:.4f}")
        print(f"   MAPE:  {mp:.2f}%")

        results.append({
            "model": "XGBoost (sklearn)",
            "rmse": r, "mae": m, "r2": r2_val, "mape": mp,
            "status": "OK",
        })
    except Exception as e:
        print(f"   ❌ Could not load/evaluate sklearn XGBoost: {e}")
        results.append({
            "model": "XGBoost (sklearn)",
            "rmse": None, "mae": None, "r2": None, "mape": None,
            "status": f"ERROR: {e}",
        })

    print(f"\n✅ Regression evaluation complete: {len(results)} models evaluated")
    return results


# ============================================================
# Task 4: Generate comparison report
# ============================================================
def generate_evaluation_comparison(**context):
    """Pull results from XCom, print comparison, rank models, save report."""
    import json
    import pandas as pd
    import mlflow
    import os

    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"

    print("=" * 70)
    print("📋 MODEL EVALUATION COMPARISON REPORT")
    print("=" * 70)

    ti = context["task_instance"]
    clf_results = ti.xcom_pull(task_ids="evaluate_classification_models") or []
    reg_results = ti.xcom_pull(task_ids="evaluate_regression_models") or []

    # ==========================================================
    # Classification comparison
    # ==========================================================
    print("\n" + "=" * 70)
    print("🎯 CLASSIFICATION MODELS — COMPARISON")
    print("=" * 70)

    clf_ok = [r for r in clf_results if r.get("status") == "OK"]
    clf_fail = [r for r in clf_results if r.get("status") != "OK"]

    if clf_ok:
        df_clf = pd.DataFrame([
            {k: v for k, v in r.items() if k not in ("confusion_matrix", "status")}
            for r in clf_ok
        ])
        # Sort by F1 score descending
        df_clf = df_clf.sort_values("f1_score", ascending=False).reset_index(drop=True)
        df_clf.index = df_clf.index + 1  # 1-based rank
        df_clf.index.name = "Rank"

        print("\n" + df_clf.to_string())

        best_clf = df_clf.iloc[0]
        worst_clf = df_clf.iloc[-1]
        print(f"\n🏆 Best Classification Model:  {best_clf['model']}  (F1={best_clf['f1_score']:.4f})")
        print(f"📉 Worst Classification Model: {worst_clf['model']}  (F1={worst_clf['f1_score']:.4f})")

        # Confusion matrices
        print("\n📊 Confusion Matrices:")
        for r in clf_ok:
            cm = r["confusion_matrix"]
            print(f"   {r['model']}: TN={cm['TN']}  FP={cm['FP']}  FN={cm['FN']}  TP={cm['TP']}")
    else:
        print("   ⚠️ No classification models were successfully evaluated.")

    if clf_fail:
        print(f"\n⚠️  {len(clf_fail)} classification model(s) could not be evaluated:")
        for r in clf_fail:
            print(f"   - {r['model']}: {r['status']}")

    # ==========================================================
    # Regression comparison
    # ==========================================================
    print("\n" + "=" * 70)
    print("📈 REGRESSION MODELS — COMPARISON")
    print("=" * 70)

    reg_ok = [r for r in reg_results if r.get("status") == "OK"]
    reg_fail = [r for r in reg_results if r.get("status") != "OK"]

    if reg_ok:
        df_reg = pd.DataFrame([
            {k: v for k, v in r.items() if k != "status"}
            for r in reg_ok
        ])
        # Sort by R² descending (higher is better)
        df_reg = df_reg.sort_values("r2", ascending=False).reset_index(drop=True)
        df_reg.index = df_reg.index + 1
        df_reg.index.name = "Rank"

        print("\n" + df_reg.to_string())

        best_reg = df_reg.iloc[0]
        worst_reg = df_reg.iloc[-1]
        print(f"\n🏆 Best Regression Model:  {best_reg['model']}  (R²={best_reg['r2']:.4f})")
        print(f"📉 Worst Regression Model: {worst_reg['model']}  (R²={worst_reg['r2']:.4f})")
    else:
        print("   ⚠️ No regression models were successfully evaluated.")

    if reg_fail:
        print(f"\n⚠️  {len(reg_fail)} regression model(s) could not be evaluated:")
        for r in reg_fail:
            print(f"   - {r['model']}: {r['status']}")

    # ==========================================================
    # Summary
    # ==========================================================
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print(f"   Classification models evaluated: {len(clf_ok)}/{len(clf_results)}")
    print(f"   Regression models evaluated:     {len(reg_ok)}/{len(reg_results)}")
    print(f"   Total models:                    {len(clf_ok) + len(reg_ok)}/{len(clf_results) + len(reg_results)}")

    # ==========================================================
    # Save JSON report
    # ==========================================================
    from pathlib import Path

    reports_dir = Path("/opt/airflow/reports/evaluation")
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": datetime.now().isoformat(),
        "classification": {
            "results": clf_results,
            "models_evaluated": len(clf_ok),
            "models_failed": len(clf_fail),
            "best_model": clf_ok[0]["model"] if clf_ok else None,
            "best_f1": max((r["f1_score"] for r in clf_ok), default=None),
        },
        "regression": {
            "results": reg_results,
            "models_evaluated": len(reg_ok),
            "models_failed": len(reg_fail),
            "best_model": reg_ok[0]["model"] if reg_ok else None,
            "best_r2": max((r["r2"] for r in reg_ok), default=None),
        },
        "total_models_evaluated": len(clf_ok) + len(reg_ok),
    }

    report_file = reports_dir / f"evaluation_report_{ts}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Report saved to: {report_file}")

    # ==========================================================
    # Log to MLflow
    # ==========================================================
    try:
        mlflow.set_experiment("model_evaluation_comparison")
        with mlflow.start_run(run_name=f"evaluation_{ts}"):
            for r in clf_ok:
                prefix = r["model"].replace(" ", "_").replace("(", "").replace(")", "").lower()
                mlflow.log_metrics({
                    f"{prefix}_accuracy": r["accuracy"],
                    f"{prefix}_precision": r["precision"],
                    f"{prefix}_recall": r["recall"],
                    f"{prefix}_f1_score": r["f1_score"],
                })
            for r in reg_ok:
                prefix = r["model"].replace(" ", "_").replace("(", "").replace(")", "").lower()
                mlflow.log_metrics({
                    f"{prefix}_rmse": r["rmse"],
                    f"{prefix}_mae": r["mae"],
                    f"{prefix}_r2": r["r2"],
                    f"{prefix}_mape": r["mape"],
                })
            mlflow.log_artifact(str(report_file))
        print("✅ Metrics & report logged to MLflow")
    except Exception as e:
        print(f"⚠️ MLflow logging failed (non-fatal): {e}")

    # Clean up old reports (keep last 5)
    old_reports = sorted(reports_dir.glob("evaluation_report_*.json"), reverse=True)
    for old in old_reports[5:]:
        try:
            old.unlink()
        except Exception:
            pass

    print("\n✅ Evaluation comparison complete!")
    return str(report_file)


# ============================================================
# DAG Definition
# ============================================================
default_args = {
    "owner": "ml_engineer",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    "model_evaluation_comparison",
    default_args=default_args,
    description="Evaluate all classification & regression models, compare scores, and generate report",
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=["evaluation", "classification", "regression", "comparison", "models"],
    max_active_runs=1,
) as dag:

    validate_task = PythonOperator(
        task_id="validate_test_data",
        python_callable=validate_test_data,
    )

    clf_eval_task = PythonOperator(
        task_id="evaluate_classification_models",
        python_callable=evaluate_classification_models,
    )

    reg_eval_task = PythonOperator(
        task_id="evaluate_regression_models",
        python_callable=evaluate_regression_models,
    )

    comparison_task = PythonOperator(
        task_id="generate_evaluation_comparison",
        python_callable=generate_evaluation_comparison,
        provide_context=True,
    )

    # Data validation → Evaluate in parallel → Comparison report
    validate_task >> [clf_eval_task, reg_eval_task] >> comparison_task
