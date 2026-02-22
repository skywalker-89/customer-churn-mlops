"""
Model Performance Visualization Script (Seaborn-free version)

This script creates comprehensive visualizations for model performance analysis:
- Classification: Confusion matrices, ROC curves, Precision-Recall curves, metrics comparison
- Regression: Predicted vs Actual scatter plots, residual plots, error distributions, R² comparison

Uses only matplotlib and numpy to avoid dependency issues.
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
from minio import Minio
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from datetime import datetime

# Set matplotlib style
plt.style.use("default")

# Create reports directory for visualizations - use local path instead of /opt/airflow
REPORTS_DIR = Path(os.getenv("VIZ_OUTPUT_DIR", "./visualization_output"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load test data from MinIO"""
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

    return df


def prepare_test_data(df):
    """Prepare test data with same split as evaluation DAG"""
    drop_cols = ["total_sales", "churned", "clv_per_year"]
    X_df = df.drop(columns=drop_cols, errors="ignore").select_dtypes(
        include=[np.number]
    )

    # Classification target
    y_clf = df["churned"].values.astype(np.int32)
    # Regression target
    y_reg = df["total_sales"].values.astype(np.float64)

    # Same split as benchmarks (seed=42, test=20%)
    rng = np.random.RandomState(42)
    n = len(X_df)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(np.floor(n * 0.2))

    X_test = X_df.values.astype(np.float64)[idx[:n_test]]
    y_test_clf = y_clf[idx[:n_test]]
    y_test_reg = y_reg[idx[:n_test]]

    return X_test, y_test_clf, y_test_reg


def load_model(model_key):
    """Load model from MinIO"""
    client = Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key="minio_admin",
        secret_key="minio_password",
        secure=False,
    )

    try:
        response = client.get_object("models", f"{model_key}_latest.pkl")
        model = pickle.loads(response.read())
        response.close()
        response.release_conn()
        return model
    except Exception as e:
        print(f"Could not load model {model_key}: {e}")
        return None


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Create confusion matrix heatmap using matplotlib only"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap using matplotlib
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Not Churned", "Churned"],
        yticklabels=["Not Churned", "Churned"],
        title=f"Confusion Matrix - {model_name}",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(models_data, save_path):
    """Plot ROC curves for multiple models"""
    plt.figure(figsize=(10, 8))

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for i, (model_name, y_true, y_proba) in enumerate(models_data):
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                color=colors[i % len(colors)],
                lw=2,
                label=f"{model_name} (AUC = {roc_auc:.3f})",
            )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Classification Models")
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curves(models_data, save_path):
    """Plot Precision-Recall curves for multiple models"""
    plt.figure(figsize=(10, 8))

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for i, (model_name, y_true, y_proba) in enumerate(models_data):
        if y_proba is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = auc(recall, precision)
            plt.plot(
                recall,
                precision,
                color=colors[i % len(colors)],
                lw=2,
                label=f"{model_name} (AUC = {pr_auc:.3f})",
            )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Classification Models")
    plt.legend(loc="lower left", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_classification_metrics_comparison(results, save_path):
    """Create bar chart comparing classification metrics"""
    successful_results = [r for r in results if r.get("status") == "OK"]

    if not successful_results:
        print("No successful classification results to plot")
        return

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    models = [r["model"] for r in successful_results]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(models))
    width = 0.2

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in successful_results]
        ax.bar(
            x + i * width,
            values,
            width,
            label=metric.title(),
            color=colors[i],
            alpha=0.8,
        )

    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    ax.set_title("Classification Models - Metrics Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_predicted_vs_actual(y_true, y_pred, model_name, save_path):
    """Create predicted vs actual scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, color="blue")

    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
    )

    # Calculate R²
    from sklearn.metrics import r2_score

    r2 = r2_score(y_true, y_pred)

    ax.set_xlabel("Actual Values ($)")
    ax.set_ylabel("Predicted Values ($)")
    ax.set_title(f"Predicted vs Actual - {model_name}\nR² = {r2:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add correlation coefficient
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true, y_pred, model_name, save_path):
    """Create residual plot"""
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6, s=30, color="blue")
    ax1.axhline(y=0, color="r", linestyle="--", lw=2)
    ax1.set_xlabel("Predicted Values ($)")
    ax1.set_ylabel("Residuals ($)")
    ax1.set_title(f"Residuals vs Predicted - {model_name}")
    ax1.grid(True, alpha=0.3)

    # Residuals histogram
    ax2.hist(residuals, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    ax2.axvline(x=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("Residuals ($)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Residuals Distribution - {model_name}")
    ax2.grid(True, alpha=0.3)

    # Add statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax2.text(
        0.05,
        0.95,
        f"Mean: ${mean_residual:.2f}\nStd: ${std_residual:.2f}",
        transform=ax2.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_distribution(y_true, y_pred, model_name, save_path):
    """Create error distribution histogram with percentiles"""
    errors = np.abs(y_true - y_pred)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram
    n, bins, patches = ax.hist(
        errors, bins=30, alpha=0.7, color="lightcoral", edgecolor="black"
    )

    # Add percentiles
    percentiles = [50, 75, 90, 95]
    colors = ["green", "orange", "red", "darkred"]

    for p, color in zip(percentiles, colors):
        value = np.percentile(errors, p)
        ax.axvline(x=value, color=color, linestyle="--", alpha=0.8, lw=2)
        ax.text(
            value,
            max(n) * 0.9,
            f"{p}th: ${value:.2f}",
            rotation=90,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
        )

    ax.set_xlabel("Absolute Error ($)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Error Distribution - {model_name}")
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    ax.text(
        0.05,
        0.95,
        f"Mean Error: ${mean_error:.2f}\nMedian Error: ${median_error:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_regression_metrics_comparison(results, save_path):
    """Create bar chart comparing regression metrics"""
    successful_results = [r for r in results if r.get("status") == "OK"]

    if not successful_results:
        print("No successful regression results to plot")
        return

    metrics = ["rmse", "mae", "r2", "mape"]
    metric_names = ["RMSE", "MAE", "R²", "MAPE (%)"]
    models = [r["model"] for r in successful_results]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    axes = [ax1, ax2, ax3, ax4]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (metric, metric_name, ax) in enumerate(zip(metrics, metric_names, axes)):
        values = [r[metric] for r in successful_results]
        bars = ax.bar(models, values, color=colors[i], alpha=0.7, edgecolor="black")
        ax.set_title(f"{metric_name} Comparison")
        ax.set_ylabel(metric_name)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}" if metric == "r2" else f"${value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle("Regression Models - Metrics Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_classification_visualizations():
    """Create all classification visualizations"""
    print("🎯 Creating Classification Model Visualizations...")

    # Load data
    df = load_data()
    X_test, y_test_clf, _ = prepare_test_data(df)

    # Define classification models
    classification_models = [
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
        ("Random Forest (sklearn)", "random_forest_sklearn"),
    ]

    results = []
    roc_data = []

    # Create timestamp for this visualization run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = REPORTS_DIR / f"classification_{timestamp}"
    viz_dir.mkdir(exist_ok=True)

    for display_name, model_key in classification_models:
        print(f"Processing {display_name}...")

        model = load_model(model_key)
        if model is None:
            continue

        try:
            # Get predictions
            y_pred = model.predict(X_test)
            if y_pred.dtype == np.float64:
                y_proba = y_pred
                y_pred = (y_pred > 0.5).astype(np.int32)
            else:
                y_proba = None

            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
            )

            accuracy = accuracy_score(y_test_clf, y_pred)
            precision = precision_score(y_test_clf, y_pred, zero_division=0)
            recall = recall_score(y_test_clf, y_pred, zero_division=0)
            f1 = f1_score(y_test_clf, y_pred, zero_division=0)

            # Store results
            results.append(
                {
                    "model": display_name,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "status": "OK",
                }
            )

            # Store ROC data if probabilities available
            if y_proba is not None:
                roc_data.append((display_name, y_test_clf, y_proba))

            # Create confusion matrix
            cm_path = viz_dir / f"confusion_matrix_{model_key}.png"
            plot_confusion_matrix(y_test_clf, y_pred, display_name, cm_path)

        except Exception as e:
            print(f"Error processing {display_name}: {e}")
            results.append(
                {
                    "model": display_name,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1_score": None,
                    "status": f"ERROR: {e}",
                }
            )

    # Create comparison charts
    if results:
        metrics_path = viz_dir / "metrics_comparison.png"
        plot_classification_metrics_comparison(results, metrics_path)

    if roc_data:
        roc_path = viz_dir / "roc_curves.png"
        plot_roc_curves(roc_data, roc_path)

        pr_path = viz_dir / "precision_recall_curves.png"
        plot_precision_recall_curves(roc_data, pr_path)

    print(f"✅ Classification visualizations saved to: {viz_dir}")
    return viz_dir


def create_regression_visualizations():
    """Create all regression visualizations"""
    print("📈 Creating Regression Model Visualizations...")

    # Load data
    df = load_data()
    X_test, _, y_test_reg = prepare_test_data(df)

    # Define regression models
    regression_models = [
        ("Linear Regression", "linear_regression"),
        ("Multiple Regression", "multiple_regression"),
        ("Polynomial Regression", "polynomial_regression"),
        ("XGBoost", "xgboost_regression"),
        ("XGBoost (sklearn)", "xgboost_sklearn"),
    ]

    results = []

    # Create timestamp for this visualization run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = REPORTS_DIR / f"regression_{timestamp}"
    viz_dir.mkdir(exist_ok=True)

    for display_name, model_key in regression_models:
        print(f"Processing {display_name}...")

        model = load_model(model_key)
        if model is None:
            continue

        try:
            # Get predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            from sklearn.metrics import (
                mean_squared_error,
                mean_absolute_error,
                r2_score,
            )

            rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
            mae = mean_absolute_error(y_test_reg, y_pred)
            r2 = r2_score(y_test_reg, y_pred)
            mape = np.mean(np.abs((y_test_reg - y_pred) / y_test_reg)) * 100

            # Store results
            results.append(
                {
                    "model": display_name,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "mape": mape,
                    "status": "OK",
                }
            )

            # Create visualizations
            pred_vs_actual_path = viz_dir / f"predicted_vs_actual_{model_key}.png"
            plot_predicted_vs_actual(
                y_test_reg, y_pred, display_name, pred_vs_actual_path
            )

            residuals_path = viz_dir / f"residuals_{model_key}.png"
            plot_residuals(y_test_reg, y_pred, display_name, residuals_path)

            error_dist_path = viz_dir / f"error_distribution_{model_key}.png"
            plot_error_distribution(y_test_reg, y_pred, display_name, error_dist_path)

        except Exception as e:
            print(f"Error processing {display_name}: {e}")
            results.append(
                {
                    "model": display_name,
                    "rmse": None,
                    "mae": None,
                    "r2": None,
                    "mape": None,
                    "status": f"ERROR: {e}",
                }
            )

    # Create comparison charts
    if results:
        metrics_path = viz_dir / "metrics_comparison.png"
        plot_regression_metrics_comparison(results, metrics_path)

    print(f"✅ Regression visualizations saved to: {viz_dir}")
    return viz_dir


def main():
    """Main function to create all visualizations"""
    print("🚀 Starting Model Performance Visualization...")

    # Create classification visualizations
    clf_viz_dir = create_classification_visualizations()

    # Create regression visualizations
    reg_viz_dir = create_regression_visualizations()

    print("\n" + "=" * 60)
    print("🎉 Visualization Complete!")
    print(f"📊 Classification visualizations: {clf_viz_dir}")
    print(f"📈 Regression visualizations: {reg_viz_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
