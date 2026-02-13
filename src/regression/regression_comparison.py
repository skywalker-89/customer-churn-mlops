"""
Regression Comparison & Visualization

Produces all comparison plots required for the university report:
  1. Metrics comparison bar chart (RMSE, MAE, R², MAPE)
  2. Loss curves (from-scratch models)
  3. Actual vs Predicted scatter plots
  4. Residual plots
  5. Coefficient comparison (scratch vs sklearn)
  6. Feature importance bar chart

Usage:
    # Run benchmark first, then:
    python src/regression/regression_comparison.py

    # Or run standalone (will train models internally):
    python src/regression/regression_comparison.py --standalone
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings

matplotlib.use("Agg")  # non-interactive backend
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "regression")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Metrics (from scratch)
# ============================================================
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def mape_nonzero(y_true, y_pred):
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


# ============================================================
# Plot functions
# ============================================================

def plot_metrics_comparison(results_df, save_path):
    """Bar chart comparing RMSE, MAE, R² across all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    models = results_df["model"].values
    x = np.arange(len(models))

    # RMSE
    ax = axes[0]
    bars = ax.barh(x, results_df["RMSE"], color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(x))))
    ax.set_yticks(x)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("RMSE ($)")
    ax.set_title("RMSE (lower is better)", fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, results_df["RMSE"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"${val:.2f}", va="center", fontsize=8)

    # MAE
    ax = axes[1]
    bars = ax.barh(x, results_df["MAE"], color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(x))))
    ax.set_yticks(x)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("MAE ($)")
    ax.set_title("MAE (lower is better)", fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, results_df["MAE"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"${val:.2f}", va="center", fontsize=8)

    # R²
    ax = axes[2]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in results_df["R2"]]
    bars = ax.barh(x, results_df["R2"], color=colors)
    ax.set_yticks(x)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("R²")
    ax.set_title("R² Score (higher is better)", fontweight="bold")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.invert_yaxis()
    for bar, val in zip(bars, results_df["R2"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.suptitle("Regression Models — Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_loss_curves(models_with_history, save_path):
    """Plot training loss curves for from-scratch models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, model in models_with_history:
        if hasattr(model, "history") and "loss" in model.history and len(model.history["loss"]) > 0:
            losses = model.history["loss"]
            ax.plot(losses, label=name, linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE, normalised)", fontsize=12)
    ax.set_title("Training Loss Curves — From-Scratch Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_actual_vs_predicted(y_test, predictions, save_path):
    """Scatter plot: actual vs predicted for each model."""
    n = len(predictions)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#3498db")
        # Perfect prediction line
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect")
        ax.set_xlabel("Actual Revenue ($)")
        ax.set_ylabel("Predicted Revenue ($)")
        ax.set_title(name, fontsize=10, fontweight="bold")
        r2_val = r2(y_test, y_pred)
        ax.text(0.05, 0.95, f"R²={r2_val:.4f}", transform=ax.transAxes,
                fontsize=10, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Actual vs Predicted Revenue", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_residuals(y_test, predictions, save_path):
    """Residual plot for each model."""
    n = len(predictions)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.3, s=10, color="#e74c3c")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted Revenue ($)")
        ax.set_ylabel("Residual ($)")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.text(0.05, 0.95, f"Mean res={residuals.mean():.2f}\nStd={residuals.std():.2f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Residual Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_coefficient_comparison(scratch_model, sklearn_model, feature_names, save_path):
    """Compare weights from from-scratch vs sklearn."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(feature_names))
    w = 0.35

    ax.barh(x - w / 2, scratch_model.weights, w, label="From Scratch", color="#3498db", alpha=0.8)
    ax.barh(x + w / 2, sklearn_model.coef_, w, label="sklearn", color="#e74c3c", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Coefficient Comparison: From-Scratch vs sklearn", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_feature_importance(weights, feature_names, model_name, save_path):
    """Feature importance based on absolute weight values."""
    abs_weights = np.abs(weights)
    sorted_idx = np.argsort(abs_weights)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(sorted_idx)), abs_weights[sorted_idx], color="#2ecc71", alpha=0.8)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel("|Coefficient|")
    ax.set_title(f"Feature Importance — {model_name}", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {save_path}")


def plot_ridge_alpha_comparison(results_df, save_path):
    """Show how Ridge metrics change with different alpha values."""
    ridge_rows = results_df[results_df["model"].str.contains("Ridge") &
                             results_df["model"].str.contains("sklearn")].copy()
    if len(ridge_rows) < 2:
        print("   ⚠️  Not enough Ridge variants to plot alpha comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ridge_rows["model"].values
    x = np.arange(len(labels))

    ax = axes[0]
    ax.bar(x, ridge_rows["RMSE"], color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSE ($)")
    ax.set_title("Ridge — RMSE vs Alpha", fontweight="bold")

    ax = axes[1]
    ax.bar(x, ridge_rows["R2"], color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title("Ridge — R² vs Alpha", fontweight="bold")

    plt.suptitle("Ridge Regression — Effect of Regularization (Alpha)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {save_path}")


# ============================================================
# Main — run standalone
# ============================================================
def main():
    print("=" * 60)
    print("  REGRESSION COMPARISON & VISUALIZATION")
    print("=" * 60)

    standalone = "--standalone" in sys.argv

    if standalone:
        # Import and run benchmark inline
        from src.regression.regression_benchmark import (
            load_data, train_test_split_np,
            train_scratch_models, train_library_models,
        )

        print("\n📥 Loading data...")
        X, y, feature_names = load_data(strategy="converting_only")
        X_train, X_test, y_train, y_test = train_test_split_np(X, y)

        print("\n🔧 Training from-scratch models...")
        scratch_results = train_scratch_models(X_train, y_train, X_test, y_test, feature_names)

        print("\n🔧 Training library models...")
        lib_results = train_library_models(X_train, y_train, X_test, y_test, feature_names)

        all_results = scratch_results + lib_results

    else:
        # Load pre-computed results from benchmark
        csv_path = os.path.join(PROJECT_ROOT, "reports", "regression_benchmark_results.csv")
        npz_path = os.path.join(PROJECT_ROOT, "reports", "regression_predictions.npz")

        if not os.path.exists(csv_path):
            print(f"❌ {csv_path} not found. Run regression_benchmark.py first, or use --standalone")
            sys.exit(1)

        results_df = pd.read_csv(csv_path)
        data = np.load(npz_path, allow_pickle=True)
        y_test = data["y_test"]
        predictions = {k: data[k] for k in data.files if k != "y_test"}

        # Plot from pre-saved data
        print("\n📊 Generating plots from saved benchmark data...")

        plot_metrics_comparison(results_df, os.path.join(FIGURES_DIR, "01_metrics_comparison.png"))

        plot_actual_vs_predicted(y_test, predictions, os.path.join(FIGURES_DIR, "03_actual_vs_predicted.png"))
        plot_residuals(y_test, predictions, os.path.join(FIGURES_DIR, "04_residuals.png"))
        plot_ridge_alpha_comparison(results_df, os.path.join(FIGURES_DIR, "06_ridge_alpha.png"))

        print(f"\n✅ All plots saved to {FIGURES_DIR}/")
        return

    # Build results dataframe
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "model_obj"} for r in all_results])

    # Collect predictions
    predictions = {}
    for r in all_results:
        model = r.get("model_obj")
        if model is not None:
            name = r["model"]
            try:
                if "best_idx" in r:
                    predictions[name] = model.predict(X_test[:, [r["best_idx"]]])
                else:
                    predictions[name] = model.predict(X_test)
            except Exception:
                pass

    # Collect from-scratch models with history for loss curves
    scratch_models = [(r["model"], r["model_obj"]) for r in scratch_results if r.get("model_obj")]

    print("\n📊 Generating plots...")

    # 1. Metrics comparison
    plot_metrics_comparison(results_df, os.path.join(FIGURES_DIR, "01_metrics_comparison.png"))

    # 2. Loss curves
    plot_loss_curves(scratch_models, os.path.join(FIGURES_DIR, "02_loss_curves.png"))

    # 3. Actual vs Predicted (top models only to keep readable)
    top_models = {}
    for name in ["Linear (scratch)", "Multiple (scratch)", "Polynomial (scratch)",
                  "Ridge (scratch)", "Multiple (sklearn)", "Ridge (sklearn)"]:
        if name in predictions:
            top_models[name] = predictions[name]
    plot_actual_vs_predicted(y_test, top_models, os.path.join(FIGURES_DIR, "03_actual_vs_predicted.png"))

    # 4. Residuals
    plot_residuals(y_test, top_models, os.path.join(FIGURES_DIR, "04_residuals.png"))

    # 5. Coefficient comparison (Multiple: scratch vs sklearn)
    scratch_multi = next((r["model_obj"] for r in scratch_results
                          if "Multiple (scratch)" in r["model"]), None)
    sklearn_multi = next((r["model_obj"] for r in lib_results
                          if r["model"] == "Multiple (sklearn)"), None)
    if scratch_multi and sklearn_multi and hasattr(sklearn_multi, "coef_"):
        plot_coefficient_comparison(
            scratch_multi, sklearn_multi, feature_names,
            os.path.join(FIGURES_DIR, "05_coefficient_comparison.png"),
        )

    # 6. Ridge alpha comparison (REMOVED)
    # plot_ridge_alpha_comparison(results_df, os.path.join(FIGURES_DIR, "06_ridge_alpha.png"))

    # 7. Feature importance from Ridge
    ridge_scratch = next((r["model_obj"] for r in scratch_results
                          if "Ridge" in r["model"]), None)
    if ridge_scratch and ridge_scratch.weights is not None:
        plot_feature_importance(
            ridge_scratch.weights, feature_names, "Ridge Regression (scratch)",
            os.path.join(FIGURES_DIR, "07_feature_importance.png"),
        )

    print(f"\n✅ All plots saved to {FIGURES_DIR}/")
    print("\n" + "=" * 60)
    print("  📊 FINAL RESULTS TABLE")
    print("=" * 60)
    print(results_df[["model", "RMSE", "MAE", "R2", "MAPE_%"]].to_string(index=False))


if __name__ == "__main__":
    main()
