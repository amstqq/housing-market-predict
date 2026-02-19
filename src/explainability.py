"""
Vancouver Housing Market - Model Explainability
================================================
SHAP-based explanations, feature importance, and partial dependence plots.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os

from preprocessing import load_data, preprocess, split_data

os.makedirs("outputs", exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")


def load_model_and_data():
    """Load the best model and preprocessed data."""
    model = joblib.load("models/best_model.pkl")
    model_name = joblib.load("models/best_model_name.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    
    df = load_data()
    X, y, _ = preprocess(df, fit=False)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"📦 Loaded model: {model_name}")
    return model, model_name, X_train, X_test, y_train, y_test, feature_names


def compute_shap_values(model, X_data, model_name):
    """Compute SHAP values using the appropriate explainer."""
    print("🔍 Computing SHAP values (this may take a moment)...")
    
    if "XGBoost" in model_name or "Random Forest" in model_name:
        explainer = shap.TreeExplainer(model)
    else:
        # Use a subsample for KernelExplainer (slow)
        background = shap.sample(X_data, 100)
        explainer = shap.KernelExplainer(model.predict, background)
    
    shap_values = explainer(X_data)
    print(f"✅ SHAP values computed for {X_data.shape[0]} samples × {X_data.shape[1]} features")
    return shap_values, explainer


def plot_shap_summary(shap_values, X_data, feature_names):
    """SHAP bee swarm summary plot — global feature importance."""
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_data, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SHAP Feature Importance (Bee Swarm)", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved outputs/shap_summary.png")


def plot_shap_bar(shap_values, feature_names):
    """SHAP bar plot — mean absolute SHAP values."""
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title("Mean |SHAP| Feature Importance", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved outputs/shap_bar.png")


def plot_shap_waterfall(shap_values, idx=0):
    """SHAP waterfall plot for a single prediction — shows contribution of each feature."""
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Prediction Breakdown (Sample #{idx})",
              fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("outputs/shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved outputs/shap_waterfall.png (sample #{idx})")


def plot_shap_dependence(shap_values, X_data, feature_names):
    """Partial dependence-style plots for top features."""
    # Identify top 4 numeric features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[::-1]
    
    # Filter to numeric-looking features (not one-hot encoded)
    numeric_features = ["sqft", "age", "bedrooms", "bathrooms", "walk_score",
                        "distance_to_downtown_km", "lot_size", "year_built"]
    top_numeric = [f for f in numeric_features if f in feature_names][:4]
    
    if len(top_numeric) == 0:
        print("⚠️  No numeric features found for dependence plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(top_numeric):
        if feat in feature_names:
            feat_idx = feature_names.index(feat)
            ax = axes[i]
            shap.dependence_plot(
                feat_idx, shap_values.values, X_data,
                feature_names=feature_names, ax=ax, show=False,
                alpha=0.5
            )
            ax.set_title(f"SHAP Dependence: {feat}", fontsize=12, fontweight="bold")
    
    # Hide unused axes
    for j in range(len(top_numeric), 4):
        axes[j].set_visible(False)
    
    plt.suptitle("SHAP Dependence Plots — How Features Affect Price",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved outputs/shap_dependence.png")


def plot_feature_importance_simple(model, feature_names, model_name):
    """Simple feature importance bar chart (non-SHAP)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        print("⚠️  Model does not support feature_importances_ or coef_")
        return
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:15]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_names)))
    bars = ax.barh(range(len(top_names)), top_vals[::-1], color=colors, edgecolor="white")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=11)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top 15 Feature Importances — {model_name}",
                 fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved outputs/feature_importance.png")


def generate_shap_data_for_web(shap_values, X_test, feature_names):
    """Save SHAP values so the Streamlit app can use them without recomputing."""
    shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    shap_df.to_csv("models/shap_values_test.csv", index=False)
    
    # Save base value
    base_value = shap_values.base_values[0] if hasattr(shap_values.base_values, '__len__') else shap_values.base_values
    joblib.dump(base_value, "models/shap_base_value.pkl")
    
    print("💾 Saved SHAP values and base value for web app")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Vancouver Housing — Model Explainability")
    print("=" * 60)
    
    model, model_name, X_train, X_test, y_train, y_test, feature_names = load_model_and_data()
    
    # Use test set for SHAP (or a subsample if too large)
    X_explain = X_test if len(X_test) <= 500 else X_test.sample(500, random_state=42)
    
    # Compute SHAP values
    shap_values, explainer = compute_shap_values(model, X_explain, model_name)
    
    # Generate all explainability plots
    print("\n📊 Generating explainability visualizations...")
    plot_shap_summary(shap_values, X_explain, feature_names)
    plot_shap_bar(shap_values, feature_names)
    plot_shap_waterfall(shap_values, idx=0)
    plot_shap_dependence(shap_values, X_explain, feature_names)
    plot_feature_importance_simple(model, feature_names, model_name)
    
    # Save SHAP data for web app  
    generate_shap_data_for_web(shap_values, X_explain, feature_names)
    
    # Save explainer for web app
    joblib.dump(explainer, "models/shap_explainer.pkl")
    print("💾 Saved SHAP explainer to models/shap_explainer.pkl")
    
    print("\n✅ All explainability outputs generated!")
    print("   📁 Check the outputs/ directory for visualizations")


if __name__ == "__main__":
    main()
