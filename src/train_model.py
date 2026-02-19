"""
Vancouver Housing Market - Model Training & Evaluation
=======================================================
Trains Linear Regression, Random Forest, and XGBoost models.
Evaluates performance and generates comparison visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

from preprocessing import load_data, preprocess, split_data

# Output directories
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# Plotting Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
COLORS = {"Linear Regression": "#6366f1", "Random Forest": "#06b6d4", "XGBoost": "#f43f5e"}


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """Train all models and return results."""
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0
        ),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        metrics = {
            "model": model,
            "y_pred_test": y_pred_test,
            "y_pred_train": y_pred_train,
            "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "mae_test": mean_absolute_error(y_test, y_pred_test),
            "r2_train": r2_score(y_train, y_pred_train),
            "r2_test": r2_score(y_test, y_pred_test),
        }
        results[name] = metrics
        
        print(f"   R² (train): {metrics['r2_train']:.4f}")
        print(f"   R² (test):  {metrics['r2_test']:.4f}")
        print(f"   RMSE:       ${metrics['rmse_test']:,.0f}")
        print(f"   MAE:        ${metrics['mae_test']:,.0f}")
    
    return results


def save_best_model(results):
    """Save the model with the best test R²."""
    best_name = max(results, key=lambda k: results[k]["r2_test"])
    best_model = results[best_name]["model"]
    
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(best_name, "models/best_model_name.pkl")
    print(f"\n🏆 Best model: {best_name} (R² = {results[best_name]['r2_test']:.4f})")
    print(f"💾 Saved to models/best_model.pkl")
    
    return best_name, best_model


def plot_model_comparison(results):
    """Bar chart comparing model performance metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    model_names = list(results.keys())
    colors = [COLORS[n] for n in model_names]
    
    # R² Score
    r2_scores = [results[n]["r2_test"] for n in model_names]
    axes[0].bar(model_names, r2_scores, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("R² Score (Higher = Better)", fontsize=13, fontweight="bold")
    axes[0].set_ylim(min(r2_scores) - 0.05, 1.0)
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold", fontsize=11)
    
    # RMSE
    rmse_scores = [results[n]["rmse_test"] for n in model_names]
    axes[1].bar(model_names, rmse_scores, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("RMSE (Lower = Better)", fontsize=13, fontweight="bold")
    for i, v in enumerate(rmse_scores):
        axes[1].text(i, v + 500, f"${v:,.0f}", ha="center", fontweight="bold", fontsize=11)
    
    # MAE
    mae_scores = [results[n]["mae_test"] for n in model_names]
    axes[2].bar(model_names, mae_scores, color=colors, edgecolor="white", linewidth=1.5)
    axes[2].set_title("MAE (Lower = Better)", fontsize=13, fontweight="bold")
    for i, v in enumerate(mae_scores):
        axes[2].text(i, v + 500, f"${v:,.0f}", ha="center", fontweight="bold", fontsize=11)
    
    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved outputs/model_comparison.png")


def plot_actual_vs_predicted(results, y_test):
    """Scatter plots of actual vs predicted for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["y_pred_test"]
        color = COLORS[name]
        
        ax.scatter(y_test, y_pred, alpha=0.4, s=20, color=color, edgecolors="none")
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.6, linewidth=1.5,
                label="Perfect Prediction")
        
        ax.set_xlabel("Actual Price ($)", fontsize=11)
        ax.set_ylabel("Predicted Price ($)", fontsize=11)
        ax.set_title(f"{name}\nR² = {res['r2_test']:.4f}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        
        # Format tick labels
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    
    plt.suptitle("Actual vs. Predicted Housing Prices", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved outputs/actual_vs_predicted.png")


def plot_residuals(results, y_test):
    """Residual distribution plots for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (name, res) in zip(axes, results.items()):
        residuals = y_test.values - res["y_pred_test"]
        color = COLORS[name]
        
        ax.hist(residuals, bins=40, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Residual ($)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{name}\nMean Residual: ${np.mean(residuals):,.0f}", 
                     fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    
    plt.suptitle("Residual Distributions", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved outputs/residuals.png")


def plot_price_by_neighborhood(df):
    """Box plot of prices by neighborhood."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Order by median price
    order = df.groupby("neighborhood")["price"].median().sort_values(ascending=False).index
    
    sns.boxplot(data=df, x="neighborhood", y="price", order=order, ax=ax,
                palette="viridis", fliersize=2)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Neighborhood", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.set_title("Vancouver Housing Prices by Neighborhood", fontsize=16, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    
    plt.tight_layout()
    plt.savefig("outputs/price_by_neighborhood.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved outputs/price_by_neighborhood.png")


def plot_correlation_heatmap(df):
    """Correlation heatmap of numeric features."""
    numeric_cols = ["bedrooms", "bathrooms", "sqft", "lot_size", "age",
                    "distance_to_downtown_km", "walk_score", "has_garage",
                    "has_basement", "has_renovation", "price"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    corr = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved outputs/correlation_heatmap.png")


def plot_price_distribution(df):
    """Distribution of housing prices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall distribution
    axes[0].hist(df["price"], bins=50, color="#6366f1", alpha=0.7, edgecolor="white")
    axes[0].axvline(df["price"].median(), color="#f43f5e", linestyle="--", linewidth=2,
                    label=f'Median: ${df["price"].median():,.0f}')
    axes[0].set_xlabel("Price ($)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Overall Price Distribution", fontsize=13, fontweight="bold")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    axes[0].legend(fontsize=10)
    
    # By property type
    for ptype in df["property_type"].unique():
        subset = df[df["property_type"] == ptype]
        axes[1].hist(subset["price"], bins=30, alpha=0.5, label=ptype, edgecolor="white")
    axes[1].set_xlabel("Price ($)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Price Distribution by Property Type", fontsize=13, fontweight="bold")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    axes[1].legend(fontsize=10)
    
    plt.suptitle("Vancouver Housing Price Distributions", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/price_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 Saved outputs/price_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Vancouver Housing Price Prediction — Model Training")
    print("=" * 60)
    
    # Load and preprocess data
    df = load_data()
    
    # Generate EDA visualizations first
    print("\n📊 Generating EDA visualizations...")
    plot_price_distribution(df)
    plot_price_by_neighborhood(df)
    plot_correlation_heatmap(df)
    
    # Preprocess for modeling
    X, y, feature_names = preprocess(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train models
    print("\n" + "=" * 60)
    print("  Training Models")
    print("=" * 60)
    results = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)
    
    # Save best model
    best_name, best_model = save_best_model(results)
    
    # Generate comparison visualizations
    print("\n📊 Generating model comparison visualizations...")
    plot_model_comparison(results)
    plot_actual_vs_predicted(results, y_test)
    plot_residuals(results, y_test)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("  📋 Final Results Summary")
    print("=" * 60)
    summary = pd.DataFrame({
        name: {
            "R² (test)": f"{res['r2_test']:.4f}",
            "RMSE": f"${res['rmse_test']:,.0f}",
            "MAE": f"${res['mae_test']:,.0f}",
            "R² (train)": f"{res['r2_train']:.4f}",
        }
        for name, res in results.items()
    }).T
    print(summary.to_string())
    print(f"\n🏆 Best Model: {best_name}")
    print("\n✅ All visualizations saved to outputs/")
    
    return results, X_test, y_test, feature_names


if __name__ == "__main__":
    main()
