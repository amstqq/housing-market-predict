"""
Vancouver Housing Price Predictor — Streamlit Web App
=====================================================
Interactive web app that predicts housing prices and explains
each prediction using SHAP visualizations.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vancouver Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Premium Look
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: #a5b4fc;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Price result card */
    .price-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
        margin: 1.5rem 0;
        animation: fadeInUp 0.6s ease-out;
    }
    .price-card h2 {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .price-card .price {
        color: #ffffff;
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .price-card .range {
        color: rgba(255,255,255,0.7);
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        flex: 1;
        text-align: center;
    }
    .metric-card .label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        color: #e2e8f0;
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c7d2fe;
    }

    /* Animation */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Info boxes */
    .info-box {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid #6366f1;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #c7d2fe;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Constants (must match data_generator.py)
# ---------------------------------------------------------------------------
NEIGHBORHOODS = [
    "Coal Harbour",
    "Shaughnessy",
    "Point Grey",
    "Kerrisdale",
    "Dunbar",
    "Yaletown",
    "Kitsilano",
    "Oakridge",
    "West End",
    "Fairview",
    "Mount Pleasant",
    "Gastown",
    "Riley Park",
    "Grandview",
    "Marpole",
    "Hastings-Sunrise",
    "Sunset",
    "Renfrew",
    "Killarney",
    "Strathcona",
]

NEIGHBORHOOD_COORDS = {
    "Kitsilano": (49.2680, -123.1680),
    "Mount Pleasant": (49.2620, -123.1010),
    "Kerrisdale": (49.2330, -123.1560),
    "Dunbar": (49.2490, -123.1870),
    "West End": (49.2860, -123.1340),
    "Coal Harbour": (49.2910, -123.1230),
    "Yaletown": (49.2740, -123.1210),
    "Gastown": (49.2830, -123.1080),
    "Fairview": (49.2640, -123.1290),
    "Hastings-Sunrise": (49.2810, -123.0440),
    "Renfrew": (49.2520, -123.0430),
    "Riley Park": (49.2440, -123.1020),
    "Marpole": (49.2110, -123.1290),
    "Oakridge": (49.2260, -123.1180),
    "Shaughnessy": (49.2430, -123.1370),
    "Point Grey": (49.2660, -123.2000),
    "Grandview": (49.2750, -123.0700),
    "Strathcona": (49.2780, -123.0870),
    "Sunset": (49.2230, -123.0920),
    "Killarney": (49.2250, -123.0450),
}

PROPERTY_TYPES = ["House", "Condo", "Townhouse"]

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR) == "web"
    else SCRIPT_DIR
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


# ---------------------------------------------------------------------------
# Load Model & Artifacts (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load the trained model, scaler, and feature metadata."""
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return model, model_name, scaler, feature_names


@st.cache_data
def load_dataset():
    """Load the training dataset for comparisons."""
    return pd.read_csv(os.path.join(DATA_DIR, "vancouver_housing.csv"))


# ---------------------------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------------------------
def build_feature_vector(
    inputs: dict, scaler, feature_names: list
) -> pd.DataFrame:
    """
    Convert raw user inputs into the model's expected feature vector.
    Mirrors the preprocessing.py pipeline exactly.
    """
    # Start with numeric features
    row = {
        "bedrooms": inputs["bedrooms"],
        "bathrooms": inputs["bathrooms"],
        "sqft": inputs["sqft"],
        "lot_size": inputs["lot_size"],
        "year_built": inputs["year_built"],
        "age": 2025 - inputs["year_built"],
        "has_garage": int(inputs["has_garage"]),
        "has_basement": int(inputs["has_basement"]),
        "has_renovation": int(inputs["has_renovation"]),
        "distance_to_downtown_km": inputs["distance_to_downtown_km"],
        "walk_score": inputs["walk_score"],
    }

    # One-hot encode neighborhood
    for n in NEIGHBORHOODS:
        col = f"neighborhood_{n}"
        row[col] = 1 if inputs["neighborhood"] == n else 0

    # One-hot encode property type
    for pt in PROPERTY_TYPES:
        col = f"property_type_{pt}"
        row[col] = 1 if inputs["property_type"] == pt else 0

    # Build DataFrame with correct column order
    X = pd.DataFrame([row])

    # Ensure all expected columns exist (in case of ordering differences)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    # Scale numeric features (same as preprocessing.py)
    numeric_cols = [
        "bedrooms",
        "bathrooms",
        "sqft",
        "lot_size",
        "year_built",
        "age",
        "distance_to_downtown_km",
        "walk_score",
    ]
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    X[numeric_cols] = scaler.transform(X[numeric_cols])

    return X


def compute_shap_for_prediction(model, X, model_name):
    """Compute SHAP values for a single prediction."""
    if "XGBoost" in model_name or "Random Forest" in model_name:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X)

    shap_values = explainer(X)
    return shap_values


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------
def create_shap_waterfall_plotly(
    shap_values, feature_names, prediction, base_value
):
    """Create an interactive SHAP waterfall chart using Plotly."""
    sv = shap_values.values[0]

    # Get top features by absolute SHAP value
    abs_sv = np.abs(sv)
    top_n = 12
    top_idx = np.argsort(abs_sv)[::-1][:top_n]

    names = [
        feature_names[i]
        .replace("neighborhood_", "📍 ")
        .replace("property_type_", "🏠 ")
        for i in top_idx
    ]
    values = [sv[i] for i in top_idx]
    colors = ["#ef4444" if v < 0 else "#22c55e" for v in values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=names[::-1],
            x=values[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"${v:+,.0f}" for v in values[::-1]],
            textposition="outside",
            textfont=dict(size=12, color="#e2e8f0"),
        )
    )

    fig.update_layout(
        title=dict(
            text="How Each Feature Affects This Prediction",
            font=dict(size=18, color="#e2e8f0"),
        ),
        xaxis_title="Impact on Price ($)",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=12),
        height=500,
        margin=dict(l=10, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.1)",
            zeroline=True,
            zerolinecolor="rgba(148,163,184,0.3)",
        ),
    )
    return fig


def create_neighborhood_comparison(df, selected_neighborhood, predicted_price):
    """Create a neighborhood price comparison chart."""
    med_prices = (
        df.groupby("neighborhood")["price"].median().sort_values(ascending=True)
    )

    colors = [
        "#6366f1" if n != selected_neighborhood else "#f59e0b"
        for n in med_prices.index
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=med_prices.values,
            y=med_prices.index,
            orientation="h",
            marker_color=colors,
            text=[f"${v:,.0f}" for v in med_prices.values],
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
        )
    )

    # Add predicted price marker
    fig.add_vline(
        x=predicted_price,
        line_dash="dash",
        line_color="#f43f5e",
        annotation_text=f"Your Property: ${predicted_price:,.0f}",
        annotation_font_color="#f43f5e",
    )

    fig.update_layout(
        title=dict(
            text="Median Price by Neighborhood",
            font=dict(size=18, color="#e2e8f0"),
        ),
        xaxis_title="Median Price ($)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=11),
        height=600,
        margin=dict(l=10, r=80, t=50, b=40),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
    )
    return fig


def create_price_distribution(df, predicted_price, property_type):
    """Show where the predicted price falls in the distribution."""
    subset = df[df["property_type"] == property_type]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=subset["price"],
            nbinsx=40,
            marker_color="#6366f1",
            opacity=0.7,
            name=f"{property_type} prices",
        )
    )
    fig.add_vline(
        x=predicted_price,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=3,
        annotation_text=f"Your Property: ${predicted_price:,.0f}",
        annotation_font_color="#f59e0b",
        annotation_font_size=13,
    )
    fig.add_vline(
        x=subset["price"].median(),
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text=f"Median: ${subset['price'].median():,.0f}",
        annotation_font_color="#94a3b8",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(
            text=f"Where Your Price Falls Among {property_type}s",
            font=dict(size=18, color="#e2e8f0"),
        ),
        xaxis_title="Price ($)",
        yaxis_title="Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        height=400,
        showlegend=False,
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
    )
    return fig


# ---------------------------------------------------------------------------
# App Layout
# ---------------------------------------------------------------------------
def main():
    # Load artifacts
    try:
        model, model_name, scaler, feature_names = load_artifacts()
        df = load_dataset()
    except Exception as e:
        st.error(
            f"⚠️ Could not load model artifacts. Please run the training pipeline first.\n\n"
            f"```\npython src/data_generator.py\npython src/train_model.py\npython src/explainability.py\n```\n\n"
            f"Error: {e}"
        )
        return

    # ---- Hero Header ----
    st.markdown(
        """
    <div class="hero-header">
        <h1>🏠 Vancouver Housing Price Predictor</h1>
        <p>AI-powered home valuations with transparent, explainable predictions</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ---- Sidebar: Property Input Form ----
    with st.sidebar:
        st.markdown("## 🏡 Property Details")
        st.markdown("---")

        neighborhood = st.selectbox(
            "📍 Neighborhood",
            NEIGHBORHOODS,
            index=NEIGHBORHOODS.index("Kitsilano"),
        )

        property_type = st.selectbox("🏠 Property Type", PROPERTY_TYPES)

        st.markdown("---")
        st.markdown("### 📐 Size & Layout")

        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 4, 2)
        sqft = st.slider("Square Footage", 400, 5000, 1500, step=50)
        lot_size = st.slider(
            "Lot Size (sq ft)",
            0,
            8000,
            3000 if property_type == "House" else 0,
            step=100,
        )

        st.markdown("---")
        st.markdown("### 🏗️ Property Details")

        year_built = st.slider("Year Built", 1920, 2025, 2000)
        has_garage = st.checkbox("Has Garage", value=True)
        has_basement = st.checkbox(
            "Has Basement", value=property_type == "House"
        )
        has_renovation = st.checkbox("Renovated", value=False)

        st.markdown("---")
        st.markdown("### 📍 Location")

        # Auto-estimate distance and walk score based on neighborhood
        coords = NEIGHBORHOOD_COORDS.get(neighborhood, (49.28, -123.12))
        default_dist = round(
            abs(coords[0] - 49.2827) * 111 + abs(coords[1] + 123.1207) * 85, 1
        )
        default_dist = max(0.5, min(15.0, default_dist))

        distance_to_downtown = st.slider(
            "Distance to Downtown (km)",
            0.5,
            15.0,
            float(default_dist),
            step=0.5,
        )
        walk_score = st.slider(
            "Walk Score",
            30,
            100,
            int(max(30, min(100, 90 - distance_to_downtown * 5))),
        )

        st.markdown("---")
        predict_btn = st.button(
            "🔮 **Predict Price**", use_container_width=True, type="primary"
        )

    # ---- Main Content ----
    if predict_btn:
        # Build inputs
        inputs = {
            "neighborhood": neighborhood,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft": sqft,
            "lot_size": lot_size,
            "year_built": year_built,
            "has_garage": has_garage,
            "has_basement": has_basement,
            "has_renovation": has_renovation,
            "distance_to_downtown_km": distance_to_downtown,
            "walk_score": walk_score,
        }

        # Predict
        X = build_feature_vector(inputs, scaler, feature_names)
        predicted_price = model.predict(X)[0]
        predicted_price = max(
            300_000, int(round(predicted_price / 1000) * 1000)
        )

        # Confidence range (±8% based on model RMSE)
        low = int(predicted_price * 0.92)
        high = int(predicted_price * 1.08)

        # Compute SHAP
        shap_values = compute_shap_for_prediction(model, X, model_name)
        base_value = shap_values.base_values[0]

        # ---- Price Result ----
        st.markdown(
            f"""
        <div class="price-card">
            <h2>Estimated Property Value</h2>
            <div class="price">${predicted_price:,.0f}</div>
            <div class="range">Confidence Range: ${low:,.0f} – ${high:,.0f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # ---- Key Metrics ----
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📍 Neighborhood", neighborhood)
        with col2:
            st.metric("🏠 Type", property_type)
        with col3:
            st.metric("📐 Size", f"{sqft:,} sqft")
        with col4:
            # Price per sqft
            ppsf = predicted_price / sqft if sqft > 0 else 0
            st.metric("💰 Price/sqft", f"${ppsf:,.0f}")

        st.markdown("---")

        # ---- SHAP Explanation ----
        st.markdown(
            '<div class="section-header">🔍 Why This Price?</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="info-box">
            The chart below shows how each feature <strong>pushed the price up or down</strong> 
            from the average. Green bars increase the price, red bars decrease it. 
            This uses <strong>SHAP (SHapley Additive exPlanations)</strong>, a game-theory 
            based approach to explain individual predictions.
        </div>
        """,
            unsafe_allow_html=True,
        )

        fig_shap = create_shap_waterfall_plotly(
            shap_values, feature_names, predicted_price, base_value
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # ---- Comparisons ----
        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(
                '<div class="section-header">📊 Neighborhood Comparison</div>',
                unsafe_allow_html=True,
            )
            fig_nbhd = create_neighborhood_comparison(
                df, neighborhood, predicted_price
            )
            st.plotly_chart(fig_nbhd, use_container_width=True)

        with col_right:
            st.markdown(
                '<div class="section-header">📈 Price Distribution</div>',
                unsafe_allow_html=True,
            )
            fig_dist = create_price_distribution(
                df, predicted_price, property_type
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # ---- Comparable Properties ----
        st.markdown("---")
        st.markdown(
            '<div class="section-header">🏘️ Similar Properties in Dataset</div>',
            unsafe_allow_html=True,
        )

        similar = df[
            (df["neighborhood"] == neighborhood)
            & (df["property_type"] == property_type)
        ].copy()

        if len(similar) > 0:
            similar["price_diff"] = abs(similar["price"] - predicted_price)
            similar = similar.sort_values("price_diff").head(5)
            display_cols = [
                "neighborhood",
                "property_type",
                "bedrooms",
                "bathrooms",
                "sqft",
                "year_built",
                "price",
            ]
            st.dataframe(
                similar[display_cols].style.format(
                    {"price": "${:,.0f}", "sqft": "{:,}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No similar properties found in the dataset for this combination."
            )

    else:
        # ---- Welcome State ----
        st.markdown(
            """
        <div class="info-box">
            👋 <strong>Welcome!</strong> Use the sidebar to enter your property details, 
            then click <strong>"Predict Price"</strong> to get an AI-powered valuation 
            with a full explanation of what drives the price.
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Show some dataset stats
        try:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏘️ Properties Analyzed", f"{len(df):,}")
            with col2:
                st.metric("📍 Neighborhoods", f"{df['neighborhood'].nunique()}")
            with col3:
                st.metric("💰 Median Price", f"${df['price'].median():,.0f}")

            st.markdown("---")
            st.markdown(
                '<div class="section-header">📊 Market Overview</div>',
                unsafe_allow_html=True,
            )

            # Neighborhood median price chart
            med_prices = (
                df.groupby("neighborhood")["price"]
                .median()
                .sort_values(ascending=True)
            )
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=med_prices.values,
                    y=med_prices.index,
                    orientation="h",
                    marker=dict(
                        color=med_prices.values,
                        colorscale="Viridis",
                    ),
                    text=[f"${v:,.0f}" for v in med_prices.values],
                    textposition="outside",
                    textfont=dict(size=10, color="#94a3b8"),
                )
            )
            fig.update_layout(
                title=dict(
                    text="Median Price by Neighborhood",
                    font=dict(size=18, color="#e2e8f0"),
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                height=600,
                margin=dict(l=10, r=80, t=50, b=40),
                xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    # ---- Footer ----
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
        f"Powered by {model_name if 'model_name' in dir() else 'XGBoost'} · "
        "Explainability via SHAP · Built with Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
