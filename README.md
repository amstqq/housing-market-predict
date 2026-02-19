# 🏠 Vancouver Housing Price Prediction

A machine learning project that predicts Vancouver housing prices and explains **why** each prediction is made using SHAP (SHapley Additive exPlanations).

## 📊 Project Overview

This project builds and compares three ML models on Vancouver housing data, with a focus on **explainability** — helping homeowners understand what drives their home's value.

### Models Compared

| Model             | R² Score   | RMSE         | MAE         |
| ----------------- | ---------- | ------------ | ----------- |
| Linear Regression | 0.9295     | $159,789     | $112,122    |
| Random Forest     | 0.9452     | $140,910     | $95,872     |
| **XGBoost** ⭐    | **0.9523** | **$131,451** | **$89,391** |

### Features Used

- **Property:** bedrooms, bathrooms, sqft, lot_size, year_built
- **Location:** neighborhood (20 Vancouver areas), distance_to_downtown, walk_score
- **Amenities:** has_garage, has_basement, has_renovation
- **Type:** House, Condo, Townhouse

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate the dataset
python src/data_generator.py

# Train models & generate EDA visualizations
python src/train_model.py

# Generate SHAP explainability plots
python src/explainability.py
```

## 📁 Project Structure

```
├── data/
│   └── vancouver_housing.csv     # 2,000 housing records
├── src/
│   ├── data_generator.py         # Synthetic data generation
│   ├── preprocessing.py          # Feature engineering & scaling
│   ├── train_model.py            # Model training & evaluation
│   └── explainability.py         # SHAP analysis & visualizations
├── models/                       # Trained models & preprocessors
├── outputs/                      # Generated visualizations
└── requirements.txt
```

## 📈 Key Visualizations

The project generates 11 publication-quality visualizations:

**EDA:**

- Price distribution (overall & by property type)
- Prices by neighborhood (box plot)
- Feature correlation heatmap

**Model Evaluation:**

- Model comparison (R², RMSE, MAE bar charts)
- Actual vs. Predicted scatter plots
- Residual distributions

**Explainability (SHAP):**

- SHAP bee swarm summary plot
- Feature importance bar chart
- Waterfall plot (individual prediction breakdown)
- Dependence plots (how features affect price)
- Simple feature importance ranking

## 🔍 Explainability

The SHAP waterfall plot shows exactly **how each feature contributes** to a specific prediction — for example, being in Shaughnessy adds ~$250K while being a condo subtracts ~$115K. This transparency is what makes the model trustworthy for real-world use.

## 🛠 Tech Stack

- **Python** — pandas, NumPy, scikit-learn, XGBoost
- **SHAP** — model explainability
- **Matplotlib / Seaborn** — static visualizations
- **Streamlit** — web application (coming soon)

## 📝 License

MIT License
