"""
Vancouver Housing Market - Synthetic Data Generator
====================================================
Generates realistic Vancouver housing data with proper pricing 
based on real neighborhood characteristics and market dynamics.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ---------------------------------------------------------------------------
# Vancouver Neighborhood Definitions
# ---------------------------------------------------------------------------
# Each neighborhood has a base price multiplier reflecting real market trends.

NEIGHBORHOODS = {
    "Kitsilano":        {"multiplier": 1.25, "lat": 49.2680, "lon": -123.1680},
    "Mount Pleasant":   {"multiplier": 1.10, "lat": 49.2620, "lon": -123.1010},
    "Kerrisdale":       {"multiplier": 1.40, "lat": 49.2330, "lon": -123.1560},
    "Dunbar":           {"multiplier": 1.35, "lat": 49.2490, "lon": -123.1870},
    "West End":         {"multiplier": 1.15, "lat": 49.2860, "lon": -123.1340},
    "Coal Harbour":     {"multiplier": 1.50, "lat": 49.2910, "lon": -123.1230},
    "Yaletown":         {"multiplier": 1.30, "lat": 49.2740, "lon": -123.1210},
    "Gastown":          {"multiplier": 1.05, "lat": 49.2830, "lon": -123.1080},
    "Fairview":         {"multiplier": 1.15, "lat": 49.2640, "lon": -123.1290},
    "Hastings-Sunrise": {"multiplier": 0.85, "lat": 49.2810, "lon": -123.0440},
    "Renfrew":          {"multiplier": 0.80, "lat": 49.2520, "lon": -123.0430},
    "Riley Park":       {"multiplier": 1.00, "lat": 49.2440, "lon": -123.1020},
    "Marpole":          {"multiplier": 0.90, "lat": 49.2110, "lon": -123.1290},
    "Oakridge":         {"multiplier": 1.20, "lat": 49.2260, "lon": -123.1180},
    "Shaughnessy":      {"multiplier": 1.65, "lat": 49.2430, "lon": -123.1370},
    "Point Grey":       {"multiplier": 1.55, "lat": 49.2660, "lon": -123.2000},
    "Grandview":        {"multiplier": 0.95, "lat": 49.2750, "lon": -123.0700},
    "Strathcona":       {"multiplier": 0.75, "lat": 49.2780, "lon": -123.0870},
    "Sunset":           {"multiplier": 0.85, "lat": 49.2230, "lon": -123.0920},
    "Killarney":        {"multiplier": 0.80, "lat": 49.2250, "lon": -123.0450},
}

PROPERTY_TYPES = ["House", "Condo", "Townhouse"]
PROPERTY_TYPE_WEIGHTS = [0.30, 0.45, 0.25]  # Condos most common in Vancouver

# Property type base prices and feature ranges
PROPERTY_CONFIG = {
    "House": {
        "base_price": 1_600_000,
        "beds": (3, 6),
        "baths": (2, 4),
        "sqft": (1800, 4500),
        "lot_size": (3000, 8000),
    },
    "Condo": {
        "base_price": 650_000,
        "beds": (1, 3),
        "baths": (1, 2),
        "sqft": (450, 1400),
        "lot_size": (0, 0),
    },
    "Townhouse": {
        "base_price": 950_000,
        "beds": (2, 4),
        "baths": (1, 3),
        "sqft": (1000, 2200),
        "lot_size": (1200, 3000),
    },
}


def generate_record(idx: int) -> dict:
    """Generate a single housing record with realistic Vancouver pricing."""

    # Pick neighborhood and property type
    neighborhood = np.random.choice(list(NEIGHBORHOODS.keys()))
    hood = NEIGHBORHOODS[neighborhood]

    property_type = np.random.choice(PROPERTY_TYPES, p=PROPERTY_TYPE_WEIGHTS)
    cfg = PROPERTY_CONFIG[property_type]

    # Generate features
    bedrooms  = np.random.randint(*cfg["beds"])
    bathrooms = np.random.randint(*cfg["baths"])
    sqft      = int(np.random.normal((cfg["sqft"][0] + cfg["sqft"][1]) / 2,
                                      (cfg["sqft"][1] - cfg["sqft"][0]) / 4))
    sqft      = max(cfg["sqft"][0], min(cfg["sqft"][1], sqft))
    lot_size  = int(np.random.uniform(*cfg["lot_size"])) if cfg["lot_size"][1] > 0 else 0

    year_built = np.random.randint(1920, 2025)
    age = 2025 - year_built

    has_garage    = int(np.random.random() < (0.7 if property_type == "House" else 0.3))
    has_basement  = int(np.random.random() < (0.6 if property_type == "House" else 0.05))
    has_renovation = int(np.random.random() < 0.35)

    # Distance to downtown (km) — loosely correlated with neighborhood
    dist_downtown = max(0.5, np.random.normal(
        abs(hood["lat"] - 49.2827) * 111 + abs(hood["lon"] + 123.1207) * 85, 1.0
    ))
    dist_downtown = round(dist_downtown, 1)

    walk_score = int(np.clip(np.random.normal(
        90 - dist_downtown * 5, 8
    ), 30, 100))

    # --------------- Price Calculation ---------------
    # Base price from property type
    price = cfg["base_price"]

    # Neighborhood multiplier (biggest driver)
    price *= hood["multiplier"]

    # Square footage premium
    mid_sqft = (cfg["sqft"][0] + cfg["sqft"][1]) / 2
    price *= 1 + 0.3 * (sqft - mid_sqft) / mid_sqft

    # Bedrooms & bathrooms
    price += (bedrooms - 2) * 45_000
    price += (bathrooms - 1) * 30_000

    # Age discount (older = cheaper, unless character home)
    if age > 50:
        price *= 0.90  # Character home mild discount
    elif age > 20:
        price *= 0.95
    # New builds get a premium
    if age < 5:
        price *= 1.08

    # Features
    if has_garage:
        price += 35_000
    if has_basement:
        price += 50_000
    if has_renovation:
        price *= 1.05

    # Lot size premium (houses)
    if lot_size > 0:
        price += (lot_size - 3500) * 25

    # Walk score bonus
    price *= 1 + (walk_score - 70) * 0.001

    # Distance to downtown discount
    price *= 1 - (dist_downtown - 3) * 0.01

    # Random market noise (±8%)
    noise = np.random.normal(1.0, 0.08)
    price *= noise

    # Round to nearest $1,000
    price = int(round(price / 1000) * 1000)
    price = max(300_000, price)  # Floor

    return {
        "neighborhood": neighborhood,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft": sqft,
        "lot_size": lot_size,
        "year_built": year_built,
        "age": age,
        "has_garage": has_garage,
        "has_basement": has_basement,
        "has_renovation": has_renovation,
        "distance_to_downtown_km": dist_downtown,
        "walk_score": walk_score,
        "latitude": round(hood["lat"] + np.random.normal(0, 0.005), 4),
        "longitude": round(hood["lon"] + np.random.normal(0, 0.005), 4),
        "price": price,
    }


def generate_dataset(n: int = 2000) -> pd.DataFrame:
    """Generate the full Vancouver housing dataset."""
    records = [generate_record(i) for i in range(n)]
    df = pd.DataFrame(records)
    return df


def main():
    print("🏠 Generating Vancouver housing dataset...")
    df = generate_dataset(2000)

    # Save to data/
    os.makedirs("data", exist_ok=True)
    output_path = "data/vancouver_housing.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Generated {len(df)} records → {output_path}")
    print(f"\n📊 Dataset Summary:")
    print(f"   Columns:        {list(df.columns)}")
    print(f"   Price range:    ${df['price'].min():,.0f} – ${df['price'].max():,.0f}")
    print(f"   Median price:   ${df['price'].median():,.0f}")
    print(f"   Neighborhoods:  {df['neighborhood'].nunique()}")
    print(f"   Property types: {df['property_type'].value_counts().to_dict()}")
    print(f"\n📈 Feature Stats:")
    print(df.describe().round(1).to_string())


if __name__ == "__main__":
    main()
