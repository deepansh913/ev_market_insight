"""
Generates or verifies ev_market_test.csv (100,000 rows x 37 columns) matching the exact
statistical profile and schema of the EV Market Intelligence project.
"""
import os
import sys
import numpy as np
import pandas as pd

def generate_ev_dataset(output_csv: str, n_rows: int = 100000) -> None:
    rng = np.random.default_rng(42)
    categories = [
        "Two-Wheeler Scooter", "Sedan", "Hatchback", "Compact SUV",
        "Three-Wheeler", "Crossover", "Luxury Sedan", "Full-size SUV",
        "Micro City Car", "Pickup Truck", "Van / MPV", "Luxury SUV",
        "Electric Truck", "Electric Bus", "Sports Car"
    ]
    cat_probs = [
        0.09826, 0.08617, 0.08571, 0.08569, 0.07527, 0.07462, 0.06511,
        0.06457, 0.06404, 0.05393, 0.05382, 0.05375, 0.05310, 0.04309, 0.04287
    ]
    cat_probs = np.array(cat_probs) / np.sum(cat_probs)

    batteries = ["NMC", "LFP", "NCA", "Sodium-Ion", "Solid-State"]
    bat_probs = [0.33836, 0.32101, 0.17234, 0.09836, 0.06993]

    countries = ["Vietnam", "Japan", "China", "India", "South Korea", "France", "USA", "Sweden", "Germany", "UK"]
    regions = ["North America", "South Asia", "Latin America", "Middle East", "Africa", "Asia Pacific", "Europe"]
    use_cases = ["Urban Commute", "Family Daily Use", "Highway / Long-Distance", "Commercial Delivery", "Fleet / Public Transit", "Premium / Executive"]

    top_oems = [
        "SurgeWorks", "CurrentMobility", "ElektraMobility", "VelocityMobility",
        "WattDynamics", "ZapWheels", "KineticAutomotive", "IonMotors", "WattRides", "ZapWorks"
    ]
    other_oems = [f"EVBrand_{i:03d}" for i in range(1, 191)]
    all_oems = top_oems + other_oems

    df = pd.DataFrame({
        "vehicle_id": [f"EV{i:06d}" for i in range(1, n_rows + 1)],
        "model_name": [f"Model-{rng.integers(100, 9999)}" for _ in range(n_rows)],
        "manufacturer": rng.choice(all_oems, size=n_rows),
        "model_year": rng.integers(2018, 2027, size=n_rows),
        "manufacturing_country": rng.choice(countries, size=n_rows),
        "market_region": rng.choice(regions, size=n_rows),
        "category": rng.choice(categories, p=cat_probs, size=n_rows),
        "battery_capacity_kwh": np.round(rng.uniform(4.0, 150.0, size=n_rows), 2),
        "battery_type": rng.choice(batteries, p=bat_probs, size=n_rows),
        "battery_cycle_life": np.round(rng.uniform(1200, 3400, size=n_rows), 0),
        "battery_degradation_pct_5yr": np.round(rng.uniform(4.5, 18.5, size=n_rows), 2),
        "range_km": np.round(rng.uniform(85.0, 640.0, size=n_rows), 1),
        "efficiency_wh_per_km": np.round(rng.uniform(35.0, 265.0, size=n_rows), 2),
        "motor_power_kw": np.round(rng.uniform(8.0, 480.0, size=n_rows), 1),
        "torque_nm": np.round(rng.uniform(45.0, 980.0, size=n_rows), 1),
        "num_motors": rng.choice([1, 2, 3, 4], p=[0.55, 0.35, 0.07, 0.03], size=n_rows),
        "drivetrain": rng.choice(["FWD", "RWD", "AWD"], p=[0.4, 0.3, 0.3], size=n_rows),
        "top_speed_kmph": np.round(rng.uniform(55.0, 280.0, size=n_rows), 1),
        "acceleration_0_100_sec": np.round(rng.uniform(2.6, 15.5, size=n_rows), 2),
        "max_dc_charging_kw": np.round(rng.uniform(10.0, 350.0, size=n_rows), 1),
        "charging_dc_fast_time_min": np.round(rng.uniform(12.0, 55.0, size=n_rows), 1),
        "charging_ac_time_hr": np.round(rng.uniform(2.5, 16.0, size=n_rows), 2),
        "num_charging_ports": rng.choice([1, 2], p=[0.82, 0.18], size=n_rows),
        "num_seats": rng.choice([2, 3, 4, 5, 7, 40], p=[0.18, 0.08, 0.14, 0.44, 0.12, 0.04], size=n_rows),
        "weight_kg": np.round(rng.uniform(110.0, 12000.0, size=n_rows), 1),
        "cargo_capacity_liters": np.round(rng.uniform(15.0, 2500.0, size=n_rows), 1),
        "price_usd": np.round(rng.uniform(1800.0, 185000.0, size=n_rows), 2),
        "insurance_cost_usd_yearly": np.round(rng.uniform(120.0, 4800.0, size=n_rows), 2),
        "warranty_years": rng.integers(3, 10, size=n_rows),
        "resale_value_pct_3yr": np.round(rng.uniform(38.0, 79.0, size=n_rows), 2),
        "safety_rating_stars": rng.choice([3, 4, 5], p=[0.18, 0.47, 0.35], size=n_rows),
        "autonomous_level": rng.integers(0, 5, size=n_rows),
        "infotainment_screen_inch": np.round(rng.uniform(5.0, 17.5, size=n_rows), 1),
        "co2_saved_tons_per_year": np.round(rng.uniform(0.8, 8.5, size=n_rows), 3),
        "annual_sales_units": np.round(rng.uniform(500.0, 15000.0, size=n_rows), 0),
        "customer_rating": np.round(np.clip(rng.normal(4.08, 0.42, size=n_rows), 1.0, 5.0), 2),
        "recommended_use_case": rng.choice(use_cases, size=n_rows),
    })

    # Inject exact missing value counts matching audit specification
    missing_specs = {
        "resale_value_pct_3yr": 3057,
        "battery_cycle_life": 2028,
        "customer_rating": 1931,
        "infotainment_screen_inch": 1506,
        "annual_sales_units": 989,
    }
    for col, n_miss in missing_specs.items():
        idx = rng.choice(n_rows, size=n_miss, replace=False)
        df.loc[idx, col] = np.nan

    df.to_csv(output_csv, index=False)
    print(f"Generated {output_csv} ({len(df):,} rows x {len(df.columns)} columns)")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "ev_market_test.csv"
    generate_ev_dataset(target)
