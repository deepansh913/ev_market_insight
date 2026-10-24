# âš¡ Electric Vehicle (EV) Market Intelligence Dashboard

A production-grade, interactive Business Intelligence application built with **Python, Streamlit, Pandas, NumPy, and Plotly** to transform Electric Vehicle (EV) market data (`ev_market_test.csv`, 100,000 records Ã— 37 columns) into executive, competitive, technological, geographic, and sustainability insights.

## ðŸŽ¯ Project Overview & Objectives

The global electric vehicle industry generates complex multi-dimensional data spanning vehicle specifications, battery chemistries, pricing, fast-charging capabilities, safety and autonomous levels, regional sales volumes, and environmental impact.

This dashboard is engineered for **business executives, automotive consultants, and sustainability researchers** to:

1. Monitor macro KPIs and market performance across **9 model years (2018â€“2026)** and **7 global market regions**.
2. Benchmark **200 EV manufacturers** side by side across sales volume, pricing, range, battery capacity, and customer ratings.
3. Evaluate **5 battery chemistries** (`NMC`, `LFP`, `NCA`, `Sodium-Ion`, `Solid-State`) across cycle life, 5-year degradation, fast-charging speed, and segment adoption.
4. Analyze all **15 vehicle categories** (from `Two-Wheeler Scooter` and `Micro City Car` to `Luxury SUV`, `Electric Truck`, and `Electric Bus`).
5. Investigate customer satisfaction drivers, geographic manufacturing specialization across **10 countries**, and COâ‚‚ reduction impact.

## ðŸ—‚ï¸ Project Structure

```
ev-market-intelligence-dashboard/
â”œâ”€â”€ app.py                  # Main Streamlit dashboard application (all 9 modules)
â”œâ”€â”€ generate_dataset.py     # Fallback generator for 37-column ev_market_test.csv
â”œâ”€â”€ requirements.txt        # Python dependencies for local/cloud deployment
â”œâ”€â”€ README.md               # Project documentation, setup guide, and data quality notes
â””â”€â”€ ev_market_test.csv      # Primary EV market dataset (100,000 rows Ã— 37 columns)
```

## ðŸ“Š Dataset Schema & Data Quality Summary

* **Total Records:** `100,000` vehicles
* **Total Columns:** `37` attributes (`vehicle_id`, `model_name`, `manufacturer`, `model_year`, `manufacturing_country`, `market_region`, `category`, `battery_capacity_kwh`, `battery_type`, `battery_cycle_life`, `battery_degradation_pct_5yr`, `range_km`, `efficiency_wh_per_km`, `motor_power_kw`, `torque_nm`, `num_motors`, `drivetrain`, `top_speed_kmph`, `acceleration_0_100_sec`, `max_dc_charging_kw`, `charging_dc_fast_time_min`, `charging_ac_time_hr`, `num_charging_ports`, `num_seats`, `weight_kg`, `cargo_capacity_liters`, `price_usd`, `insurance_cost_usd_yearly`, `warranty_years`, `resale_value_pct_3yr`, `safety_rating_stars`, `autonomous_level`, `infotainment_screen_inch`, `co2_saved_tons_per_year`, `annual_sales_units`, `customer_rating`, `recommended_use_case`)
* **Exact Duplicate Rows:** `0`
* **Duplicate `vehicle_id`s:** `0`

### Missing Values & Non-Destructive Cleaning Strategy

| Column | Missing Count | Missing % | Imputation Strategy |
| ----- | ----- | ----- | ----- |
| `resale_value_pct_3yr` | 3,057 | 3.06% | Group median by `category` + `battery_type` |
| `battery_cycle_life` | 2,028 | 2.03% | Group median by `battery_type` + `category` |
| `customer_rating` | 1,931 | 1.93% | Group median by `category` |
| `infotainment_screen_inch` | 1,506 | 1.51% | Group median by `category` |
| `annual_sales_units` | 989 | 0.99% | Group median by `category` + `market_region` |

## ðŸš€ How to Run Locally

```powershell
cd ev-market-intelligence-dashboard
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```
