# NeuralRetail — AI-Powered Retail Analytics Platform

NeuralRetail is an end-to-end machine learning pipeline built on the **Online Retail II** dataset (500,000+ UK e-commerce transactions). It cleans raw transaction data, engineers customer and product features, trains three machine learning models, and serves live predictions through an interactive Streamlit dashboard.

---

## 🎯 What This Project Does

| Problem | Solution | Result |
|---------|----------|--------|
| Which customers are loyal vs at risk of leaving? | K-Means customer segmentation | 5 customer segments (Champions, Loyal, Potential Loyalists, At Risk, Lost) |
| Which customers will stop buying? | Stacked XGBoost + LightGBM + Logistic Regression churn model | AUC 0.83 (target ≥ 0.75) |
| How many units of each product will sell next week? | LightGBM demand forecasting regressor | WMAPE ≤ 30% daily, ≤ 10% monthly |

---

## 📂 Project Structure

```
NeuralRetail/
├── app.py                              # Streamlit dashboard (analytics + live predictions)
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Excludes large raw files from git
│
├── 01_data_ingestion_cleaning.ipynb    # Notebook 1 — Clean raw Excel data
├── 02_feature_engineering.ipynb        # Notebook 2 — Build RFM + behavioural + lag features
├── 03_model_training_evaluation.ipynb  # Notebook 3 — Train & evaluate all 3 models
│
├── data/
│   ├── silver_retail.parquet           # Cleaned transaction-level data
│   ├── gold_customer_features.parquet  # One row per customer with all features
│   ├── gold_churn_scores.parquet       # Churn probability per customer
│   ├── gold_sku_timeseries.parquet     # Daily demand + lag features per SKU
│   ├── gold_monthly_revenue.parquet    # Monthly revenue summary
│   └── gold_clv_scores.parquet         # Customer Lifetime Value scores
│
└── models/
    ├── kmeans_model.pkl                # Trained K-Means segmentation model
    ├── kmeans_scaler.pkl               # RobustScaler used before K-Means
    ├── cluster_labels.json             # Maps cluster number → segment name
    ├── xgb_churn_model.pkl             # XGBoost base churn model
    ├── lgb_churn_model.pkl             # LightGBM base churn model
    ├── meta_churn_model.pkl            # Logistic Regression meta-model (stacking)
    ├── churn_features.json             # Exact feature list/order for churn model
    ├── lgb_forecast_model.pkl          # LightGBM demand forecasting model
    ├── forecast_features.json          # Exact feature list/order for forecast model
    └── sku_encoder.json                # Maps stock codes → integer SKU_ID
```

---

## 🧠 The Pipeline — 3 Notebooks

### 1️⃣ Data Ingestion & Cleaning (`01_data_ingestion_cleaning.ipynb`)
- Loads both sheets of `online_retail_II.xlsx` and merges them
- Removes cancelled orders (invoices starting with "C"), negative quantities, zero/negative prices, invalid stock codes, missing customer IDs, and duplicate rows
- Caps outliers using the IQR method (3× multiplier) instead of deleting them
- Creates the `Revenue` column (`Quantity × Price`)
- Saves clean data as `silver_retail.parquet`

### 2️⃣ Feature Engineering (`02_feature_engineering.ipynb`)
- Builds **RFM** features (Recency, Frequency, Monetary) + quintile-based R/F/M scores per customer
- Builds behavioural features: `ProductDiversity`, `RevenueConsistency`, `OrderFrequencyRate`, `AvgBasketRevenue`, preferred shopping day/hour, and more
- Builds time-series features for the top 20 SKUs: lag features (1/7/14/30 days), rolling mean/std (7/14/30 days), and calendar features (day of week, month, week of year, weekend flag)
- Saves all outputs as gold-layer parquet files

### 3️⃣ Model Training & Evaluation (`03_model_training_evaluation.ipynb`)
- **K-Means**: scales features with `RobustScaler`, selects optimal *k* via Silhouette Score, names clusters based on average Recency/Frequency
- **Churn Model**: trains XGBoost + LightGBM, generates out-of-fold predictions with `cross_val_predict`, stacks them with a Logistic Regression meta-model. Recency is deliberately excluded to avoid data leakage.
- **Demand Forecasting**: trains a LightGBM regressor on lag/rolling/calendar features using a strict temporal train/test split (no shuffling)
- Saves all trained models and feature configs to `models/`

---

## 📏 Key Metrics

| Model | Metric | Why This Metric | Score |
|-------|--------|-----------------|-------|
| K-Means | Silhouette Score | Measures cluster tightness *and* separation — Inertia alone can't do this | ≥ 0.35 target |
| Churn Model | AUC-ROC | Threshold-independent, immune to class imbalance unlike Accuracy | **0.83** (target ≥ 0.75) |
| Demand Forecast | WMAPE | Regular MAPE explodes on near-zero daily sales; WMAPE weights errors by actual volume | ≤ 30% daily target |

---

## ⚠️ Key Engineering Decisions

- **No Recency in churn features** — Recency (days since last purchase) and the churn label (no purchase in 90+ days) measure the same thing. Including it caused a fake AUC of 1.0 (data leakage). Removing it gave a realistic AUC of 0.83.
- **WMAPE over MAPE** — MAPE reported 159% error early on due to near-zero daily sales values distorting the percentage. WMAPE fixed this by weighting errors by sales volume.
- **Temporal split for forecasting** — time series data is split by date (train on past, test on last 30 days), never shuffled randomly.
- **`scale_pos_weight`** — corrects class imbalance in the churn dataset so the model doesn't just predict "no churn" for everyone.
- **IQR capping + RobustScaler** — handles outliers (extreme wholesale orders) without deleting valid rows.

---

## 🖥️ Running the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard includes:
- **Analytics pages**: Overview, Demand Forecasting, Customer Segments, Churn Risk, Inventory Optimizer, Model Monitoring
- **Live Prediction pages**: Churn Predictor, Segment Predictor, Demand Predictor — run the actual trained `.pkl` models in real time on new input data

---

## 🚀 Deployment

Deployed on **Streamlit Community Cloud**, connected to a private GitHub repository. Large raw files (`online_retail_II.xlsx`, `silver_retail.csv`) are excluded via `.gitignore`; only the cleaned parquet files and trained models are version-controlled.

---

## 🔧 Tech Stack

- **Data processing**: pandas, numpy, pyarrow
- **Modeling**: scikit-learn (K-Means, Logistic Regression, RobustScaler), XGBoost, LightGBM
- **Dashboard**: Streamlit, Plotly
- **Deployment**: Streamlit Community Cloud, GitHub

---

## 🚧 Roadmap to Production

- Automated retraining pipeline (Airflow / scheduled jobs)
- Hyperparameter tuning with Optuna
- Data drift monitoring (PSI)
- SHAP-based model explainability
- REST API layer (FastAPI) for programmatic predictions
- MLflow model versioning
- Authentication on the dashboard + GDPR-compliant data handling
- A/B testing framework to validate future model improvements
