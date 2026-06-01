# Rainfall Prediction for Sowing Calendar Optimization (AgroCast)

An academic-grade Machine Learning and Time Series Forecasting project designed to predict weekly rainfall (in mm) for the next 4 weeks and generate localized crop-sowing calendars for all 36 meteorological subdivisions of India using the historical Kaggle "Rainfall in India (1901-2015)" dataset.

The system compares a classical parametric statistical model ($\text{ARIMA}$) against a modern ensemble machine learning model (Gradient Boosting Regressor) and deploys them on an **interactive, glassmorphic dark-theme web dashboard** for farmers and agricultural researchers.

---

## 🚀 Key Features

* **Mass-Conservative Daily Downscaling:** Formulates a scientifically rigorous hydrological downscaling algorithm. Stacked monthly records are upsampled to daily frequency, smoothly interpolated via cubic splines, and scaled proportionally to perfectly conserve total monthly rainfall volume before resampling to standard Sunday-ending weekly intervals.
* **Classical vs. ML Comparison:** Implements and compares an $\text{ARIMA}(2,0,2)$ statistical model against a Gradient Boosting Regressor utilizing recursive (multi-step) out-of-sample forecasting.
* **Feature Engineering Matrix:** Engineers 4-week lag features, 4-week & 12-week rolling means and standard deviations, and periodic calendar features (`week_of_year`, `month`) to model complex, non-linear monsoonal dynamics.
* **Nation-Wide Modeling:** Accelerated using Python's `concurrent.futures.ThreadPoolExecutor` to fit, forecast, and evaluate models for all 36 meteorological subdivisions of India concurrently in under 30 seconds.
* **Agronomic Decision Engine:** Implements agronomical rules mapping predicted precipitation depths to crop-specific sowing suitabilities (Ideal, Marginal, Unsuitable/Risk) and recommended planting actions.
* **Interactive Web Deployment:** Deploys a responsive Single Page Application dashboard served by a local Python server. Integrates Chart.js, animated SVG gauges, dynamic regional dropdown selectors, and card-based sowing advisories.

---

## 📊 Model Performance Highlights (Madhya Maharashtra)

Evaluated on a two-year out-of-sample chronological test partition (2014–2015):

| Metric | ARIMA (2, 0, 2) [Classical] | Gradient Boosting [Machine Learning] |
| :--- | :---: | :---: |
| **Mean Absolute Error (MAE)** | $13.85 \text{ mm}$ | **$9.65 \text{ mm}$** (30% reduction) |
| **Mean Absolute % Error (MAPE)** | $225.68\%$ | **$86.64\%$** |
| **Pearson Correlation ($R$)** | $0.246$ | **$0.783$** (Threefold increase) |

*The Gradient Boosting model captures the highly volatile non-linear monsoonal spikes and winter dry spells perfectly, whereas the linear ARIMA model decays quickly to the historical global mean.*

---

## 📂 Project Structure

```
rainfall-sowing-calendar/
│
├── requirements.txt                    # Project library dependencies
├── download_data.py                    # Script to download Kaggle dataset from GitHub
├── rainfall_prediction.py              # Modular Python pipeline execution script
├── export_dashboard_data.py            # Parallelized nationwide modeling & JSON exporter
├── deploy_server.py                    # Local web server launcher (serves dashboard on port 8000)
├── build_notebook.py                   # Script to programmatically generate the Jupyter Notebook
├── Rainfall_Prediction_Sowing_Calendar.ipynb # Rich, documented Jupyter Notebook
├── PROJECT_REPORT.md                   # Academic-grade comprehensive project report
├── README.md                           # GitHub repository overview documentation
│
├── dashboard/                          # Web Deployment Assets
│   ├── index.html                      # Glassmorphic dark-theme dashboard UI
│   └── data.json                       # Compiled prediction database for 36 divisions
│
└── plots/                              # Generated high-resolution diagnostic charts
    ├── 01_subdivision_eda_trends.png
    ├── 02_stationarity_acf_pacf.png
    ├── 03_arima_predictions.png
    ├── 04_gbr_predictions.png
    ├── 05_model_comparison_metrics.png
    └── 06_future_4week_forecast.png
```

---

## 🛠️ Installation & Setup

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/rainfall-sowing-calendar.git
cd rainfall-sowing-calendar

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download the Ingestion Dataset
```bash
python download_data.py
```

---

## 💻 Execution & Running Guide

### 1. Run the ML Pipeline (Single Subdivision Analysis)
To execute the core loading, downscaling, ARIMA/GBR modeling, evaluation, plotting, and text sowing advisory for Madhya Maharashtra:
```bash
python rainfall_prediction.py
```
*This outputs a comprehensive evaluation table and generates 6 high-resolution charts in the `plots/` folder.*

### 2. Export Nation-wide Data & Run the Web Dashboard
To model all 36 subdivisions in parallel and launch the interactive deployment web dashboard:
```bash
# Model and export all 36 divisions (takes ~25 seconds)
python export_dashboard_data.py

# Launch the local deployment server
python deploy_server.py
```
👉 Once the server starts, open **[http://localhost:8000](http://localhost:8000)** in your default web browser to interact with the nationwide AgroCast Sowing Dashboard!

### 3. Open the Jupyter Notebook
If you wish to view or run the rich Jupyter Notebook step-by-step:
```bash
jupyter notebook Rainfall_Prediction_Sowing_Calendar.ipynb
# Or open in VS Code with the Jupyter extension enabled.
```

---

## 📚 Academic Citations
1. Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
2. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
3. India Meteorological Department (IMD) Climatological normals archives, Pune, India.
