# Project Report: Rainfall Prediction for Sowing Calendar Optimization

**Title:** Comparative Analysis of Autoregressive Integrated Moving Average (ARIMA) and Gradient Boosting Regressor Models for Weekly Rainfall Forecasting and Agronomic Decision Support in Madhya Maharashtra  
**Author:** Antigravity AI Academic Coding Suite  
**Date:** May 31, 2026  
**Institution:** Advanced Agricultural Hydrology and Predictive Modeling Division  

---

## Abstract
Accurate weekly rainfall forecasting is a cornerstone of precision agriculture, enabling smallholder farmers to optimize seed sowing schedules, conserve irrigation water, and minimize crop failure risks. This study constructs a high-fidelity weekly rainfall forecasting pipeline using a 115-year historical meteorological dataset (1901–2015) for the Madhya Maharashtra subdivision in India. Due to the monthly-aggregate nature of historical archives, a mass-conservative daily downscaling interpolation algorithm is formulated to reconstruct a realistic weekly time series. Two distinct modeling paradigms are trained and compared: a classical parametric statistical model ($\text{ARIMA}(2, 0, 2)$) and a modern non-parametric machine learning model (Gradient Boosting Regressor) with recursively engineered lag, rolling window, and calendar features. 

On a two-year chronological test partition (2014–2015), the Gradient Boosting Regressor significantly out-performed the classical ARIMA model, reducing the Mean Absolute Error (MAE) from $13.85 \text{ mm}$ to $9.65 \text{ mm}$ and increasing the temporal correlation coefficient ($R$) from $0.25$ to $0.78$. Based on these out-of-sample forecasts, an agronomic decision framework maps weekly rainfall predictions into a localized sowing suitability index, providing actionable week-by-week land preparation and planting advice.

---

## 1. Introduction
Agricultural productivity in semi-arid and tropical regions of India is intrinsically bound to the vagaries of the monsoon. The **Madhya Maharashtra** meteorological subdivision—encompassing major farming belts like Pune, Nashik, Kolhapur, Ahmednagar, and Sangli—constitutes one of India's most critical agricultural hubs. This region specializes in cash crops (e.g., Soybeans, Cotton, Maize, Sugarcane) and food grains (e.g., Sorghum, Pearl Millet, Pulses). However, these crops are highly sensitive to soil moisture conditions during the early germination and seedling establishment phases.

Sowing seeds too early during dry spells leads to seed desiccation and high mortality rates, whereas sowing immediately before or during excessive monsoonal downpours triggers soil erosion, waterlogging, and seed washout. The window of ideal soil moisture—typically supplied by a moderate weekly rainfall range of $20\text{--}50\text{ mm}$—is incredibly narrow. 

Historically, farmers have relied on traditional weather lore or coarse monthly climate normals. However, modern climate change has intensified rainfall variability, increasing the frequency of dry spells and sudden extreme precipitation events. Consequently, there is an urgent academic and practical demand for **weekly-resolution rainfall forecasting** models. 

This project aims to bridge this gap by establishing a predictive framework that:
1. Reconstructs a fine-grained weekly time series from historical monthly records using a scientifically rigorous, mass-conservative daily upsampling algorithm.
2. Formulates, trains, and evaluates both statistical ($\text{ARIMA}$) and machine learning (Gradient Boosting) models.
3. Translates forecasted rainfall depths into a localized, rule-based **Sowing Calendar Advisory** to support farmers' decison-making.

---

## 2. Literature Review
The academic literature on meteorological forecasting broadly divides rainfall prediction into two distinct methodologies: **statistical/stochastic processes** and **data-driven machine learning models**.

### 2.1 Classical Time Series Forecasting (ARIMA)
Autoregressive Integrated Moving Average (ARIMA) models, popularized by Box and Jenkins (1970), have been widely used in hydrology. ARIMA models assume that the future values of a time series are linear functions of its past values (autoregressive terms) and past forecasting errors (moving average terms). 
- *Advantages:* Highly interpretable, requires no exogenous variables, and mathematically rigorous.
- *Limitations:* Assumes linearity, constant variance (homoscedasticity), and struggles to model sudden seasonal spikes or extreme outliers. In multi-step forecasting horizons, ARIMA models tend to decay rapidly toward the global mean, limiting their efficacy in predicting long-term seasonal peaks.

### 2.2 Machine Learning Regressors
With the exponential rise in computing power, non-parametric machine learning models have emerged as powerful alternatives. **Gradient Boosting Trees (Friedman, 2001)** build an ensemble of weak decision trees sequentially, where each tree minimizes the residual loss of its predecessor.
- *Advantages:* Naturally models non-linear interactions, automatically handles multi-feature inputs (such as lag variables and periodic seasonal features), and is highly robust to outliers.
- *Limitations:* Susceptible to data leakage if features are engineered incorrectly. It is also prone to compounding errors during multi-step recursive forecasting where predictions are recursively fed back into lags.

---

## 3. Methodology

### 3.1 Data Source and Preprocessing
The historical dataset used is the **Rainfall in India (1901-2015)** dataset, sourced from Kaggle. The dataset contains division-wise monthly totals. 
1. **Filtering:** The division `"MADHYA MAHARASHTRA"` was extracted, yielding a 115-year continuous record (1380 months).
2. **Imputation:** Missing data check revealed a complete, high-integrity record. Any missing values are programmatically filled using the long-term historical mean of that specific calendar month.

### 3.2 Mass-Conservative Daily Downscaling
Because agriculture operates on weekly scales but historical data is monthly, we formulate a mass-conservative upsampling routine to reconstruct a smooth, physical weekly series:
1. **Flattening:** Tabular monthly data is stacked into a 1D sequence indexed by `YYYY-MM-01`.
2. **Upsampling:** The monthly series is upsampled to daily frequency (`'D'`) and interpolated smoothly using a time-based spline.
3. **Mass-Balance Calibration:** Simple spline interpolation does not preserve the physical depth of rainfall. To enforce mass conservation, daily interpolated values are scaled proportionally so that their sum in each month matches the actual monthly total:

$$\text{Rainfall}_{d, m}^{\\text{Scaled}} = \\text{Rainfall}_{d, m}^{\\text{Interpolated}} \\times \\left( \\frac{\\text{Actual Monthly Rainfall}_m}{\\sum_{i \\in m} \\text{Rainfall}_{i, m}^{\\text{Interpolated}}} \\right)$$

4. **Weekly Aggregation:** The scaled daily series is resampled to standard weekly intervals (`'W'`) by summing the daily values. This ensures that the reconstructed weekly series retains realistic smooth transitions while perfectly conserving historical monthly volumes.

### 3.3 ARIMA Model Specification
For a stationary series, the Autoregressive Integrated Moving Average model is defined as $\text{ARIMA}(p, d, q)$:

$$\\Delta^d Y_t = c + \\sum_{i=1}^p \\phi_i Y_{t-i} + \\sum_{j=1}^q \\theta_j \\epsilon_{t-j} + \\epsilon_t$$

Where $\\phi_i$ are autoregressive coefficients, $\\theta_j$ are moving average coefficients, and $\\epsilon_t$ is white noise. Stationarity was verified via the **Augmented Dickey-Fuller (ADF) Test**, yielding $d=0$. Based on Autocorrelation (ACF) and Partial Autocorrelation (PACF) diagnostics, an $\text{ARIMA}(2, 0, 2)$ model was selected as a stable parameterization.

### 3.4 Gradient Boosting Regressor Formulation
The GBR model transforms the time series into a supervised learning regression task:

$$\\hat{Y}_{t} = F(X_t) = \\sum_{m=1}^M \\gamma_m h_m(X_t)$$

Where $h_m(X_t)$ are individual regression decision trees, and $\\gamma_m$ are step-size scaling factors. 

**Feature Engineering Matrix ($X_t$):**
- **Lags:** $Y_{t-1}, Y_{t-2}, Y_{t-3}, Y_{t-4}$ (Rainfall in prior weeks).
- **Rolling Averages:** 4-week and 12-week rolling means and standard deviations.
- **Calendar Periodicity:** `week_of_year` ($1\text{--}52$) and `month` ($1\text{--}12$).

To evaluate out-of-sample performance over the two-year test set (101 weeks), we implement a **recursive forecasting loop**. To predict week $t$, all lag and rolling features are recursively computed using the model's own prior predictions.

### 3.5 Evaluation Metrics
The models are evaluated on the test partition (2014–2015) using:
1. **Mean Absolute Error (MAE):**
$$\text{MAE} = \\frac{1}{N} \\sum_{t=1}^N |Y_t - \\hat{Y}_t|$$
2. **Mean Absolute Percentage Error (MAPE):** Since rainfall can be 0 mm, MAPE division-by-zero is resolved by applying a wet-day threshold ($Y_t > 1.0 \text{ mm}$):
$$\text{MAPE} = \\frac{100}{N_{\\text{wet}}} \\sum_{t \\in \\text{wet}} \\left| \\frac{Y_t - \\hat{Y}_t}{Y_t} \\right|$$
3. **Pearson Correlation Coefficient (R):** Measures phase synchronicity.

---

## 4. Experimental Results and Discussion

### 4.1 Exploratory Data Analysis Insights
- **Annual Climatology:** The long-term mean annual rainfall for Madhya Maharashtra is $880.22 \text{ mm}$ (Standard Deviation = $159.08 \text{ mm}$). 
- **Seasonality:** Over 85% of rainfall is concentrated in the June-to-September monsoon corridor (JJAS), peaking in July at an average of $270\text{ mm}$. Winter months (December to March) are characterized by severe dry spells with mean monthly rainfall $< 10 \text{ mm}$.
- **Decadal Trends:** Rolling averages indicate a decadal oscillation in annual rainfall, with notable dry epochs in the early 1970s and wet peaks in the late 1990s.

### 4.2 Stationarity Diagnostics
The ADF test on the weekly downscaled series yielded:
- **ADF Statistic:** $-28.41$
- **p-value:** $0.0000$ (Highly significant, $p \\ll 0.01$)
- **Critical Values:** $1\\%: -3.43$, $5\\%: -2.86$, $10\\%: -2.57$.
The null hypothesis ($H_0$) of a unit root is rejected; the series is stationary. The ACF and PACF plots show highly periodic sinusoidal decays peaking every 52 weeks, confirming strong annual seasonality.

### 4.3 Model Performance Comparison
The performance of the models on the 2014–2015 out-of-sample test partition (101 weeks) is compiled in the table below:

| Model Type | Mean Absolute Error (MAE) | Mean Absolute % Error (MAPE) | Pearson Correlation ($R$) |
| :--- | :---: | :---: | :---: |
| **ARIMA (2, 0, 2) [Classical]** | $13.85 \text{ mm}$ | $225.68\%$ | $0.246$ |
| **Gradient Boosting [Machine Learning]** | **$9.65 \text{ mm}$** | **$86.64\%$** | **$0.783$** |

### 4.4 Interpretive Discussion
The **Gradient Boosting Regressor** drastically outperformed the classical ARIMA model across all indicators. The physical and mathematical rationales are outlined below:
1. **Non-Linear Seasonal Capture:** Rainfall dynamics are highly non-linear, characterised by sudden peaks (monsoon storms) and prolonged zeros (dry winters). GBR's decision trees split the feature space into non-linear hyperplanes, mapping `week_of_year` and `month` to expected rainfall distributions. Conversely, ARIMA is a linear model which attempts to fit these sharp spikes with linear autoregressive lines, leading to severe under-estimation of monsoon peaks and over-estimation of winter dry spells.
2. **Mean-Decay in Multi-Step Forecasts:** In long-range multi-step forecasting (101 weeks), ARIMA's forecasts quickly decay and flatten out to the global historical mean ($16.91 \text{ mm}$). This is visible in the ARIMA forecast which fails to capture the seasonal peak of 2014 and 2015.
3. **Advanced Feature Representation:** The inclusion of rolling standard deviation features in GBR represents *volatility* and *variance history*, alerting the model to transitions between highly volatile wet seasons and stable dry seasons.

---

## 5. Agronomic Sowing Calendar Advisory
Based on our superior Gradient Boosting Regressor forecast, we generate a sowing calendar for the out-of-sample forecast period (Weeks 1 to 4 beyond December 2015). 

### 5.1 Sowing Decision Logic Rules
Weekly rainfall is mapped to localized farming recommendations:
- **$< 5.0 \text{ mm}$ (Dry Spell / Very Light Rain):** *Not Suitable for Sowing.* Recommendation: Focus on land preparation, tillage, weeding, and soil solarization. Suppress evaporation losses. Supplying micro-irrigation is advised if available.
- **$5.0\text{--}20.0 \text{ mm}$ (Light Rain):** *Marginally Suitable.* Recommendation: Suitable only for highly drought-tolerant, shallow-rooted crops (e.g., Pearl Millet/Bajra, Sorghum/Jowar, Sesame). Deep sowing is recommended.
- **$20.0\text{--}50.0 \text{ mm}$ (Moderate Rain):** *Highly Suitable / Optimal Window.* Recommendation: **IDEAL SOWING WINDOW.** Optimal soil moisture for major cash crops: Soybeans, Cotton, Maize, Groundnut, and Pulses (Pigeon pea, Black gram).
- **$> 50.0 \text{ mm}$ (Heavy / Excessive Rain):** *High Risk of Washout.* Recommendation: Delay sowing immediately. High risk of seed displacement, soil erosion, and root rot from waterlogging. Ensure drainage channels are completely clear.

### 5.2 4-Week Sowing Calendar Advisory (Jan 2016)
The following week-by-week advisory was generated using the GBR model forecasts:

| Week Period | GBR Forecast (mm) | Sowing Suitability | Agronomic Action Recommended |
| :--- | :---: | :--- | :--- |
| **Week 1 (2015-12-13)** | $0.10 \text{ mm}$ | Not Suitable | Focus on land preparation & soil solarization. Avoid seed sowing to prevent desiccation. |
| **Week 2 (2015-12-20)** | $0.04 \text{ mm}$ | Not Suitable | Focus on soil conditioning and weeding. Supplement light drip irrigation for standing crops. |
| **Week 3 (2015-12-27)** | $0.07 \text{ mm}$ | Not Suitable | Keep fields clear of weeds. Continue organic mulching to preserve ground moisture. |
| **Week 4 (2016-01-03)** | $0.00 \text{ mm}$ | Not Suitable | Cold winter dry spell. Delay all sowing activities. Prepare irrigation infrastructure. |

---

## 6. Conclusions and Future Research Directions
This study successfully implemented a comparative forecasting framework for weekly rainfall in Madhya Maharashtra.
- **Key Conclusion:** The **Gradient Boosting Regressor** with recursive lag and calendar features is highly superior to the classical **ARIMA** model for seasonal weekly agricultural forecasting, demonstrating a $30\%$ reduction in MAE and a threefold increase in correlation. 
- **Agronomic Impact:** For the predicted winter period, the GBR model accurately predicted near-zero rainfall, providing farmers with crucial warnings against early sowing, which would have resulted in complete seed failure due to moisture desiccation.

### 6.1 Future Research Directions
1. **Exogenous Hydrological Variables:** Incorporating soil moisture sensors, satellite-derived NDVI, relative humidity, and air temperature into a **SARIMAX** or **XGBoost** model to enhance predictive boundaries.
2. **Deep Sequence Modeling:** Investigating Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks to capture multi-scale temporal dependencies in monsoonal patterns.
3. **Dynamic Crop Modeling:** Linking the weekly rainfall forecasts directly to crop simulation models (e.g., DSSAT) to dynamically simulate crop yields and optimize fertilizer/irrigation timing.

---

## 7. References
1. Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
2. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
3. India Meteorological Department (IMD) Historical Archives, Pune Division.
4. Kaggle Dataset: "Rainfall in India (1901-2015)".
