import os
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress warnings from statsmodels during parallel fit
warnings.filterwarnings("ignore")

def get_subdivision_data(csv_path, subdivision):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    sub_df = df[df['DIVISION'].str.upper() == subdivision.upper()].copy()
    if sub_df.empty:
        raise ValueError(f"Subdivision '{subdivision}' not found.")
        
    # Impute missing values
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    for m in months:
        if sub_df[m].isnull().any():
            sub_df[m] = sub_df[m].fillna(sub_df[m].mean())
    sub_df['ANNUAL'] = sub_df[months].sum(axis=1)
    
    # 1D monthly sequence
    records = []
    for _, row in sub_df.iterrows():
        year = int(row['YEAR'])
        for m_idx, m_name in enumerate(months, start=1):
            date_str = f"{year}-{m_idx:02d}-01"
            records.append({'Date': pd.to_datetime(date_str), 'Rainfall': float(row[m_name])})
            
    monthly_series = pd.DataFrame(records).set_index('Date').sort_index()
    
    # Upsample to daily and interpolate
    daily_df = monthly_series.resample('D').asfreq()
    daily_df['Rainfall'] = daily_df['Rainfall'].interpolate(method='time')
    
    # Mass-conservative scaling
    daily_df['YearMonth'] = daily_df.index.to_period('M')
    monthly_series['YearMonth'] = monthly_series.index.to_period('M')
    daily_monthly_sums = daily_df.groupby('YearMonth')['Rainfall'].transform('sum')
    original_monthly_vals = daily_df['YearMonth'].map(monthly_series.set_index('YearMonth')['Rainfall'])
    
    scale_factors = original_monthly_vals / (daily_monthly_sums + 1e-8)
    daily_df['Rainfall'] = (daily_df['Rainfall'] * scale_factors).clip(lower=0.0)
    
    # Resample daily data to weekly
    weekly_series = daily_df[['Rainfall']].resample('W').sum()
    return weekly_series, sub_df

def run_models_for_subdivision(weekly_series):
    # Train-test split
    train_cutoff_date = '2014-01-01'
    train_data = weekly_series[weekly_series.index < train_cutoff_date]
    test_data = weekly_series[weekly_series.index >= train_cutoff_date]
    
    # Fit ARIMA (2, 0, 2)
    arima_model = ARIMA(train_data['Rainfall'], order=(2, 0, 2))
    arima_fit = arima_model.fit()
    arima_pred = arima_fit.forecast(steps=len(test_data)).clip(lower=0.0)
    
    # Fit GBR with Feature Engineering
    df = weekly_series.copy()
    for l in range(1, 5):
        df[f'lag_{l}'] = df['Rainfall'].shift(l)
    df['rolling_mean_4'] = df['lag_1'].rolling(window=4).mean()
    df['rolling_std_4'] = df['lag_1'].rolling(window=4).std()
    df['rolling_mean_12'] = df['lag_1'].rolling(window=12).mean()
    df['rolling_std_12'] = df['lag_1'].rolling(window=12).std()
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    ml_data = df.dropna()
    
    feature_cols = [c for c in ml_data.columns if c != 'Rainfall']
    
    train_df = ml_data[ml_data.index < train_cutoff_date]
    test_df = ml_data[ml_data.index >= train_cutoff_date]
    
    gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42, subsample=0.8)
    gbr.fit(train_df[feature_cols], train_df['Rainfall'])
    
    # Recursive Forecasting on Test Set
    gbr_test_predictions = []
    current_history = weekly_series[weekly_series.index < train_cutoff_date]['Rainfall'].tolist()
    test_indices = test_df.index
    
    for date in test_indices:
        step_lags = [current_history[-l] for l in range(1, 5)]
        roll_4_mean = np.mean(current_history[-4:])
        roll_4_std = np.std(current_history[-4:])
        roll_12_mean = np.mean(current_history[-12:])
        roll_12_std = np.std(current_history[-12:])
        week = int(date.isocalendar().week)
        month = int(date.month)
        
        feat_dict = {
            'lag_1': step_lags[0], 'lag_2': step_lags[1], 'lag_3': step_lags[2], 'lag_4': step_lags[3],
            'rolling_mean_4': roll_4_mean, 'rolling_std_4': roll_4_std,
            'rolling_mean_12': roll_12_mean, 'rolling_std_12': roll_12_std,
            'week_of_year': week, 'month': month
        }
        X_step = pd.DataFrame([feat_dict])[feature_cols]
        pred_val = max(0.0, gbr.predict(X_step)[0])
        gbr_test_predictions.append(pred_val)
        current_history.append(pred_val)
        
    # Future 4-Week Forecast (Weeks 1-4, 2016)
    last_date = weekly_series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=4, freq='W')
    
    arima_future = arima_fit.forecast(steps=4).clip(lower=0.0).tolist()
    
    gbr_future_predictions = []
    current_history = weekly_series['Rainfall'].tolist()
    
    for date in future_dates:
        step_lags = [current_history[-l] for l in range(1, 5)]
        roll_4_mean = np.mean(current_history[-4:])
        roll_4_std = np.std(current_history[-4:])
        roll_12_mean = np.mean(current_history[-12:])
        roll_12_std = np.std(current_history[-12:])
        week = int(date.isocalendar().week)
        month = int(date.month)
        
        feat_dict = {
            'lag_1': step_lags[0], 'lag_2': step_lags[1], 'lag_3': step_lags[2], 'lag_4': step_lags[3],
            'rolling_mean_4': roll_4_mean, 'rolling_std_4': roll_4_std,
            'rolling_mean_12': roll_12_mean, 'rolling_std_12': roll_12_std,
            'week_of_year': week, 'month': month
        }
        X_step = pd.DataFrame([feat_dict])[feature_cols]
        pred_val = max(0.0, gbr.predict(X_step)[0])
        gbr_future_predictions.append(pred_val)
        current_history.append(pred_val)
        
    return {
        'test_dates': [d.strftime('%Y-%m-%d') for d in test_indices],
        'test_actuals': test_df['Rainfall'].tolist(),
        'arima_test_preds': arima_pred.tolist(),
        'gbr_test_preds': gbr_test_predictions,
        'future_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
        'arima_future': arima_future,
        'gbr_future': gbr_future_predictions
    }

def process_single_subdivision(csv_path, sub):
    try:
        print(f"Modeling subdivision: {sub}")
        weekly, sub_df = get_subdivision_data(csv_path, sub)
        model_results = run_models_for_subdivision(weekly)
        
        annual_mean = float(sub_df['ANNUAL'].mean())
        annual_min = float(sub_df['ANNUAL'].min())
        annual_max = float(sub_df['ANNUAL'].max())
        
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        monthly_normals = sub_df[months].mean().to_dict()
        
        result_dict = {
            'annual_mean': annual_mean,
            'annual_min': annual_min,
            'annual_max': annual_max,
            'monthly_normals': monthly_normals,
            'forecasts': model_results
        }
        print(f"Successfully modeled: {sub}")
        return sub, result_dict
    except Exception as e:
        print(f"Failed modeling subdivision {sub}: {e}")
        return sub, None

def main():
    csv_path = "data/rainfall_in_india_1901-2015.csv"
    
    # Load raw dataset to extract all unique meteorological divisions
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    subdivisions = sorted(df['DIVISION'].unique().tolist())
    
    print("===============================================================")
    print(f"   PARALLEL PROCESSING PIPELINE: MODELING ALL {len(subdivisions)} DIVISIONS   ")
    print("===============================================================")
    
    dashboard_dir = "dashboard"
    if not os.path.exists(dashboard_dir):
        os.makedirs(dashboard_dir)
        
    consolidated_data = {}
    
    # We use ThreadPoolExecutor for highly parallelized multi-threaded calculations on the host
    # 4 to 8 workers is extremely optimal for parallel fitting
    max_workers = min(8, os.cpu_count() or 4)
    print(f"Launching ThreadPoolExecutor with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_subdivision, csv_path, sub) for sub in subdivisions]
        
        for future in futures:
            sub_name, result = future.result()
            if result is not None:
                consolidated_data[sub_name] = result
                
    # Save consolidated dataset to JSON
    with open(os.path.join(dashboard_dir, "data.json"), "w") as f:
        json.dump(consolidated_data, f, indent=2)
        
    print("\n===============================================================")
    print("   ALL SUBDIVISIONS NATION-WIDE MODELED & EXPORTED SUCCESSFULLY  ")
    print(f"   Database saved to dashboard/data.json ({len(consolidated_data)} divisions)")
    print("===============================================================")

if __name__ == "__main__":
    main()
