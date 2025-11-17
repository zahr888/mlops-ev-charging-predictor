
import argparse
import pandas as pd
import numpy as np
import boto3
from pathlib import Path

# --- Command-line arguments for config ---
def parse_args():
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--input", required=True, help="Path to raw data (local or s3)")
    return parser.parse_args()

def save_locally(df, output_path, file_name):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    local_file_path = f"{output_path}/{file_name}"
    df.to_parquet(local_file_path, 
                     engine='pyarrow', 
                     compression='snappy',
                     index=False)
    print(f"[OK] Cleaned data saved locally: {local_file_path}")

def cleaning(df):
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    #converting el_kwh and duration_hours to numeric, coercing errors to NaN
    df['el_kwh'] = pd.to_numeric(df['el_kwh'], errors='coerce')
    df['duration_hours'] = pd.to_numeric(df['duration_hours'], errors='coerce')

    #fixing missing values in 'end_plugout' and 'duration_hours' by assigning correct values
    missing_both = df[df['end_plugout'].isna() & df['duration_hours'].isna() & df['duration_category'].isna()]

    complete_sessions = df.dropna(subset=['end_plugout', 'duration_hours']).copy()
    complete_sessions['charging_rate'] = complete_sessions['el_kwh'] / complete_sessions['duration_hours']
    avg_charging_rate = complete_sessions['charging_rate'].median()

    # Step 2: Get indices of missing rows before fixing them
    missing_indices = df[df['end_plugout'].isna() & df['duration_hours'].isna()].index

    # Step 3: Estimate missing durations using average rate
    df.loc[missing_indices, 'duration_hours'] = df.loc[missing_indices, 'el_kwh'] / avg_charging_rate

    # Step 4: Calculate end_plugout from estimated duration
    df.loc[missing_indices, 'end_plugout'] = df.loc[missing_indices, 'start_plugin'] + pd.to_timedelta(df.loc[missing_indices, 'duration_hours'], unit='h')

    # Step 5: Update end_plugout_hour
    df.loc[missing_indices, 'end_plugout_hour'] = df.loc[missing_indices, 'end_plugout'].dt.hour

    # Step 6: Update duration_category based on estimated duration
    for idx in missing_indices:
        duration = df.loc[idx, 'duration_hours']
        if duration > 18:
            df.loc[idx, 'duration_category'] = "More than 18 hours"
        elif duration > 15:
            df.loc[idx, 'duration_category'] = "Between 15 and 18 hours"
        elif duration > 12:
            df.loc[idx, 'duration_category'] = "Between 12 and 15 hours"
        elif duration > 9:
            df.loc[idx, 'duration_category'] = "Between 9 and 12 hours"
        elif duration > 6:
            df.loc[idx, 'duration_category'] = "Between 6 and 9 hours"
        elif duration > 3:
            df.loc[idx, 'duration_category'] = "Between 3 and 6 hours"
        else:
            df.loc[idx, 'duration_category'] = "Less than 3 hours"

    # Fix duration mismatches
    df['charging_duration'] = (df['end_plugout'] - df['start_plugin']).dt.total_seconds() / 3600
    duration_diff = abs(df['duration_hours'] - df['charging_duration'])
    df.loc[duration_diff > 0.1, 'duration_hours'] = df['charging_duration']
    df = df.drop('charging_duration', axis=1)

    category_columns = [
    'user_type', 'shared_id', 'month_plugin', 'weekdays_plugin', 'plugin_category', 'duration_category']
    for col in category_columns:
        df[col] = df[col].astype('category')
    df['end_plugout_hour'] = df['end_plugout_hour'].astype('int64')


    #verifying start_plugin_hour
    df['hour'] = df['start_plugin'].dt.hour


    df['start_plugin_hour'] = np.where(
        df['start_plugin_hour'] != df['hour'],
        df['hour'],                             
        df['start_plugin_hour']                 
    )

    # aggregating data on hourly basis
    df['hour'] = df['start_plugin'].dt.floor('h')
    save_locally(df, 'C:/Users/GIGABYTE/Documents/ml/mlops/data/clean','clean.parquet')

    return df

def engineering(df):
    hourly_total = (df.groupby('hour', as_index=True).agg(total_kwh=('el_kwh','sum'),n_sessions=('session_id','count'),avg_kwh=('el_kwh','mean')).sort_index())

    hourly_total['n_sessions_lag1']  = hourly_total['n_sessions'].shift(1)
    hourly_total['avg_kwh_lag1']    = hourly_total['avg_kwh'].shift(1)
    hourly_total = hourly_total.drop(columns=['n_sessions','avg_kwh'])
    hourly_total['hour_of_day'] = hourly_total.index.hour
    hourly_total['day_of_week'] = hourly_total.index.dayofweek
    hourly_total['month'] = hourly_total.index.month
    hourly_total['is_weekend'] = hourly_total.index.dayofweek.isin([5,6]).astype(int)

    #cyclical encoding 
    hourly_total['hour_sin'] = np.sin(2*np.pi*hourly_total['hour_of_day']/24)
    hourly_total['hour_cos'] = np.cos(2*np.pi*hourly_total['hour_of_day']/24)
    hourly_total['dow_sin'] = np.sin(2*np.pi*hourly_total['day_of_week']/7)
    hourly_total['dow_cos'] = np.cos(2*np.pi*hourly_total['day_of_week']/7)
    hourly_total['month_sin'] = np.sin(2*np.pi*(hourly_total['month']-1)/12)
    hourly_total['month_cos'] = np.cos(2*np.pi*(hourly_total['month']-1)/12)

    # add lags
    hourly_total['lag_1']   = hourly_total['total_kwh'].shift(1)
    hourly_total['lag_24']  = hourly_total['total_kwh'].shift(24)
    hourly_total['lag_168'] = hourly_total['total_kwh'].shift(168)

    # differences t-1 - t-2 (preventing data leakage)
    hourly_total['diff_lag1'] = hourly_total['total_kwh'].shift(1) - hourly_total['total_kwh'].shift(2)

    # rolling stats only from past values
    hourly_total['roll_mean_3h']  = hourly_total['total_kwh'].shift(1).rolling(window=3).mean()
    hourly_total['roll_mean_6h']  = hourly_total['total_kwh'].shift(1).rolling(window=6).mean()
    hourly_total['roll_mean_24h'] = hourly_total['total_kwh'].shift(1).rolling(window=24).mean()
    hourly_total['roll_std_24h']  = hourly_total['total_kwh'].shift(1).rolling(window=24).std()
    hourly_total['roll_mean_168h'] = hourly_total['total_kwh'].shift(1).rolling(window=168).mean()

    # calculating mean total_kwh for each (hour_of_day, day_of_week) combination using only past data
    expanding_means = (
        hourly_total
        .groupby(['hour_of_day', 'day_of_week'])['total_kwh']
        .apply(lambda x: x.shift(1).expanding().mean())
    )
    expanding_means = expanding_means.reset_index(level=[0, 1], drop=True)
    hourly_total['hour_dow_mean'] = expanding_means
    hourly_total = hourly_total.dropna().copy()
    save_locally(hourly_total, 'C:/Users/GIGABYTE/Documents/ml/mlops/data/features','features.parquet')

def upload_to_s3(local_path, file_name, bucket='ev-data'):
    s3 = boto3.client('s3', 
                     endpoint_url="http://localhost:4566",
                     aws_access_key_id="test", 
                     aws_secret_access_key="test")
    
    # Create bucket if not exists
    try:
        s3.create_bucket(Bucket=bucket)
    except:
        pass
    
    # Upload the file
    local_file = f"{local_path}/{file_name}"
    s3_key = f'parquets/{file_name}'
    s3.upload_file(local_file, bucket, s3_key)
    print(f"Uploaded: {s3_key}")


def main(args):
    # Load raw data
    print(f"Loading raw data from: {args.input}")
    if args.input.startswith("s3://"):  # From S3/LocalStack
        df = pd.read_parquet(args.input, storage_options={ "client_kwargs": {"endpoint_url": "http://localhost:4566"},
    "key": "test",
    "secret": "test" }, engine='pyarrow')
    else:
        df = pd.read_parquet(args.input)

    df = cleaning(df)
    engineering(df)
    upload_to_s3('C:/Users/GIGABYTE/Documents/ml/mlops/data/features','features.parquet','ev-data')
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
