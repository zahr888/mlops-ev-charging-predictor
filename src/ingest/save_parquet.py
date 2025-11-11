#!/usr/bin/env python3
# src/ingest/save_parquet.py
import argparse
import pandas as pd
import boto3
import pathlib
import os

# Get the project root directory 
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()

def load_and_clean_csv(csv_path):
    """Load and clean the CSV file"""
    # Convert to absolute path if relative
    csv_path = PROJECT_ROOT / csv_path if not os.path.isabs(csv_path) else csv_path
    
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8', 
                     parse_dates=['Start_plugin','End_plugout'],
                     decimal=',', dayfirst=True)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Convert numeric columns
    df['el_kwh'] = pd.to_numeric(df['el_kwh'], errors='coerce')
    df['duration_hours'] = pd.to_numeric(df['duration_hours'], errors='coerce')
    
    # Drop rows with missing critical data
    df = df.dropna(subset=['end_plugout', 'duration_hours'])
    
    # Fix duration mismatches
    df['charging_duration'] = (df['end_plugout'] - df['start_plugin']).dt.total_seconds() / 3600
    duration_diff = abs(df['duration_hours'] - df['charging_duration'])
    df.loc[duration_diff > 0.1, 'duration_hours'] = df['charging_duration']
    df = df.drop('charging_duration', axis=1)
    
    return df

def save_parquet_partitioned(df, output_dir):
    """Save DataFrame as partitioned Parquet files"""
    # Create year/month columns for partitioning
    df['year'] = df['start_plugin'].dt.year
    df['month'] = df['start_plugin'].dt.month
    
    # Convert to absolute path if relative
    output_dir = PROJECT_ROOT / output_dir if not os.path.isabs(output_dir) else output_dir
    
    # Ensure output directory exists
    pathlib.Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as partitioned Parquet
    df.to_parquet(str(output_dir), index=False, engine='pyarrow',
                  compression='snappy', partition_cols=['year','month'])
    
    print(f"Partitioned Parquet saved at: {output_dir}")
    return output_dir

def upload_to_s3(local_path, bucket='ev-data'):
    """Upload files to LocalStack S3"""
    s3 = boto3.client('s3', 
                     endpoint_url="http://localhost:4566",
                     aws_access_key_id="test", 
                     aws_secret_access_key="test")
    
    # Create bucket if not exists
    try:
        s3.create_bucket(Bucket=bucket)
    except Exception:
        pass  # Bucket likely already exists
    
    # Convert to absolute path if relative
    local_path = PROJECT_ROOT / local_path if not os.path.isabs(local_path) else local_path
    
    # Upload files recursively
    folder = pathlib.Path(local_path)
    for file_path in folder.rglob('*.parquet'):
        relative_path = file_path.relative_to(folder)
        s3_key = f'raw/trondheim_partitioned/{relative_path.as_posix()}'
        s3.upload_file(str(file_path), bucket, s3_key)
        print(f"Uploaded: {s3_key}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process EV charging data from CSV to S3')
    parser.add_argument('--csv', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='data/processed/raw_parquet/partitionned',
                       help='Output directory for partitioned Parquet files')
    parser.add_argument('--bucket', default='ev-data', help='S3 bucket name')
    parser.add_argument('--upload', action='store_true', help='Upload to S3 after processing')
    
    args = parser.parse_args()
    
    # Execute pipeline
    print("Loading and cleaning CSV...")
    df = load_and_clean_csv(args.csv)
    
    print("Saving as partitioned Parquet...")
    output_path = save_parquet_partitioned(df, args.output)
    
    if args.upload:
        print("Uploading to S3...")
        upload_to_s3(output_path, args.bucket)
    
    print("✅ Pipeline completed successfully!")