import argparse
import pandas as pd
import boto3
import pathlib
import os
import io

# Get the project root directory 
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()

def load_and_clean_csv(csv_path):

    # Convert to absolute path if relative
    csv_path = PROJECT_ROOT / csv_path if not os.path.isabs(csv_path) else csv_path
    
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8', 
                     parse_dates=['Start_plugin','End_plugout'],
                     decimal=',', dayfirst=True)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    return df

def save_parquet_partitioned(df, output_dir):

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

def upload_dataframe_to_s3(df, bucket='ev-data-raw', s3_key_prefix='ev-raw/'):
    
    # Initialize S3 client
    s3 = boto3.client('s3', 
                     endpoint_url="http://localhost:4566",
                     aws_access_key_id="test", 
                     aws_secret_access_key="test")
    
    # Create bucket if not exists
    try:
        s3.create_bucket(Bucket=bucket)
    except Exception:
        pass  # Bucket likely already exists
    
    # Create year/month columns for partitioning
    df['year'] = df['start_plugin'].dt.year
    df['month'] = df['start_plugin'].dt.month
    
    # Upload each partition directly to S3
    for (year, month), partition_df in df.groupby(['year', 'month']):
        # Create in-memory buffer
        buffer = io.BytesIO()
        
        # Write partition to buffer
        partition_df.to_parquet(buffer, index=False, engine='pyarrow', compression='snappy')
        
        # Reset buffer position
        buffer.seek(0)
        
        # Upload to S3
        s3_key = f"{s3_key_prefix}year={year}/month={month}/data.parquet"
        s3.upload_fileobj(buffer, bucket, s3_key)
        print(f"Uploaded: s3://{bucket}/{s3_key}")
    
    print(f"✅ All partitions uploaded directly to S3!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process EV charging data from CSV to S3')
    parser.add_argument('--csv', required=True, help='Input CSV file path')
    parser.add_argument('--bucket', default='ev-data-raw', help='S3 bucket name')
    parser.add_argument('--upload', action='store_true', help='Upload to S3 after processing')
    
    args = parser.parse_args()
    
    # Execute pipeline
    print("Loading CSV...")
    df = load_and_clean_csv(args.csv)
    
    if args.upload:
        print("Uploading to S3...")
        upload_dataframe_to_s3(df, args.bucket)
    else:
        output_path = save_parquet_partitioned(df, 'data/processed/trondheim_partitioned/')
        print(f"Data saved locally at: {output_path}")
    
    print("✅ Pipeline completed successfully!")