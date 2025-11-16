import argparse
import pandas as pd
import boto3
import pathlib

def load_and_clean_csv(csv_path):

    df = pd.read_csv(csv_path, sep=';', encoding='utf-8', 
                     parse_dates=['Start_plugin','End_plugout'],
                     decimal=',', dayfirst=True)
    
    return df

def save_parquet_simple(df, output_dir):

    # Ensure output directory exists
    pathlib.Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    
    output_file = f"{output_dir}/ingest.parquet"
    df.to_parquet(output_file, index=False, engine='pyarrow',
                  compression='snappy')
    
    print(f"Parquet saved at: {output_file}")
    return output_dir

def upload_to_s3(local_path, bucket='ev-data'):
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
    local_file = f"{local_path}/ingest.parquet"
    s3_key = 'parquets/ingest.parquet'
    s3.upload_file(local_file, bucket, s3_key)
    print(f"Uploaded: {s3_key}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ingestion pipeline from CSV to S3')
    parser.add_argument('--csv', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='data/raw',
                       help='Output directory for Parquet file')
    parser.add_argument('--bucket', default='ev-data', help='S3 bucket name')
    parser.add_argument('--upload', action='store_true', help='Upload to S3 after processing')
    
    args = parser.parse_args()
    
    # Execute pipeline
    print("Loading and cleaning CSV...")
    df = load_and_clean_csv(args.csv)
    
    print("Saving as Parquet...")
    output_path = save_parquet_simple(df, args.output)
    
    if args.upload:
        print("Uploading to S3...")
        upload_to_s3(output_path, args.bucket)
    
    print("✅ Pipeline completed successfully!")