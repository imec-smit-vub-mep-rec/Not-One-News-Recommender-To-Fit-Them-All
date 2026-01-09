#!/usr/bin/env python3

import pandas as pd
import argparse
import os
from pathlib import Path

def convert_csv_to_parquet(input_file: str, output_file: str = None) -> None:
    """
    Convert a CSV file to Parquet format.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to the output Parquet file. If not provided,
                                   will use the same name as input file with .parquet extension
    """
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        
        # If output file is not specified, create one based on input filename
        if output_file is None:
            input_path = Path(input_file)
            output_file = str(input_path.with_suffix('.parquet'))
        
        # Save as Parquet
        print(f"Converting to Parquet: {output_file}")
        df.to_parquet(output_file, index=False)
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert CSV file to Parquet format')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('-o', '--output', help='Path to the output Parquet file (optional)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return
    
    convert_csv_to_parquet(args.input_file, args.output)

if __name__ == "__main__":
    main() 