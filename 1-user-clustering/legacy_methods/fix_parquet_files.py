import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
import os

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
ARTICLES_INPUT_FILE = 'articles_addressa.parquet'
BEHAVIORS_INPUT_FILE = 'behaviors_addressa.parquet'
ARTICLES_OUTPUT_FILE = 'articles_addressa_fixed.parquet'
BEHAVIORS_OUTPUT_FILE = 'behaviors_addressa_fixed.parquet'

def fix_parquet_file(input_file, output_file):
    """
    Attempt to fix a corrupted Parquet file by reading it with pandas
    and writing it back with pyarrow.
    """
    if not os.path.exists(input_file):
        logging.error(f"Input file '{input_file}' not found.")
        return False
    
    try:
        # Try to read the file with pandas
        logging.info(f"Reading {input_file}...")
        df = pd.read_parquet(input_file)
        logging.info(f"Successfully read {input_file} with {len(df)} rows.")
        
        # Write the file with pyarrow
        logging.info(f"Writing to {output_file}...")
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)
        logging.info(f"Successfully wrote to {output_file}.")
        
        return True
    except Exception as e:
        logging.error(f"Error processing {input_file}: {e}", exc_info=True)
        return False

def main():
    # Fix articles file
    if os.path.exists(ARTICLES_INPUT_FILE):
        logging.info(f"Attempting to fix articles file: {ARTICLES_INPUT_FILE}")
        fix_parquet_file(ARTICLES_INPUT_FILE, ARTICLES_OUTPUT_FILE)
    else:
        logging.warning(f"Articles file not found: {ARTICLES_INPUT_FILE}")
    
    # Fix behaviors file
    if os.path.exists(BEHAVIORS_INPUT_FILE):
        logging.info(f"Attempting to fix behaviors file: {BEHAVIORS_INPUT_FILE}")
        fix_parquet_file(BEHAVIORS_INPUT_FILE, BEHAVIORS_OUTPUT_FILE)
    else:
        logging.warning(f"Behaviors file not found: {BEHAVIORS_INPUT_FILE}")

if __name__ == "__main__":
    main() 