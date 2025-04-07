import re
import os

# --- Configuration ---
ORIGINAL_SCRIPT = 'convert_addressa_to_ekstra_2.py'
FIXED_SCRIPT = 'convert_addressa_to_ekstra_2_fixed.py'

def fix_parquet_writing():
    """
    Fix the Parquet writing logic in the original script.
    """
    if not os.path.exists(ORIGINAL_SCRIPT):
        print(f"Original script '{ORIGINAL_SCRIPT}' not found.")
        return False
    
    try:
        # Read the original script
        with open(ORIGINAL_SCRIPT, 'r') as f:
            content = f.read()
        
        # Fix the Parquet writing logic
        # 1. Initialize writers at the beginning
        content = re.sub(
            r'# --- Use ParquetWriter for chunked writing ---',
            '# --- Use ParquetWriter for chunked writing ---\narticles_writer = None\nbehaviors_writer = None',
            content
        )
        
        # 2. Fix the chunk writing logic
        content = re.sub(
            r'if articles_writer is None:\s+articles_writer = pq\.ParquetWriter\(ARTICLES_OUTPUT_FILE, articles_table\.schema\)',
            'if articles_writer is None:\n            articles_writer = pq.ParquetWriter(ARTICLES_OUTPUT_FILE, articles_table.schema)\n        else:\n            # Close the previous writer and create a new one\n            articles_writer.close()\n            articles_writer = pq.ParquetWriter(ARTICLES_OUTPUT_FILE, articles_table.schema)',
            content
        )
        
        content = re.sub(
            r'if behaviors_writer is None:\s+behaviors_writer = pq\.ParquetWriter\(BEHAVIORS_OUTPUT_FILE, behaviors_table\.schema\)',
            'if behaviors_writer is None:\n            behaviors_writer = pq.ParquetWriter(BEHAVIORS_OUTPUT_FILE, behaviors_table.schema)\n        else:\n            # Close the previous writer and create a new one\n            behaviors_writer.close()\n            behaviors_writer = pq.ParquetWriter(BEHAVIORS_OUTPUT_FILE, behaviors_table.schema)',
            content
        )
        
        # 3. Fix the final chunk writing logic
        content = re.sub(
            r'if articles_writer is None:\s+articles_writer = pq\.ParquetWriter\(ARTICLES_OUTPUT_FILE, articles_table\.schema\)',
            'if articles_writer is None:\n            articles_writer = pq.ParquetWriter(ARTICLES_OUTPUT_FILE, articles_table.schema)\n        else:\n            # Close the previous writer and create a new one\n            articles_writer.close()\n            articles_writer = pq.ParquetWriter(ARTICLES_OUTPUT_FILE, articles_table.schema)',
            content
        )
        
        content = re.sub(
            r'if behaviors_writer is None:\s+behaviors_writer = pq\.ParquetWriter\(BEHAVIORS_OUTPUT_FILE, behaviors_table\.schema\)',
            'if behaviors_writer is None:\n            behaviors_writer = pq.ParquetWriter(BEHAVIORS_OUTPUT_FILE, behaviors_table.schema)\n        else:\n            # Close the previous writer and create a new one\n            behaviors_writer.close()\n            behaviors_writer = pq.ParquetWriter(BEHAVIORS_OUTPUT_FILE, behaviors_table.schema)',
            content
        )
        
        # Write the fixed script
        with open(FIXED_SCRIPT, 'w') as f:
            f.write(content)
        
        print(f"Fixed script written to '{FIXED_SCRIPT}'.")
        return True
    except Exception as e:
        print(f"Error fixing script: {e}")
        return False

if __name__ == "__main__":
    fix_parquet_writing() 