# bigquery_migrator.py (Definitive Final Version)
import os
from dotenv import load_dotenv
from google.cloud import bigquery
import csv

# --- CONFIGURATION ---
load_dotenv()
PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID")
DATASET_ID = os.getenv("BIGQUERY_DATASET_ID")

# --- Schemas ---
PROPERTIES_SCHEMA = [
    bigquery.SchemaField("trans_group_en", "STRING"),
    bigquery.SchemaField("procedure_en", "STRING"),
    bigquery.SchemaField("trans_value", "FLOAT64"),
    bigquery.SchemaField("prop_type_en", "STRING"),
    bigquery.SchemaField("prop_sub_type_en", "STRING"),
    bigquery.SchemaField("rooms_en", "STRING"),
    bigquery.SchemaField("area_en", "STRING"),
    bigquery.SchemaField("usage_en", "STRING"),
    bigquery.SchemaField("reg_type_en", "STRING"),
    bigquery.SchemaField("is_offplan_en", "STRING"),
    bigquery.SchemaField("project_en", "STRING"),
]

RENTALS_SCHEMA = [
    bigquery.SchemaField("contract_reg_type", "STRING"),
    bigquery.SchemaField("annual_amount", "FLOAT64"),
    bigquery.SchemaField("area_en", "STRING"),
    bigquery.SchemaField("prop_type_en", "STRING"),
    bigquery.SchemaField("prop_sub_type_en", "STRING"),
    bigquery.SchemaField("rooms", "STRING"),
    bigquery.SchemaField("usage_en", "STRING"),
    bigquery.SchemaField("nearest_landmark_en", "STRING"),
    bigquery.SchemaField("nearest_metro_en", "STRING"),
]

SOURCES = {
    "properties": {
        "file": "data/transactions-2025-08-09.csv",
        "schema": PROPERTIES_SCHEMA
    },
    "rentals": {
        "file": "data/rental-transactions.csv",
        "schema": RENTALS_SCHEMA
    }
}

def migrate_to_bigquery(table_id, source_info):
    source_file_path = source_info["file"]
    schema = source_info["schema"]
    
    if not os.path.exists(source_file_path):
        print(f"❌ ERROR: CSV file not found at '{source_file_path}'. Skipping.")
        return

    try:
        client = bigquery.Client(project=PROJECT_ID)
        print("✅ BigQuery client initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize BigQuery client: {e}")
        return

    full_table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_id}"
    clean_file_path = f"data/clean_{table_id}.csv"
    schema_names = [field.name for field in schema]

    print(f"🧹 Cleaning '{source_file_path}'...")
    try:
        with open(source_file_path, mode='r', encoding='utf-8', errors='ignore') as infile, \
             open(clean_file_path, mode='w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
            
            original_header = [h.lower().replace(' ', '_') for h in next(reader)]
            writer.writerow(schema_names) # Write our clean header first
            
            col_indices = {name: original_header.index(name) for name in schema_names if name in original_header}
            
            # Create a more robust header detection system
            header_values = set(h.upper() for h in original_header)
            
            for row in reader:
                # Skip if the row is empty
                if not row:
                    continue
                
                # Skip if this row appears to be a duplicate header row
                # Check if any cell in this row matches our known header values
                row_upper = [str(cell).upper() for cell in row[:len(original_header)]]
                is_header_row = any(cell in header_values for cell in row_upper if cell.strip())
                
                if is_header_row:
                    continue

                clean_row = []
                for name in schema_names:
                    if name in col_indices and col_indices[name] < len(row):
                        clean_row.append(row[col_indices[name]])
                    else:
                        clean_row.append(None)
                writer.writerow(clean_row)

        print(f"✅ Successfully created clean file at '{clean_file_path}'.")
    except Exception as e:
        print(f"❌ Failed during CSV cleaning: {e}")
        return

    print(f"🚀 Preparing to upload '{table_id}'.")
    client.delete_table(full_table_id, not_found_ok=True)
    table = bigquery.Table(full_table_id, schema=schema)
    client.create_table(table)
    print(f"✅ Table '{table_id}' created successfully.")

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        allow_quoted_newlines=True,
        autodetect=False,
    )
    
    try:
        print(f"🔥 Starting load job from '{clean_file_path}'...")
        with open(clean_file_path, "rb") as source_file:
            job = client.load_table_from_file(source_file, full_table_id, job_config=job_config)
        
        job.result()
        
        destination_table = client.get_table(full_table_id)
        print(f"🎉 Success! Loaded {destination_table.num_rows} rows into '{table_id}'.")

    except Exception as e:
        print(f"❌ An error occurred during the upload: {e}")
        if 'job' in locals() and job.errors:
             print(f"  - BigQuery Job Errors: {job.errors}")
    finally:
        if os.path.exists(clean_file_path):
            os.remove(clean_file_path)
            print(f"🗑️ Cleaned up temporary file: {clean_file_path}")

if __name__ == "__main__":
    print("--- Starting BigQuery Data Migration ---")
    if not PROJECT_ID or not DATASET_ID:
         print("❌ ERROR: Your BigQuery project details are not in the .env file.")
    else:
        for table, info in SOURCES.items():
            print(f"\n--- Processing table: {table} ---")
            migrate_to_bigquery(table, info)
    print("\n--- Migration Process Finished ---")