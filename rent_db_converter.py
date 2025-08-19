# rent_db_converter.py
import pandas as pd
import sqlite3
import os

# --- CONFIGURATION ---
# IMPORTANT: Make sure your new rental CSV is inside a 'data' folder
# and is named 'rental-transactions.csv'
RENTAL_CSV_FILE = "data/rental-transactions.csv" 
DB_FILE = "properties.db"
TABLE_NAME = "rentals"

def create_rental_table():
    """
    Reads data from the large rental CSV and loads it into a new 'rentals' table
    in the existing SQLite database. This is a one-time operation.
    """
    if not os.path.exists(RENTAL_CSV_FILE):
        print(f"❌ ERROR: Rental CSV file not found at '{RENTAL_CSV_FILE}'")
        print("Please make sure your rental data CSV is in the 'data' folder and named correctly.")
        return

    if not os.path.exists(DB_FILE):
        print(f"❌ ERROR: Main database file '{DB_FILE}' not found.")
        print("Please run the original 'db_converter.py' script first to create the database.")
        return

    print(f"🚀 Reading data from the large rental CSV at '{RENTAL_CSV_FILE}'...")
    # Use low_memory=False as recommended for large files with mixed types
    df = pd.read_csv(RENTAL_CSV_FILE, low_memory=False)
    
    # --- Data Cleaning ---
    # Make all column names lowercase and replace spaces with underscores for consistency
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    print(f"✅ Successfully read {len(df)} rental records.")
    
    # --- Data Type Correction (Crucial for Analytics) ---
    # We will look for the annual_amount column and ensure it is a number
    price_column = None
    for col in ['annual_amount', 'annual_rent', 'rent_value']:
        if col in df.columns:
            price_column = col
            break
            
    if price_column:
        print(f"🔄 Forcing the '{price_column}' column to a numeric data type...")
        df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
        # Drop any rows where the rent amount could not be converted
        df.dropna(subset=[price_column], inplace=True)
        print("✅ Data type conversion complete.")
    else:
        print("⚠️ Warning: Could not find a recognizable rent amount column. Analytics may not work correctly.")


    # --- Database Connection and Data Writing ---
    conn = sqlite3.connect(DB_FILE)
    
    print(f"\n📝 Writing {len(df)} valid rental records to the '{TABLE_NAME}' table in '{DB_FILE}'...")
    # Use chunksize for large datasets to manage memory efficiently
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False, chunksize=10000)
    
    # --- Create an index for performance ---
    # This will make filtering by rent amount much faster
    if price_column:
        print(f"⚡ Creating a database index on the '{price_column}' column for high-speed queries...")
        cursor = conn.cursor()
        cursor.execute(f'CREATE INDEX "idx_rent_{price_column}" ON "{TABLE_NAME}" ("{price_column}");')
    
    conn.commit()
    conn.close()

    print(f"\n✅ Phase 1 Complete! Successfully created and populated the '{TABLE_NAME}' table.")
    print("We are now ready to build the backend API for the rental data.")

if __name__ == "__main__":
    create_rental_table()