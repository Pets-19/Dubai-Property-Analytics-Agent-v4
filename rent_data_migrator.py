# rent_data_migrator.py
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---
RENTAL_CSV_FILE = "data/rental-transactions.csv" 
TABLE_NAME = "rentals"

# Load the Neon database URL from your .env file
load_dotenv()
NEON_DB_URL = os.getenv("DATABASE_URL")

def migrate_rental_data_to_cloud():
    """
    Reads data from the rental CSV and writes it directly to a new 'rentals' table
    in the live PostgreSQL database on Neon.
    """
    if not NEON_DB_URL:
        print("❌ ERROR: DATABASE_URL not found in your .env file.")
        print("Please ensure your Neon connection string is in the .env file.")
        return

    if not os.path.exists(RENTAL_CSV_FILE):
        print(f"❌ ERROR: Rental CSV file not found at '{RENTAL_CSV_FILE}'")
        print("Please make sure your rental CSV is in the 'data' folder and named correctly.")
        return

    # 1. Read the rental CSV into a DataFrame
    print(f"🚀 Reading data from the large rental CSV at '{RENTAL_CSV_FILE}'...")
    df = pd.read_csv(RENTAL_CSV_FILE, low_memory=False)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    print(f"✅ Successfully read {len(df)} rental records.")

    # 2. Data Type Correction
    price_column = 'annual_amount' # Assuming this is the correct column name
    if price_column in df.columns:
        print(f"🔄 Forcing the '{price_column}' column to a numeric data type...")
        df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
        df.dropna(subset=[price_column], inplace=True)
        print("✅ Data type conversion complete.")
    else:
        print(f"⚠️ Warning: Column '{price_column}' not found. Analytics may not work correctly.")

    # 3. Connect to the live Neon (PostgreSQL) database
    print("\n🔵 Connecting to the Neon PostgreSQL database...")
    try:
        engine_pg = create_engine(NEON_DB_URL)
        print("✅ Connection successful.")
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        return

    # 4. Write the data to the new 'rentals' table in the cloud
    print(f"\n🔵 Migrating {len(df)} valid rental records to the '{TABLE_NAME}' table in Neon...")
    try:
        df.to_sql(TABLE_NAME, engine_pg, if_exists='replace', index=False, chunksize=10000)
        print("✅ Data migration completed successfully!")
        print("\nYour 'rentals' table is now live in your Neon database.")
    except Exception as e:
        print(f"❌ Failed to write data to PostgreSQL: {e}")

if __name__ == "__main__":
    migrate_rental_data_to_cloud()