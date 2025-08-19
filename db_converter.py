# db_converter.py (Final Corrected Version)
import pandas as pd
import sqlite3
import os

CSV_FILE = "data/transactions-2025-08-09.csv" # Make sure this path is correct
DB_FILE = "properties.db"
TABLE_NAME = "properties"

def find_price_column(df_columns):
    """Intelligently finds the likely price column from a list of column names."""
    for col in df_columns:
        if 'trans_value' in col or 'price' in col or 'amount' in col or 'value' in col:
            return col
    return None

def create_database():
    """
    Reads data from a CSV file and loads it into a new SQLite database, ensuring correct data types.
    """
    if not os.path.exists(CSV_FILE):
        print(f"❌ ERROR: CSV file not found at '{CSV_FILE}'")
        return

    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"🗑️ Removed existing '{DB_FILE}' to rebuild it.")

    print(f"🚀 Reading data from '{CSV_FILE}'...")
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    print("\n✅ Found the following columns in your CSV:")
    print(df.columns.tolist())
    
    price_column_name = find_price_column(df.columns)
    
    if not price_column_name:
        print("\n❌ CRITICAL ERROR: Could not automatically find a price-related column.")
        return
        
    print(f"\n💡 Automatically detected '{price_column_name}' as the price column.")

    # --- THIS IS THE CRITICAL FIX ---
    # We force the price column to be a number, turning any non-numeric values into 'NaN' (Not a Number)
    print(f"🔄 Forcing the '{price_column_name}' column to a numeric data type...")
    df[price_column_name] = pd.to_numeric(df[price_column_name], errors='coerce')
    # We also drop any rows where the price could not be converted, as they are unusable.
    df.dropna(subset=[price_column_name], inplace=True)
    print("✅ Data type conversion complete.")

    conn = sqlite3.connect(DB_FILE)
    
    print(f"\n📝 Writing {len(df)} valid records to the '{TABLE_NAME}' table in '{DB_FILE}'...")
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
    
    print(f"⚡ Creating a database index on the '{price_column_name}' column for high-speed queries...")
    cursor = conn.cursor()
    cursor.execute(f'CREATE INDEX "idx_{price_column_name}" ON "{TABLE_NAME}" ("{price_column_name}");')
    
    conn.commit()
    conn.close()

    print(f"\n✅ Successfully created and populated '{DB_FILE}'. You can now run the data migrator.")

if __name__ == "__main__":
    create_database()