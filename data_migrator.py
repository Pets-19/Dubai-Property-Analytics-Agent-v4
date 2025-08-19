# data_migrator.py
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---
LOCAL_DB = "properties.db"
TABLE_NAME = "properties"

# Load the Neon database URL from your .env file
load_dotenv()
NEON_DB_URL = os.getenv("DATABASE_URL")

def migrate_data():
    """
    Reads data from a local SQLite database and writes it to a 
    PostgreSQL database on Neon. This is a one-time operation.
    """
    if not NEON_DB_URL:
        print("❌ ERROR: DATABASE_URL not found in your .env file.")
        print("Please add your Neon connection string to the .env file.")
        return

    # 1. Connect to the local SQLite database and read data into a DataFrame
    print(f"🔵 Reading data from local database '{LOCAL_DB}'...")
    try:
        conn_sqlite = sqlite3.connect(LOCAL_DB)
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn_sqlite)
        conn_sqlite.close()
        print(f"✅ Successfully read {len(df)} records.")
    except Exception as e:
        print(f"❌ Failed to read from SQLite database: {e}")
        return

    # 2. Connect to the new Neon (PostgreSQL) database
    print("\n🔵 Connecting to the Neon PostgreSQL database...")
    try:
        # The 'create_engine' command uses the URL to connect
        engine_pg = create_engine(NEON_DB_URL)
        print("✅ Connection successful.")
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        return

    # 3. Write the data to the new database
    print(f"\n🔵 Migrating {len(df)} records to the '{TABLE_NAME}' table in Neon...")
    try:
        # This command writes the DataFrame to a SQL table.
        # 'if_exists='replace'' will create the table and delete any old data.
        df.to_sql(TABLE_NAME, engine_pg, if_exists='replace', index=False, chunksize=1000)
        print("✅ Data migration completed successfully!")
        print("\nYou can now deploy your application on Render.")
    except Exception as e:
        print(f"❌ Failed to write data to PostgreSQL: {e}")

if __name__ == "__main__":
    # Add your Neon URL to your .env file before running!
    # It should look like this:
    # DATABASE_URL="postgres://..."
    
    # Create a temporary .env file for the migrator
    with open(".env", "w") as f:
        f.write(f'DATABASE_URL="{NEON_DB_URL}"\n')
        f.write(f'GEMINI_API_KEY="{os.getenv("GEMINI_API_KEY")}"\n')

    # Run the migration
    migrate_data()