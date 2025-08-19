# db_checker.py
import sqlite3
import os

DB_FILE = "properties.db"

def analyze_database():
    """
    Connects to the database and prints the unique values for key filter columns.
    """
    if not os.path.exists(DB_FILE):
        print(f"❌ ERROR: Database file '{DB_FILE}' not found.")
        print("Please make sure you have run 'db_converter.py' successfully first.")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    print("\n--- 🕵️ Database Content Analysis ---")

# --- Analysis for Area Type ---
    try:
        cursor.execute("SELECT DISTINCT area_en FROM properties;")
        results = [row[0] for row in cursor.fetchall()]
        print("\n✅ Unique Area Types Found:")
        for item in results:
            print(f"  - '{item}'")
    except sqlite3.OperationalError:
        print("\n⚠️ Could not find the 'area_en' column.")
    # --- Analysis for Property Type ---
    try:
        cursor.execute("SELECT DISTINCT prop_type_en FROM properties;")
        results = [row[0] for row in cursor.fetchall()]
        print("\n✅ Unique Property Types Found:")
        for item in results:
            print(f"  - '{item}'")
    except sqlite3.OperationalError:
        print("\n⚠️ Could not find the 'prop_type_en' column.")

    # --- Analysis for Bedrooms ---
    try:
        cursor.execute("SELECT DISTINCT rooms_en FROM properties;")
        results = [row[0] for row in cursor.fetchall()]
        print("\n✅ Unique Bedroom Styles Found:")
        # Print a limited number of examples if there are too many
        for item in results[:15]:
             print(f"  - '{item}'")
        if len(results) > 15:
            print("  - ... and more.")
    except sqlite3.OperationalError:
        print("\n⚠️ Could not find the 'rooms_en' column.")

    # --- Analysis for Development Status ---
    try:
        cursor.execute("SELECT DISTINCT is_offplan_en FROM properties;")
        results = [row[0] for row in cursor.fetchall()]
        print("\n✅ Unique Development Statuses Found:")
        for item in results:
            print(f"  - '{item}'")
    except sqlite3.OperationalError:
        print("\n⚠️ Could not find the 'is_offplan_en' column.")

    print("\n------------------------------------")
    conn.close()

if __name__ == "__main__":
    analyze_database()