# app.py (Final Corrected Version - Rental Column Fix)
import os
import sqlite3
from flask import Flask, jsonify, render_template, request
import time
import json
import re
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# --- Configuration ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- AI Configuration ---
USE_AI_SUMMARY = True
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API key configured successfully.")
else:
    print("⚠️ Gemini API key not found. AI summary will be disabled.")
    USE_AI_SUMMARY = False

# --- Database Engine ---
engine = None
if DATABASE_URL:
    try:
        # Connection settings optimized for Neon serverless
        engine = create_engine(
            DATABASE_URL,
            echo=True,  # Enable debug logging
            pool_pre_ping=True,  # Enable connection health checks
            pool_size=2,  # Reduced pool size
            max_overflow=5,  # Reduced max overflow
            pool_timeout=30,
            pool_recycle=1800,  # Recycle connections every 30 minutes
            connect_args={
                'sslmode': 'require',  # Force SSL mode
                'connect_timeout': 10,
                'keepalives': 1,       # Enable keepalive
                'keepalives_idle': 30,  # Idle time before sending keepalive
                'keepalives_interval': 10,  # Interval between keepalives
                'keepalives_count': 5   # Max number of keepalive retries
            }
        )
        # Test the connection immediately
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection test successful")
        print("✅ Database engine created successfully.")
    except Exception as e:
        print(f"❌ Failed to create database engine: {e}")
        # Try alternative connection string format
        try:
            if 'postgresql://' in DATABASE_URL:
                alt_url = DATABASE_URL.replace('postgresql://', 'postgresql+psycopg2://')
                engine = create_engine(
                    alt_url,
                    pool_pre_ping=True,
                    pool_size=2,
                    connect_args={'sslmode': 'require'}
                )
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                print("✅ Database connection successful with alternative URL")
        except Exception as e2:
            print(f"❌ Alternative connection also failed: {e2}")
else:
    print("❌ DATABASE_URL not found. The application cannot start.")

# --- Dynamic Column Mapping ---
def get_table_columns(table_name):
    if not engine: return []
    try:
        with engine.connect() as conn:
            from sqlalchemy.inspection import inspect
            inspector = inspect(engine)
            return [col['name'] for col in inspector.get_columns(table_name)]
    except Exception as e:
        print(f"⚠️ Could not inspect columns for table '{table_name}': {e}")
        return []

def find_column_name(all_columns, potential_matches):
    for match in potential_matches:
        if match in all_columns:
            return match
    return potential_matches[0]

SALES_COLUMNS = get_table_columns('properties')
RENTALS_COLUMNS = get_table_columns('rentals')

print(f"🔍 SALES COLUMNS: {SALES_COLUMNS}")
print(f"🔍 RENTALS COLUMNS: {RENTALS_COLUMNS}")

SALES_MAP = {
    'price': find_column_name(SALES_COLUMNS, ['trans_value', 'price']),
    'property_type': find_column_name(SALES_COLUMNS, ['prop_type_en', 'property_type']),
    'bedrooms': find_column_name(SALES_COLUMNS, ['rooms_en', 'bedrooms']),
    'status': find_column_name(SALES_COLUMNS, ['is_offplan_en', 'development_status']),
    'area_name': find_column_name(SALES_COLUMNS, ['area_en']),
    'name': find_column_name(SALES_COLUMNS, ['procedure_en', 'property_name']),
}

# --- EXPANDED RENTAL COLUMN SEARCH ---
RENTALS_MAP = {
    'price': find_column_name(RENTALS_COLUMNS, ['annual_amount', 'annual_rent', 'rent_amount', 'amount', 'price', 'rent']),
    'property_type': find_column_name(RENTALS_COLUMNS, ['prop_sub_type_en', 'property_type', 'prop_type_en', 'type']),
    'bedrooms': find_column_name(RENTALS_COLUMNS, ['rooms', 'rooms_en', 'bedrooms', 'bedroom']),
    'area_name': find_column_name(RENTALS_COLUMNS, ['area_en', 'area', 'location']),
    'name': find_column_name(RENTALS_COLUMNS, ['project_en', 'name', 'property_name']),
}

print(f"🔍 SALES MAP: {SALES_MAP}")
print(f"🔍 RENTALS MAP: {RENTALS_MAP}")


app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Shared Logic ---
def generate_ai_summary(filters, results_df, total_results, search_type):
    if not USE_AI_SUMMARY or results_df.empty: return None
    
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    if search_type == 'buy':
        analysis_subject, price_metric, user_goal, budget_key = "sales transactions", "sale price", "a potential buyer", "budget"
    else: # rent
        analysis_subject, price_metric, user_goal, budget_key = "rental contracts", "annual rent", "a potential renter", "budget"

    query_text = " ".join([f for f in [filters.get("propertyType"), filters.get("bedrooms"), filters.get("status"), f"in {filters.get('area')}" if filters.get('area') else None] if f and 'Any' not in f and 'All' not in f]) or "all properties"
    
    # Get budget value from either budget or annual_rent parameter
    budget_value = filters.get(budget_key) or filters.get('annual_rent') or 999999999
    
    prompt = f"""
    You are a professional Dubai real estate market analyst. Your goal is to provide concise, actionable insights to {user_goal} based on recent {analysis_subject}.
    The user is analyzing the market for: "{query_text}" with a maximum {price_metric} of {int(budget_value):,} AED.
    Our database found a total of {total_results} recent {analysis_subject} that match these criteria. Based on an analysis of this data, generate a professional and helpful summary.
    - Start with a clear opening sentence stating the total number of contracts/transactions found.
    - Analyze the data to identify key insights, like the typical price range or popular areas.
    - Conclude with a direct, actionable tip for the user.
    """
    try:
        data_sample = results_df.head(100).to_string(index=False)
        full_prompt = f"{prompt}\n\nHere is a representative sample of the data:\n{data_sample}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"❌ Gemini API call failed: {e}")
        return "An error occurred while generating the AI summary."

# --- Helper to build WHERE clauses ---
def build_where_clause(filters, map, price_key, is_rent=False):
    conditions, params = [], {}
    
    # Handle both 'budget' and 'annual_rent' parameter names - FIXED
    budget_value = filters.get('budget') or filters.get('annual_rent') or filters.get(price_key) or 999999999
    conditions.append(f"\"{map['price']}\" <= :budget_param")
    params['budget_param'] = int(budget_value)

    prop_type = filters.get('propertyType')
    if prop_type and 'All Types' not in prop_type:
        # Fix property type mapping - frontend sends "Unit" but data has "Flat"
        if prop_type == 'Unit':
            prop_type = 'Flat'  # Map frontend "Unit" to database "Flat"
        conditions.append(f"\"{map['property_type']}\" = :prop_type")
        params['prop_type'] = prop_type

    bedrooms = filters.get('bedrooms')
    if bedrooms and 'Any' not in bedrooms:
        beds_col = f"\"{map['bedrooms']}\""
        if 'Studio' in bedrooms:
            conditions.append(f"({beds_col} = 'Studio' OR {beds_col} IS NULL)")
        else:
            try:
                num_beds_str = re.findall(r'\d+', bedrooms)[0]
                # Since ROOMS column might be empty, be more flexible
                conditions.append(f"({beds_col} LIKE '{num_beds_str} %' OR {beds_col} LIKE '%{num_beds_str}%')")
            except (ValueError, IndexError): 
                pass
    
    if not is_rent:
        status = filters.get('status')
        if status and 'Any' not in status:
            conditions.append(f"\"{map['status']}\" = :status")
            params['status'] = status
            
    area = filters.get('area')
    if area:
        conditions.append(f"\"{map['area_name']}\" LIKE :area")
        params['area'] = f"%{area}%"
    
    print(f"🔍 WHERE CLAUSE: {' AND '.join(conditions)}")
    print(f"🔍 PARAMS: {params}")
    return " AND ".join(conditions), params

# --- BUY TAB ENDPOINTS ---
@app.route('/search', methods=['POST'])
def search_buy():
    if not engine: return jsonify({'summary': "DB not configured.", 'data': []})
    data = request.json
    where_clause, params = build_where_clause(data, SALES_MAP, 'budget')
    
    count_query = text(f"SELECT COUNT(*) FROM properties WHERE {where_clause};")
    display_query = text(f"SELECT * FROM properties WHERE {where_clause} LIMIT 500;")
    
    with engine.connect() as conn:
        try:
            total_results = conn.execute(count_query, params).scalar_one()
            results_df = pd.read_sql_query(display_query, conn, params=params)
            display_results_list = results_df.to_dict(orient='records')
        except Exception as e:
            print(f"❌ BUY SEARCH FAILED: {e}")
            total_results, results_df, display_results_list = 0, pd.DataFrame(), []

    ai_summary = generate_ai_summary(data, results_df, total_results, 'buy')
    return jsonify({'summary': ai_summary, 'data': display_results_list})

@app.route('/api/analytics', methods=['POST'])
def get_buy_analytics():
    if not engine: return jsonify({'stats': {}})
    data = request.json
    where_clause, params = build_where_clause(data, SALES_MAP, 'budget')
    price_col = f"\"{SALES_MAP['price']}\""
    analytics_query = text(f"SELECT COUNT(*) as total_transactions, SUM({price_col}) as total_volume, AVG({price_col}) as average_price FROM properties WHERE {where_clause};")
    
    with engine.connect() as conn:
        try:
            result = conn.execute(analytics_query, params)
            stats_raw = result.fetchone()
            stats = dict(stats_raw._mapping) if stats_raw else {}
        except Exception as e:
            print(f"❌ BUY ANALYTICS FAILED: {e}")
            stats = {}
    return jsonify({'stats': stats})

# --- RENT TAB ENDPOINTS ---
@app.route('/rent-search', methods=['POST'])
def search_rent():
    if not engine: return jsonify({'summary': "DB not configured.", 'data': []})
    data = request.json
    print(f"🔍 RENT SEARCH DATA: {data}")
    
    # Use any price key since we handle all in build_where_clause
    where_clause, params = build_where_clause(data, RENTALS_MAP, 'annual_rent', is_rent=True)
    
    count_query = text(f"SELECT COUNT(*) FROM rentals WHERE {where_clause};")
    display_query = text(f"SELECT *, {RENTALS_MAP['price']} as trans_value FROM rentals WHERE {where_clause} LIMIT 500;")
    
    # Retry logic for database connection
    max_retries = 3
    retry_count = 0
    total_results, results_df, display_results_list = 0, pd.DataFrame(), []
    
    while retry_count < max_retries:
        try:
            # Create a new connection for this request
            conn = engine.connect()
            try:
                total_results = conn.execute(count_query, params).scalar_one()
                results_df = pd.read_sql_query(display_query, conn, params=params)
                display_results_list = results_df.to_dict(orient='records')
                print(f"🔍 RENT RESULTS: {total_results} records found")
                conn.close()
                break
            except Exception as e:
                print(f"❌ RENT SEARCH FAILED (attempt {retry_count + 1}): {e}")
                conn.close()
                retry_count += 1
        except Exception as e:
            print(f"❌ CONNECTION FAILED (attempt {retry_count + 1}): {e}")
            retry_count += 1
            time.sleep(1)  # Wait before retrying

    ai_summary = generate_ai_summary(data, results_df, total_results, 'rent')
    return jsonify({'summary': ai_summary, 'data': display_results_list})

@app.route('/api/rent-analytics', methods=['POST'])
def get_rent_analytics():
    if not engine: return jsonify({'stats': {}})
    data = request.json
    print(f"🔍 RENT ANALYTICS DATA: {data}")
    
    # FIRST: Let's check what data we actually have
    try:
        conn = engine.connect()
        # Check total records
        total_check = text("SELECT COUNT(*) FROM rentals;")
        total_records = conn.execute(total_check).scalar_one()
        print(f"🔍 TOTAL RENTAL RECORDS: {total_records}")
        
        # Check distinct property types
        prop_check = text("SELECT DISTINCT \"PROP_SUB_TYPE_EN\" FROM rentals WHERE \"PROP_SUB_TYPE_EN\" IS NOT NULL LIMIT 10;")
        prop_result = conn.execute(prop_check)
        prop_types = [row[0] for row in prop_result]
        print(f"🔍 SAMPLE PROPERTY TYPES: {prop_types}")
        
        # Check distinct room types
        room_check = text("SELECT DISTINCT \"ROOMS\" FROM rentals WHERE \"ROOMS\" IS NOT NULL AND \"ROOMS\" != '' LIMIT 10;")
        room_result = conn.execute(room_check)
        room_types = [row[0] for row in room_result]
        print(f"🔍 SAMPLE ROOM TYPES: {room_types}")
        
        # Check sample annual amounts
        amount_check = text("SELECT \"ANNUAL_AMOUNT\" FROM rentals WHERE \"ANNUAL_AMOUNT\" > 0 LIMIT 5;")
        amount_result = conn.execute(amount_check)
        amounts = [row[0] for row in amount_result]
        print(f"🔍 SAMPLE ANNUAL AMOUNTS: {amounts}")
        
        conn.close()
    except Exception as e:
        print(f"🔍 DEBUG CHECK FAILED: {e}")
    
    # Use any price key since we handle all in build_where_clause
    where_clause, params = build_where_clause(data, RENTALS_MAP, 'annual_rent', is_rent=True)
    price_col = f"\"{RENTALS_MAP['price']}\""
    analytics_query = text(f"SELECT COUNT(*) as total_transactions, SUM({price_col}) as total_volume, AVG({price_col}) as average_price FROM rentals WHERE {where_clause};")
    
    # Retry logic for database connection
    max_retries = 3
    retry_count = 0
    stats = {}
    
    while retry_count < max_retries:
        try:
            # Create a new connection for this request
            conn = engine.connect()
            try:
                result = conn.execute(analytics_query, params)
                stats_raw = result.fetchone()
                stats = dict(stats_raw._mapping) if stats_raw else {}
                print(f"🔍 RENT ANALYTICS STATS: {stats}")
                conn.close()
                break
            except Exception as e:
                print(f"❌ RENT ANALYTICS FAILED (attempt {retry_count + 1}): {e}")
                conn.close()
                retry_count += 1
        except Exception as e:
            print(f"❌ CONNECTION FAILED (attempt {retry_count + 1}): {e}")
            retry_count += 1
            time.sleep(1)  # Wait before retrying
    
    return jsonify({'stats': stats})

# --- GENERAL APP ROUTES ---
@app.route('/')
def home():
    return render_template('index.html', sales_map=SALES_MAP, rentals_map=RENTALS_MAP)

@app.route('/api/areas/<search_type>')
def get_areas(search_type):
    if not engine: return jsonify([])
    table = 'properties' if search_type == 'buy' else 'rentals'
    area_col = SALES_MAP['area_name'] if search_type == 'buy' else RENTALS_MAP['area_name']
    
    # Retry logic for database connection
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Create a new connection for this request
            conn = engine.connect()
            try:
                query = text(f"SELECT DISTINCT \"{area_col}\" FROM {table} WHERE \"{area_col}\" IS NOT NULL ORDER BY \"{area_col}\";")
                result = conn.execute(query)
                areas = [row[0] for row in result]
                conn.close()
                return jsonify(areas)
            except Exception as e:
                print(f"❌ AREAS FETCH FAILED for {search_type} (attempt {retry_count + 1}): {e}")
                conn.close()
                retry_count += 1
        except Exception as e:
            print(f"❌ CONNECTION FAILED (attempt {retry_count + 1}): {e}")
            retry_count += 1
            time.sleep(1)  # Wait before retrying
    
    return jsonify([])

if __name__ == '__main__':
    if not DATABASE_URL:
        print("FATAL: DATABASE_URL is not set for local development.")
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)