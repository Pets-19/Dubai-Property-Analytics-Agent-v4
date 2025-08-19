# app.py (Corrected for Buy & Rent Tabs)
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
        engine = create_engine(DATABASE_URL)
        print("✅ Database engine created successfully.")
    except Exception as e:
        print(f"❌ Failed to create database engine: {e}")
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

SALES_MAP = {
    'price': find_column_name(SALES_COLUMNS, ['trans_value', 'price']),
    'property_type': find_column_name(SALES_COLUMNS, ['prop_type_en', 'property_type']),
    'bedrooms': find_column_name(SALES_COLUMNS, ['rooms_en', 'bedrooms']),
    'status': find_column_name(SALES_COLUMNS, ['is_offplan_en', 'development_status']),
    'area_name': find_column_name(SALES_COLUMNS, ['area_en']),
    'name': find_column_name(SALES_COLUMNS, ['procedure_en', 'property_name']),
}
RENTALS_MAP = {
    'price': find_column_name(RENTALS_COLUMNS, ['annual_amount', 'annual_rent']),
    'property_type': find_column_name(RENTALS_COLUMNS, ['property_sub_type_en', 'property_type']),
    'bedrooms': find_column_name(RENTALS_COLUMNS, ['rooms_en', 'bedrooms']),
    'area_name': find_column_name(RENTALS_COLUMNS, ['area_en']),
    'name': find_column_name(RENTALS_COLUMNS, ['project_en', 'name']),
}


app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Shared Logic ---
def generate_ai_summary(filters, results_df, total_results, search_type):
    if not USE_AI_SUMMARY or results_df.empty: return None
    
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    if search_type == 'buy':
        analysis_subject = "sales transactions"
        price_metric = "sale price"
        user_goal = "a potential buyer"
        budget_key = "budget"
    else: # rent
        analysis_subject = "rental contracts"
        price_metric = "annual rent"
        user_goal = "a potential renter"
        budget_key = "annual_rent"

    query_text = " ".join([f for f in [filters.get("propertyType"), filters.get("bedrooms"), filters.get("status"), f"in {filters.get('area')}" if filters.get('area') else None] if f and 'Any' not in f and 'All' not in f]) or "all properties"
    
    prompt = f"""
    You are a professional Dubai real estate market analyst. Your goal is to provide concise, actionable insights to {user_goal} based on recent {analysis_subject}.

    The user is analyzing the market for: "{query_text}" with a maximum {price_metric} of {int(filters.get(budget_key)):,} AED.

    Our database found a total of {total_results} recent {analysis_subject} that match these criteria. Based on an analysis of this data, generate a professional and helpful summary (around 80-100 words).
    - Start with a clear, confident opening stating the total number of contracts/transactions found.
    - Analyze the data to identify key insights, like the typical price range or popular areas.
    - Conclude with a direct, actionable tip for the user.
    - Tone: Be professional, insightful, and data-driven.
    """
    try:
        data_sample = results_df.head(100).to_string(index=False)
        full_prompt = f"{prompt}\n\nHere is a representative sample of the data:\n{data_sample}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"❌ Gemini API call failed: {e}")
        return "An error occurred while generating the AI summary."

# --- BUY TAB ENDPOINTS ---
@app.route('/search', methods=['POST'])
def search_buy():
    if not engine: return jsonify({'summary': "DB not configured.", 'data': []})
    data = request.json
    
    conditions, params = [], {}
    conditions.append(f"\"{SALES_MAP['price']}\" <= :budget")
    params['budget'] = int(data.get('budget', 999999999))
    if data.get('propertyType') and 'All Types' not in data.get('propertyType'):
        conditions.append(f"\"{SALES_MAP['property_type']}\" = :prop_type")
        params['prop_type'] = data.get('propertyType')
    if data.get('bedrooms') and 'Any' not in data.get('bedrooms'):
        beds_col = f"\"{SALES_MAP['bedrooms']}\""
        if 'Studio' in data.get('bedrooms'):
            conditions.append(f"{beds_col} = 'Studio'")
        else:
            try:
                num_beds_str = re.findall(r'\d+', data.get('bedrooms'))[0]
                conditions.append(f"{beds_col} LIKE '{num_beds_str} %'")
            except (ValueError, IndexError): pass
    if data.get('status') and 'Any' not in data.get('status'):
        conditions.append(f"\"{SALES_MAP['status']}\" = :status")
        params['status'] = data.get('status')
    if data.get('area'):
        conditions.append(f"\"{SALES_MAP['area_name']}\" LIKE :area")
        params['area'] = f"%{data.get('area')}%"

    where_clause = " AND ".join(conditions)
    
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
    conditions, params = [], {}
    conditions.append(f"\"{SALES_MAP['price']}\" <= :budget")
    params['budget'] = int(data.get('budget', 999999999))
    if data.get('propertyType') and 'All Types' not in data.get('propertyType'):
        conditions.append(f"\"{SALES_MAP['property_type']}\" = :prop_type")
        params['prop_type'] = data.get('propertyType')
    if data.get('bedrooms') and 'Any' not in data.get('bedrooms'):
        beds_col = f"\"{SALES_MAP['bedrooms']}\""
        if 'Studio' in data.get('bedrooms'):
            conditions.append(f"{beds_col} = 'Studio'")
        else:
            try:
                num_beds_str = re.findall(r'\d+', data.get('bedrooms'))[0]
                conditions.append(f"{beds_col} LIKE '{num_beds_str} %'")
            except (ValueError, IndexError): pass
    if data.get('status') and 'Any' not in data.get('status'):
        conditions.append(f"\"{SALES_MAP['status']}\" = :status")
        params['status'] = data.get('status')
    if data.get('area'):
        conditions.append(f"\"{SALES_MAP['area_name']}\" LIKE :area")
        params['area'] = f"%{data.get('area')}%"
    
    where_clause = " AND ".join(conditions)
    
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
    
    conditions, params = [], {}
    conditions.append(f"\"{RENTALS_MAP['price']}\" <= :annual_rent")
    params['annual_rent'] = int(data.get('annual_rent', 9999999))
    if data.get('propertyType') and 'All Types' not in data.get('propertyType'):
        conditions.append(f"\"{RENTALS_MAP['property_type']}\" = :prop_type")
        params['prop_type'] = data.get('propertyType')
    if data.get('bedrooms') and 'Any' not in data.get('bedrooms'):
        beds_col = f"\"{RENTALS_MAP['bedrooms']}\""
        if 'Studio' in data.get('bedrooms'):
            conditions.append(f"{beds_col} = 'Studio'")
        else:
            try:
                num_beds_str = re.findall(r'\d+', data.get('bedrooms'))[0]
                conditions.append(f"{beds_col} LIKE '{num_beds_str} %'")
            except (ValueError, IndexError): pass
    if data.get('area'):
        conditions.append(f"\"{RENTALS_MAP['area_name']}\" LIKE :area")
        params['area'] = f"%{data.get('area')}%"
    
    where_clause = " AND ".join(conditions)
    
    count_query = text(f"SELECT COUNT(*) FROM rentals WHERE {where_clause};")
    display_query = text(f"SELECT *, annual_amount as trans_value FROM rentals WHERE {where_clause} LIMIT 500;")
    
    with engine.connect() as conn:
        try:
            total_results = conn.execute(count_query, params).scalar_one()
            results_df = pd.read_sql_query(display_query, conn, params=params)
            display_results_list = results_df.to_dict(orient='records')
        except Exception as e:
            print(f"❌ RENT SEARCH FAILED: {e}")
            total_results, results_df, display_results_list = 0, pd.DataFrame(), []

    ai_summary = generate_ai_summary(data, results_df, total_results, 'rent')
    return jsonify({'summary': ai_summary, 'data': display_results_list})


@app.route('/api/rent-analytics', methods=['POST'])
def get_rent_analytics():
    if not engine: return jsonify({'stats': {}})
    data = request.json
    
    conditions, params = [], {}
    conditions.append(f"\"{RENTALS_MAP['price']}\" <= :annual_rent")
    params['annual_rent'] = int(data.get('annual_rent', 9999999))
    if data.get('propertyType') and 'All Types' not in data.get('propertyType'):
        conditions.append(f"\"{RENTALS_MAP['property_type']}\" = :prop_type")
        params['prop_type'] = data.get('propertyType')
    if data.get('bedrooms') and 'Any' not in data.get('bedrooms'):
        beds_col = f"\"{RENTALS_MAP['bedrooms']}\""
        if 'Studio' in data.get('bedrooms'):
            conditions.append(f"{beds_col} = 'Studio'")
        else:
            try:
                num_beds_str = re.findall(r'\d+', data.get('bedrooms'))[0]
                conditions.append(f"{beds_col} LIKE '{num_beds_str} %'")
            except (ValueError, IndexError): pass
    if data.get('area'):
        conditions.append(f"\"{RENTALS_MAP['area_name']}\" LIKE :area")
        params['area'] = f"%{data.get('area')}%"

    where_clause = " AND ".join(conditions)
    
    price_col = f"\"{RENTALS_MAP['price']}\""
    analytics_query = text(f"SELECT COUNT(*) as total_transactions, SUM({price_col}) as total_volume, AVG({price_col}) as average_price FROM rentals WHERE {where_clause};")
    
    with engine.connect() as conn:
        try:
            result = conn.execute(analytics_query, params)
            stats_raw = result.fetchone()
            stats = dict(stats_raw._mapping) if stats_raw else {}
        except Exception as e:
            print(f"❌ RENT ANALYTICS FAILED: {e}")
            stats = {}
    return jsonify({'stats': stats})


# --- GENERAL APP ROUTES ---
@app.route('/')
def home():
    # THIS IS THE CRITICAL FIX
    # The home route must pass both maps to the template
    return render_template('index.html', sales_map=SALES_MAP, rentals_map=RENTALS_MAP)

@app.route('/api/areas/<search_type>')
def get_areas(search_type):
    if not engine: return jsonify([])
    
    table = 'properties' if search_type == 'buy' else 'rentals'
    area_col = SALES_MAP['area_name'] if search_type == 'buy' else RENTALS_MAP['area_name']
    
    with engine.connect() as conn:
        query = text(f"SELECT DISTINCT \"{area_col}\" FROM {table} WHERE \"{area_col}\" IS NOT NULL ORDER BY \"{area_col}\";")
        result = conn.execute(query)
        areas = [row[0] for row in result]
    return jsonify(areas)

if __name__ == '__main__':
    if not DATABASE_URL:
        print("FATAL: DATABASE_URL is not set for local development.")
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)