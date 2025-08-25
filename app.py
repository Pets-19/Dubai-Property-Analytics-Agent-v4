
# app.py (BigQuery Version)
import os
from flask import Flask, jsonify, render_template, request
import time
import json
import re
import pandas as pd
import google.generativeai as genai
from google.cloud import bigquery
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
BIGQUERY_PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID")
BIGQUERY_DATASET_ID = os.getenv("BIGQUERY_DATASET_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- AI Configuration ---
USE_AI_SUMMARY = True
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API key configured successfully.")
else:
    print("⚠️ Gemini API key not found. AI summary will be disabled.")
    USE_AI_SUMMARY = False

# --- BigQuery Client ---
bigquery_client = None
if BIGQUERY_PROJECT_ID and BIGQUERY_DATASET_ID:
    try:
        bigquery_client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
        # Test the connection
        query = f"SELECT 1 FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.properties` LIMIT 1"
        result = bigquery_client.query(query).result()
        print("✅ BigQuery connection test successful")
        print("✅ BigQuery client created successfully.")
    except Exception as e:
        print(f"❌ Failed to create BigQuery client: {e}")
        bigquery_client = None
else:
    print("❌ BigQuery configuration not found. The application cannot start.")

# --- Dynamic Column Mapping ---
def get_table_columns(table_name):
    if not bigquery_client: 
        return []
    try:
        table_ref = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{table_name}"
        table = bigquery_client.get_table(table_ref)
        return [field.name for field in table.schema]
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
    'property_type': find_column_name(SALES_COLUMNS, ['prop_type_en', 'prop_sub_type_en', 'property_type']),
    'bedrooms': find_column_name(SALES_COLUMNS, ['rooms_en', 'bedrooms']),
    'status': find_column_name(SALES_COLUMNS, ['is_offplan_en', 'development_status']),
    'area_name': find_column_name(SALES_COLUMNS, ['area_en']),
    'name': find_column_name(SALES_COLUMNS, ['project_en', 'procedure_en', 'property_name']),
}

# --- EXPANDED RENTAL COLUMN SEARCH ---
RENTALS_MAP = {
    'price': find_column_name(RENTALS_COLUMNS, ['annual_amount', 'annual_rent', 'rent_amount', 'amount', 'price', 'rent']),
    'property_type': find_column_name(RENTALS_COLUMNS, ['prop_type_en', 'prop_sub_type_en', 'property_type', 'type']),
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
    
    project_list_for_prompt = "" # Initialize
    project_insights_prompt_section = "" # Initialize

    if search_type == 'buy':
        analysis_subject, price_metric, user_goal, budget_key = "sales transactions", "sale price", "a potential buyer", "budget"
        query_text = " ".join([f for f in [filters.get("propertyType"), filters.get("bedrooms"), filters.get("status"), f"in {filters.get('area')}" if filters.get('area') else None] if f and 'Any' not in f and 'All' not in f]) or "all properties"
        
        # --- NEW: Extract project names for the prompt ---
        project_col = SALES_MAP.get('name')
        if project_col and project_col in results_df.columns:
            project_names = results_df[project_col].dropna().unique()
            project_names = [name for name in project_names if str(name).strip()]
            if project_names:
                sample_projects = project_names[:10]
                project_list_for_prompt = f"The data includes properties from various projects."
                # --- NEW: Add a section for Project Insights (Concise Version) ---
                project_insights_prompt_section = f"""
### Project Insights
In 2-3 sentences, summarize the key differences between the most prominent projects in the data. For example, mention which projects have higher transaction volumes (like DAMAC HILLS - CARSON), which ones offer more affordable units, and which ones feature larger apartments (like DAMAC HILLS - GOLF TERRACE).
"""

    else: # rent
        analysis_subject, price_metric, user_goal, budget_key = "rental contracts", "annual rent", "a potential renter", "budget"
        query_text = " ".join([f for f in [filters.get("propertyType"), filters.get("status"), f"in {filters.get('area')}" if filters.get('area') else None] if f and 'Any' not in f and 'All' not in f]) or "all properties"
    
    # Get budget value from either budget or annual_rent parameter
    budget_value = filters.get(budget_key) or filters.get('annual_rent') or 999999999
    
    prompt = f"""
    You are a professional Dubai real estate market analyst. Your goal is to provide concise, actionable insights to {user_goal} based on recent {analysis_subject}.
    The user is analyzing the market for: "{query_text}" with a maximum {price_metric} of {int(budget_value):,} AED.
    Our database found a total of {total_results} recent {analysis_subject} that match these criteria. {project_list_for_prompt}

    Please structure your response in two parts:

    ### Quick Summary
    Start with a clear opening sentence stating the total number of transactions found. Analyze the overall data to identify key insights, like the typical price range or popular areas.

    {project_insights_prompt_section}

    ### Recommendation
    Conclude with a direct, actionable tip for the user. For the 'buy' tab, if there are multiple projects, suggest using the 'Filter by Project Name' dropdown to explore specific developments.
    """
    try:
        data_sample = results_df.head(100).to_string(index=False)
        full_prompt = f"{prompt}\n\nHere is a representative sample of the data:\n{data_sample}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"❌ Gemini API call failed: {e}")
        return "An error occurred while generating the AI summary."

# --- Helper to build WHERE clauses for BigQuery ---
def build_where_clause(filters, map, price_key, is_rent=False):
    conditions, params = [], {}
    
    # Handle both 'budget' and 'annual_rent' parameter names - FIXED
    budget_value = filters.get('budget') or filters.get('annual_rent') or filters.get(price_key) or 999999999
    conditions.append(f"{map['price']} <= @budget_param")
    params['budget_param'] = int(budget_value)

    prop_type = filters.get('propertyType')
    if prop_type and 'All Types' not in prop_type:
        if is_rent:
            # For rentals, search in both PROP_TYPE_EN and PROP_SUB_TYPE_EN
            # Build an OR condition to search both columns
            prop_type_col = find_column_name(RENTALS_COLUMNS, ['prop_type_en'])
            prop_sub_type_col = find_column_name(RENTALS_COLUMNS, ['prop_sub_type_en'])
            
            if prop_type_col and prop_sub_type_col:
                conditions.append(f"({prop_type_col} = @prop_type OR {prop_sub_type_col} = @prop_type)")
            elif prop_type_col:
                conditions.append(f"{prop_type_col} = @prop_type")
            elif prop_sub_type_col:
                conditions.append(f"{prop_sub_type_col} = @prop_type")
        else:
            # For sales, use the standard property_type mapping
            conditions.append(f"{map['property_type']} = @prop_type")
        
        params['prop_type'] = prop_type

    # Only apply bedrooms filtering for sales (buy), not for rentals
    if not is_rent:
        bedrooms = filters.get('bedrooms')
        if bedrooms and 'Any' not in bedrooms:
            beds_col = map['bedrooms']
            if 'Studio' in bedrooms:
                conditions.append(f"({beds_col} = 'Studio' OR {beds_col} IS NULL)")
            else:
                try:
                    num_beds_str = re.findall(r'\d+', bedrooms)[0]
                    # Since bedrooms column might be empty, be more flexible
                    conditions.append(f"({beds_col} LIKE CONCAT('{num_beds_str}', ' %') OR {beds_col} LIKE CONCAT('%', '{num_beds_str}', '%'))")
                except (ValueError, IndexError): 
                    pass
    
    if not is_rent:
        status = filters.get('status')
        if status and 'Any' not in status:
            conditions.append(f"{map['status']} = @status")
            params['status'] = status
            
    area = filters.get('area')
    if area:
        conditions.append(f"{map['area_name']} LIKE @area")
        params['area'] = f"%{area}%"
    
    print(f"🔍 WHERE CLAUSE: {' AND '.join(conditions)}")
    print(f"🔍 PARAMS: {params}")
    return " AND ".join(conditions), params

# --- BUY TAB ENDPOINTS ---
@app.route('/search', methods=['POST'])
def search_buy():
    if not bigquery_client: 
        return jsonify({'summary': "BigQuery not configured.", 'data': []})
    
    data = request.json
    where_clause, params = build_where_clause(data, SALES_MAP, 'budget')
    
    count_query = f"""
        SELECT COUNT(*) as count 
        FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.properties` 
        WHERE {where_clause}
    """
    
    display_query = f"""
        SELECT * 
        FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.properties` 
        WHERE {where_clause} 
        LIMIT 500
    """
    
    try:
        # Configure query job with parameters
        query_parameters = []
        for name, value in params.items():
            if isinstance(value, int):
                query_parameters.append(bigquery.ScalarQueryParameter(name, 'INT64', value))
            else:
                query_parameters.append(bigquery.ScalarQueryParameter(name, 'STRING', value))
        
        job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
        
        # Get total count
        count_job = bigquery_client.query(count_query, job_config=job_config)
        count_result = list(count_job.result())
        total_results = count_result[0].count if count_result else 0
        
        # Get display results
        display_job = bigquery_client.query(display_query, job_config=job_config)
        results_df = display_job.to_dataframe()
        display_results_list = results_df.to_dict(orient='records')
        
    except Exception as e:
        print(f"❌ BUY SEARCH FAILED: {e}")
        total_results, results_df, display_results_list = 0, pd.DataFrame(), []

    ai_summary = generate_ai_summary(data, results_df, total_results, 'buy')
    return jsonify({'summary': ai_summary, 'data': display_results_list})

@app.route('/api/analytics', methods=['POST'])
def get_buy_analytics():
    if not bigquery_client: 
        return jsonify({'stats': {}})
    
    data = request.json
    where_clause, params = build_where_clause(data, SALES_MAP, 'budget')
    price_col = SALES_MAP['price']
    
    analytics_query = f"""
        SELECT 
            COUNT(*) as total_transactions, 
            SUM({price_col}) as total_volume, 
            AVG({price_col}) as average_price 
        FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.properties` 
        WHERE {where_clause}
    """
    
    try:
        query_parameters = []
        for name, value in params.items():
            if isinstance(value, int):
                query_parameters.append(bigquery.ScalarQueryParameter(name, 'INT64', value))
            else:
                query_parameters.append(bigquery.ScalarQueryParameter(name, 'STRING', value))
        
        job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
        
        job = bigquery_client.query(analytics_query, job_config=job_config)
        result = list(job.result())
        stats = dict(result[0]) if result else {}
        
    except Exception as e:
        print(f"❌ BUY ANALYTICS FAILED: {e}")
        stats = {}
    
    return jsonify({'stats': stats})

# --- RENT TAB ENDPOINTS ---
@app.route('/rent-search', methods=['POST'])
def search_rent():
    if not bigquery_client: 
        return jsonify({'summary': "BigQuery not configured.", 'data': []})
    
    data = request.json
    print(f"🔍 RENT SEARCH DATA: {data}")
    
    # Use any price key since we handle all in build_where_clause
    where_clause, params = build_where_clause(data, RENTALS_MAP, 'annual_rent', is_rent=True)
    
    count_query = f"""
        SELECT COUNT(*) as count 
        FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.rentals` 
        WHERE {where_clause}
    """
    
    display_query = f"""
        SELECT *, {RENTALS_MAP['price']} as trans_value 
        FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.rentals` 
        WHERE {where_clause} 
        LIMIT 500
    """
    
    total_results, results_df, display_results_list = 0, pd.DataFrame(), []
    
    try:
        # Configure query job with parameters
        query_parameters = []
        for name, value in params.items():
            if isinstance(value, int):
                query_parameters.append(bigquery.ScalarQueryParameter(name, 'INT64', value))
            else:
                query_parameters.append(bigquery.ScalarQueryParameter(name, 'STRING', value))
        
        job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
        
        # Get total count
        count_job = bigquery_client.query(count_query, job_config=job_config)
        count_result = list(count_job.result())
        total_results = count_result[0].count if count_result else 0
        
        # Get display results
        display_job = bigquery_client.query(display_query, job_config=job_config)
        results_df = display_job.to_dataframe()
        display_results_list = results_df.to_dict(orient='records')
        
        print(f"🔍 RENT RESULTS: {total_results} records found")
        
    except Exception as e:
        print(f"❌ RENT SEARCH FAILED: {e}")

    ai_summary = generate_ai_summary(data, results_df, total_results, 'rent')
    return jsonify({'summary': ai_summary, 'data': display_results_list})

@app.route('/api/rent-analytics', methods=['POST'])
def get_rent_analytics():
    if not bigquery_client: 
        return jsonify({'stats': {}})
    
    data = request.json
    print(f"🔍 RENT ANALYTICS DATA: {data}")
    
    # FIRST: Let's check what data we actually have
    try:
        # Check total records
        total_check = f"SELECT COUNT(*) as count FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.rentals`"
        total_job = bigquery_client.query(total_check)
        total_result = list(total_job.result())
        total_records = total_result[0].count if total_result else 0
        print(f"🔍 TOTAL RENTAL RECORDS: {total_records}")
        
        # Check distinct property types
        prop_check = f"""
            SELECT DISTINCT prop_sub_type_en 
            FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.rentals` 
            WHERE prop_sub_type_en IS NOT NULL 
            LIMIT 10
        """
        prop_job = bigquery_client.query(prop_check)
        prop_types = [row.prop_sub_type_en for row in prop_job.result()]
        print(f"🔍 SAMPLE PROPERTY TYPES: {prop_types}")
        
        # Check sample annual amounts
        amount_check = f"""
            SELECT annual_amount 
            FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.rentals` 
            WHERE annual_amount > 0 
            LIMIT 5
        """
        amount_job = bigquery_client.query(amount_check)
        amounts = [row.annual_amount for row in amount_job.result()]
        print(f"🔍 SAMPLE ANNUAL AMOUNTS: {amounts}")
        
    except Exception as e:
        print(f"🔍 DEBUG CHECK FAILED: {e}")
    
    # Use any price key since we handle all in build_where_clause
    where_clause, params = build_where_clause(data, RENTALS_MAP, 'annual_rent', is_rent=True)
    price_col = RENTALS_MAP['price']
    
    analytics_query = f"""
        SELECT 
            COUNT(*) as total_transactions, 
            SUM({price_col}) as total_volume, 
            AVG({price_col}) as average_price 
        FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.rentals` 
        WHERE {where_clause}
    """
    
    stats = {}
    
    try:
        query_parameters = []
        for name, value in params.items():
            if isinstance(value, int):
                query_parameters.append(bigquery.ScalarQueryParameter(name, 'INT64', value))
            else:
                query_parameters.append(bigquery.ScalarQueryParameter(name, 'STRING', value))
        
        job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
        
        job = bigquery_client.query(analytics_query, job_config=job_config)
        result = list(job.result())
        stats = dict(result[0]) if result else {}
        print(f"🔍 RENT ANALYTICS STATS: {stats}")
        
    except Exception as e:
        print(f"❌ RENT ANALYTICS FAILED: {e}")
    
    return jsonify({'stats': stats})

# --- GENERAL APP ROUTES ---
@app.route('/')
def home():
    return render_template('index.html', sales_map=SALES_MAP, rentals_map=RENTALS_MAP)

@app.route('/api/areas/<search_type>')
def get_areas(search_type):
    if not bigquery_client: 
        return jsonify([])
    
    table = 'properties' if search_type == 'buy' else 'rentals'
    area_col = SALES_MAP['area_name'] if search_type == 'buy' else RENTALS_MAP['area_name']
    
    try:
        query = f"""
            SELECT DISTINCT {area_col} as area
            FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{table}` 
            WHERE {area_col} IS NOT NULL 
            ORDER BY {area_col}
        """
        job = bigquery_client.query(query)
        areas = [row.area for row in job.result()]
        return jsonify(areas)
        
    except Exception as e:
        print(f"❌ AREAS FETCH FAILED for {search_type}: {e}")
        return jsonify([])

@app.route('/api/property-types/<search_type>')
def get_property_types(search_type):
    if not bigquery_client: 
        return jsonify([])
    
    # Only provide dynamic property types for rentals
    if search_type == 'rent':
        try:
            # Get unique values from both prop_type_en and prop_sub_type_en
            prop_type_query = f"""
                SELECT DISTINCT prop_type_en as type
                FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.rentals` 
                WHERE prop_type_en IS NOT NULL
            """
            
            prop_sub_type_query = f"""
                SELECT DISTINCT prop_sub_type_en as type
                FROM `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.rentals` 
                WHERE prop_sub_type_en IS NOT NULL
            """
            
            prop_job = bigquery_client.query(prop_type_query)
            prop_types = [row.type for row in prop_job.result()]
            
            prop_sub_job = bigquery_client.query(prop_sub_type_query)
            prop_sub_types = [row.type for row in prop_sub_job.result()]
            
            # Combine and remove duplicates
            all_types = list(set(prop_types + prop_sub_types))
            all_types.sort()
            
            return jsonify(all_types)
            
        except Exception as e:
            print(f"❌ PROPERTY TYPES FETCH FAILED for {search_type}: {e}")
            return jsonify(['Unit', 'Villa'])  # Fallback
    else:
        # For sales, return static options
        return jsonify(['Unit', 'Building', 'Land'])

if __name__ == '__main__':
    if not BIGQUERY_PROJECT_ID or not BIGQUERY_DATASET_ID:
        print("FATAL: BigQuery configuration is not set for local development.")
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)