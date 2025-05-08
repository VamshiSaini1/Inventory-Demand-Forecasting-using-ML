# sales_forecast_app.py

import streamlit as st
import pandas as pd
import mysql.connector
import pickle
import xgboost as xgb
from datetime import datetime

# --- Database connection ---
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Application@123",
        database="Saini_Distributions"
    )

# --- Load the model ---
@st.cache_resource
def load_model():
    with open("xgboost_model.pkl", "rb") as f:
        return pickle.load(f)

# --- Load metadata ---
@st.cache_data
def load_metadata():
    sales_df = pd.read_csv("train.csv", parse_dates=['date'])
    products_df = pd.read_csv("products_table.csv")
    stores_df = pd.read_csv("stores_table.csv")
    sales_reps_df = pd.read_csv("sales_reps_table.csv")
    return sales_df, products_df, stores_df, sales_reps_df

# --- Get stores by sales rep ID ---
def get_stores_for_rep(rep_id, db):
    cursor = db.cursor()
    cursor.execute("SELECT store_id, store_name FROM stores WHERE salesrep_id = %s", (rep_id,))
    result = cursor.fetchall()
    cursor.close()
    return result

# --- Generate forecast for a specific store dynamically ---
def generate_forecast_for_store(model, store_id, sales_df, products_df, stores_df, sales_reps_df, target_date):
    # Merge and prepare data
    df = sales_df.merge(products_df, left_on='item', right_on='product_id', how='left')
    df = df.merge(stores_df, left_on='store', right_on='store_id', how='left')
    df = df.merge(sales_reps_df, on='salesrep_id', how='left')
    df.drop(columns=['product_id', 'store_id'], inplace=True)
    df.sort_values(by=['store', 'item', 'date'], inplace=True)

    # Feature engineering
    df['lag_1'] = df.groupby(['store', 'item'])['sales'].shift(1)
    df['lag_7'] = df.groupby(['store', 'item'])['sales'].shift(7)
    df['lag_14'] = df.groupby(['store', 'item'])['sales'].shift(14)
    df['rolling_mean_7'] = df.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(1).rolling(7).mean())
    df['rolling_std_7'] = df.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(1).rolling(7).std())
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month.astype('category')
    df['day_of_week'] = df['date'].dt.dayofweek.astype('category')

    df = pd.get_dummies(df, columns=['month', 'day_of_week', 'store_name', 'product_name', 'name'], drop_first=True)

    # Use latest date from training data for now (pretend it's today)
    latest_date = df['date'].max()
    pred_input = df[(df['date'] == latest_date) & (df['store'] == store_id)].copy()

    X_pred = pred_input.drop(columns=['sales', 'date', 'email', 'phone', 'location'], errors='ignore')
    pred_input['predicted_sales'] = model.predict(X_pred)

    # Attach product names and ids
    result = pred_input[['item', 'predicted_sales']].merge(products_df, left_on='item', right_on='product_id', how='left')
    return result[['product_id', 'product_name', 'predicted_sales']].sort_values(by='predicted_sales', ascending=False)

# --- UI Layout ---
st.title("📦 Sales Rep Forecast Dashboard")

st.sidebar.header("Login")
rep_id = st.sidebar.text_input("Enter your Sales Rep ID")

if rep_id:
    db = connect_db()
    stores = get_stores_for_rep(rep_id, db)

    if stores:
        store_options = {f"{name} (ID: {id})": id for id, name in stores}
        selected_store = st.selectbox("Select a Store", list(store_options.keys()))
        store_id = store_options[selected_store]

        # Load model and metadata
        model = load_model()
        sales_df, products_df, stores_df, sales_reps_df = load_metadata()

        # Simulate current date as Jan 1, 2018 for now
        simulated_today = datetime(2018, 1, 1)

        # Generate forecast dynamically
        store_forecast = generate_forecast_for_store(
            model, store_id, sales_df, products_df, stores_df, sales_reps_df, simulated_today
        )

        st.subheader(f"📊 Forecast for {selected_store} — Week of {simulated_today.strftime('%B %d, %Y')}")
        st.dataframe(store_forecast)

        # Show actual sales for the most recent week in this store
        current_week_end = sales_df['date'].max()
        current_week_start = current_week_end - pd.Timedelta(days=6)
        current_week = sales_df[
            (sales_df['store'] == store_id) &
            (sales_df['date'] >= current_week_start)
        ]
        current_week = current_week.merge(products_df, left_on='item', right_on='product_id')

        weekly_sales = current_week.groupby(['item', 'product_name'])['sales'].sum().reset_index()
        weekly_sales.rename(columns={'item': 'product_id'}, inplace=True)

        top_5_actual = weekly_sales.sort_values(by='sales', ascending=False).head(5)
        bottom_5_actual = weekly_sales.sort_values(by='sales').head(5)

        st.markdown("### 🟢 Top 5 Best-Selling Products (This Week)")
        st.table(top_5_actual[['product_id', 'product_name', 'sales']])

        st.markdown("### 🔴 Bottom 5 Least-Selling Products (This Week)")
        st.table(bottom_5_actual[['product_id', 'product_name', 'sales']])

    else:
        st.error("No stores found for this Sales Rep ID.")
else:
    st.info("Please enter your Sales Rep ID to begin.")
