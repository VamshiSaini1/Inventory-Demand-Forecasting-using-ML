import streamlit as st
import pandas as pd
import mysql.connector
import pickle
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- Load all supporting data ---
@st.cache_data
def load_metadata():
    sales_df = pd.read_csv("train.csv", parse_dates=['date'])
    products_df = pd.read_csv("products_table.csv")
    stores_df = pd.read_csv("stores_table.csv")
    return sales_df, products_df, stores_df

# --- Generate forecast ---
def predict_next_week(model, store_id, sales_df, products_df, stores_df):
    sales_reps_df = pd.read_csv("sales_reps_table.csv")

    # Merge data
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

    # One-hot encode
    df = pd.get_dummies(df, columns=['month', 'day_of_week', 'store_name', 'product_name', 'name'], drop_first=True)

    latest_date = df['date'].max()
    pred_input = df[(df['date'] == latest_date) & (df['store'] == store_id)].copy()

    # Drop non-numeric
    X_pred = pred_input.drop(columns=['sales', 'date', 'email', 'phone', 'location'], errors='ignore')

    # Predict
    pred_input['predicted_sales'] = model.predict(X_pred)

    # Attach product info
    result = pred_input[['item', 'predicted_sales']].merge(products_df, left_on='item', right_on='product_id', how='left')
    return result[['product_id', 'product_name', 'predicted_sales']].sort_values(by='predicted_sales', ascending=False)

# --- UI Layout ---
st.title("🏬 Store Owner Dashboard")

store_id_input = st.sidebar.text_input("Enter your Store ID")

if store_id_input:
    store_id = int(store_id_input)

    model = load_model()
    sales_df, products_df, stores_df = load_metadata()

    store_name = stores_df[stores_df['store_id'] == store_id]['store_name'].values[0]
    st.header(f"📍 {store_name} (Store ID: {store_id})")

    # Forecast
    forecast_df = predict_next_week(model, store_id, sales_df, products_df, stores_df)
    st.subheader("📦 Predicted Stock Needed for Next Week")
    st.dataframe(forecast_df)

    # Current week sales
    current_week_end = sales_df['date'].max()
    current_week_start = current_week_end - pd.Timedelta(days=6)
    current_week = sales_df[(sales_df['store'] == store_id) & (sales_df['date'] >= current_week_start)]
    current_week = current_week.merge(products_df, left_on='item', right_on='product_id')
    weekly_sales = current_week.groupby(['item', 'product_name'])['sales'].sum().reset_index()
    weekly_sales.rename(columns={'item': 'product_id'}, inplace=True)

    st.subheader("🧾 Current Week Sales")
    st.dataframe(weekly_sales.sort_values(by='sales', ascending=False))

    # Best & Worst
    top_5 = weekly_sales.sort_values(by='sales', ascending=False).head(5)[['product_id', 'product_name', 'sales']]
    bottom_5 = weekly_sales.sort_values(by='sales').head(5)[['product_id', 'product_name', 'sales']]

    st.markdown("### 🟢 Best-Selling Products This Week")
    st.table(top_5)

    st.markdown("### 🔴 Least-Selling Products This Week")
    st.table(bottom_5)

    # Chart: Sales Trend
    st.markdown("### 📈 Weekly Sales Trend")
    trend_data = sales_df[sales_df['store'] == store_id].copy()
    trend_data = trend_data.merge(products_df, left_on='item', right_on='product_id')
    trend_data['week'] = trend_data['date'].dt.isocalendar().week
    weekly_trend = trend_data.groupby(['week'])['sales'].sum().reset_index()

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=weekly_trend, x='week', y='sales')
    plt.title("Weekly Sales Trend")
    plt.xlabel("Week")
    plt.ylabel("Total Sales")
    st.pyplot(plt)
else:
    st.info("Please enter your Store ID to view your dashboard.")
