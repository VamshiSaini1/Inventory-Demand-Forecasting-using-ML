# 🛒 Inventory Demand Forecasting Using Machine Learning

An end-to-end intelligent forecasting system that predicts weekly product demand at individual store levels using machine learning. The system is deployed via two web applications tailored for sales reps and store owners to support real-time inventory planning.

---

## 📌 Project Objective

In the context of retail and supply chain operations, anticipating product demand is crucial to avoid overstocking or stockouts. This project uses historical sales data and machine learning to forecast upcoming weekly sales for each product-store pair, enabling more informed and proactive inventory decisions.

---

## ⚙️ Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** XGBoost, Random Forest, Linear Regression (comparison)  
- **Data Handling:** Pandas, NumPy, SQLAlchemy  
- **Database:** MySQL  
- **Web Framework:** Streamlit  
- **Model Evaluation:** MAE, RMSE, MAPE  

---

## 🧪 Workflow Summary

### 1. **Data Setup**
- Used 9M+ rows of transactional data from 2013–2017 (`train.csv`)
- Created synthetic metadata for products, stores, and sales reps
- Merged and loaded into a MySQL database (`Saini_Distributions`)

### 2. **Exploratory Data Analysis**
- Analyzed trends, seasonality, product/store performance
- Identified key features and patterns for modeling

### 3. **Feature Engineering**
- Lag features (`lag_1`, `lag_7`, `lag_14`)
- Rolling statistics (`rolling_mean_7`, `rolling_std_7`)
- Date parts (`month`, `week`, `day_of_week`)
- One-hot encoding for stores, products, reps

### 4. **Modeling**
- Framed as a supervised regression problem
- Trained and evaluated multiple models  
- **XGBoost** selected based on superior accuracy:
  - MAE: 10.73
  - RMSE: 13.92
  - MAPE: 3.04%

### 5. **Application Deployment**
- **Sales Rep App:** View forecasts per store, Top 5 / Bottom 5 products
- **Store Owner App:** Track store-wide forecasts and performance
- Both apps dynamically use the trained XGBoost model and fetch data from MySQL

---

## 📈 Results

- XGBoost outperformed baseline models and achieved a MAPE of **3.04%**
- Streamlit apps provided role-specific insights in a user-friendly way
- Enabled real-time forecasting and decision support without technical overhead for end users

---

## 🚀 Future Enhancements

- Include holiday and promotional effects in modeling
- Build in-app ordering system for sales reps
- Extend to long-term monthly or quarterly forecasts
- Hyperparameter tuning for further model optimization

---

## 📚 References

1. S. Shekhar, “Types of Demand Forecast,” *GeeksforGeeks*, 2022.  
2. Y. Gülcan, “Linear Regression vs XGBoost Regressor,” *Kaggle*.  
3. XGBoost Documentation – https://xgboost.readthedocs.io/  
4. Streamlit – https://streamlit.io/  
