import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="E-commerce Churn Dashboard", layout="wide")

# Load Data
@st.cache_data
def load_data():
    filepath = "../data/processed/customer_features.csv"
    if not os.path.exists(filepath):
        st.error(f"❌ File not found: {filepath}. Please run the preprocessing script first.")
        return pd.DataFrame()
    return pd.read_csv(filepath, parse_dates=["first_txn_date", "last_txn_date"])

df = load_data()

if df.empty:
    st.stop()

# 🛍️ Header
st.title("🛒 E-commerce Churn Prediction Dashboard")
st.markdown("📂 Looking for file at: `../data/processed/customer_features.csv`")

# ✅ Success message
st.success("✅ Data loaded successfully!")

# 📊 Show first few rows
st.dataframe(df.head())

# ✅ Check for churn column
if 'churn' not in df.columns:
    st.warning("⚠️ No `churn` column found in data.")
    st.stop()

# 🔍 Filter Section
churn_filter = st.selectbox("🔍 Filter by Churn Status:", options=["All", "Churned", "Retained"])
if churn_filter == "Churned":
    df = df[df["churn"] == 1]
elif churn_filter == "Retained":
    df = df[df["churn"] == 0]

# 📈 Churn Distribution
st.subheader("🔄 Churn Distribution")

churn_counts = df['churn'].value_counts().rename({0: "Retained", 1: "Churned"})
fig, ax = plt.subplots()
sns.barplot(x=churn_counts.index, y=churn_counts.values, palette="viridis", ax=ax)
ax.set_ylabel("Number of Customers")
ax.set_title("Churn vs Retained")
st.pyplot(fig)
