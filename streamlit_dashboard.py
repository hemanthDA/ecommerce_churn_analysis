import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load Data
@st.cache_data

def load_data():
    filepath = "../data/processed/customer_features.csv"
    if not os.path.exists(filepath):
        st.error(f"❌ File not found: {filepath}")
        return pd.DataFrame()
    df = pd.read_csv(filepath, parse_dates=['Date'])
    return df

df = load_data()
if df.empty:
    st.stop()

# Sidebar filters
st.sidebar.header("Filter Options")
regions = st.sidebar.multiselect("Select Region(s):", df["Region"].unique(), default=df["Region"].unique())
categories = st.sidebar.multiselect("Select Category:", df["Category"].unique(), default=df["Category"].unique())
date_range = st.sidebar.date_input("Select Date Range:", [df['Date'].min(), df['Date'].max()])
data_view = st.sidebar.radio("Toggle View:", ["Sales", "Returns"])

# Filtered data
df_filtered = df[(df['Region'].isin(regions)) &
                 (df['Category'].isin(categories)) &
                 (df['Date'] >= pd.to_datetime(date_range[0])) &
                 (df['Date'] <= pd.to_datetime(date_range[1]))]

metric_col = "Sales" if data_view == "Sales" else "Returns"

# KPI Section
st.markdown("### 📊 Key Metrics")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Sales", f"₹{df_filtered['Sales'].sum():,.0f}")
kpi2.metric("Total Profit", f"₹{df_filtered['Profit'].sum():,.0f}")
kpi3.metric("Churn Rate", f"{df_filtered['Churn'].mean() * 100:.2f}%")
kpi4.metric("Average Rating", f"{df_filtered['Customer_Rating'].mean():.2f} ⭐")

# Layout
col1, col2 = st.columns([2, 1])

# Bar Chart – Top 10 Selling Products
with col1:
    st.subheader("Top 10 Products by " + metric_col)
    top_products = df_filtered.groupby("Product")[metric_col].sum().nlargest(10).reset_index()
    bar_fig = px.bar(top_products, x=metric_col, y="Product", orientation="h", title="Top 10 Products",
                     color=metric_col, height=400)
    st.plotly_chart(bar_fig, use_container_width=True)

# Pie Chart – Sales by Region
with col2:
    st.subheader(f"{metric_col} Distribution by Region")
    region_pie = df_filtered.groupby("Region")[metric_col].sum().reset_index()
    pie_fig = px.pie(region_pie, values=metric_col, names="Region", hole=0.4)
    st.plotly_chart(pie_fig, use_container_width=True)

# Line Chart – Time Series
st.subheader(f"{metric_col} Over Time")
time_series = df_filtered.groupby("Date")[metric_col].sum().reset_index()
line_fig = px.line(time_series, x="Date", y=metric_col, title=f"{metric_col} Over Time")
st.plotly_chart(line_fig, use_container_width=True)

# Scatter Plot – Rating vs Churn
st.subheader("Customer Rating vs. Churn Probability")
scatter_fig = px.scatter(df_filtered, x="Customer_Rating", y="Churn", color="Region",
                         title="Ratings vs Churn Probability", size=metric_col)
st.plotly_chart(scatter_fig, use_container_width=True)

# Heatmap – Correlation
st.subheader("Correlation Heatmap")
numeric_df = df_filtered.select_dtypes(include=np.number)
corr = numeric_df.corr(numeric_only=True)
heat_fig = px.imshow(corr, text_auto=True, title="Correlation Matrix", height=500)
st.plotly_chart(heat_fig, use_container_width=True)

# Area Chart – Cumulative Monthly Sales
st.subheader("Cumulative Monthly Sales")
df_filtered['Month'] = df_filtered['Date'].dt.to_period('M').dt.to_timestamp()
monthly = df_filtered.groupby("Month")["Sales"].sum().cumsum().reset_index()
area_fig = px.area(monthly, x="Month", y="Sales", title="Cumulative Monthly Sales")
st.plotly_chart(area_fig, use_container_width=True)

# Donut Chart – Churn Distribution
st.subheader("Churn vs Non-Churn")
churn_counts = df_filtered['Churn'].value_counts().rename({0: 'Active', 1: 'Churned'}).reset_index()
churn_fig = px.pie(churn_counts, names="index", values="Churn", hole=0.5, title="Customer Churn Distribution")
st.plotly_chart(churn_fig, use_container_width=True)

# Map (only if 'Latitude' & 'Longitude' exist)
if {'Latitude', 'Longitude'}.issubset(df_filtered.columns):
    st.subheader("Regional Sales Map")
    map_fig = px.scatter_mapbox(df_filtered, lat="Latitude", lon="Longitude",
                                color=metric_col, size=metric_col,
                                mapbox_style="open-street-map", zoom=3, title="Regional Sales")
    st.plotly_chart(map_fig, use_container_width=True)

# Table
st.subheader("Customer/Product Data Table")
st.dataframe(df_filtered.head(100), use_container_width=True)

# Download buttons
st.download_button("Download Filtered Data as CSV", df_filtered.to_csv(index=False), file_name="filtered_data.csv")
