import pandas as pd

# Load dataset
file_path = "data/Online Retail.xlsx"
df = pd.read_excel(file_path)

print("Original shape:", df.shape)

# Remove exact duplicate transactions
df = df.drop_duplicates()

# Remove cancelled invoices
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Keep only positive quantities
df = df[df["Quantity"] > 0]

# Keep only positive prices
df = df[df["UnitPrice"] > 0]

print("Cleaned shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# Calculate revenue for each transaction
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

print("\nRevenue calculated successfully!")

print("\nRevenue Statistics:")
print(df["Revenue"].describe())

print("\nTotal Revenue:")
print(df["Revenue"].sum())

# Top 10 products by revenue
top_products = (
    df.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Products by Revenue:")
print(top_products)

# Top 10 countries by revenue
top_countries = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Countries by Revenue:")
print(top_countries)

# Monthly revenue analysis
df["Month"] = df["InvoiceDate"].dt.to_period("M")

monthly_revenue = (
    df.groupby("Month")["Revenue"]
    .sum()
    .sort_index()
)

print("\nMonthly Revenue:")
print(monthly_revenue)
best_month = monthly_revenue.idxmax()
best_month_revenue = monthly_revenue.max()

print("\nBest Revenue Month:")
print(best_month)

print("Revenue:")
print(best_month_revenue)

# Create customer analysis dataset
customer_df = df.dropna(subset=["CustomerID"]).copy()

print("\nCustomer Analysis Dataset:")
print(customer_df.shape)

print("\nUnique Customers:")
print(customer_df["CustomerID"].nunique())

# RFM Analysis

# Set analysis date as the day after the last transaction
analysis_date = customer_df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = customer_df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (analysis_date - x.max()).days,
    "InvoiceNo": "nunique",
    "Revenue": "sum"
})

# Rename columns
rfm.columns = ["Recency", "Frequency", "Monetary"]

print("\nRFM Analysis:")
print(rfm.head())

print("\nRFM Shape:")
print(rfm.shape)

# Create RFM scores

rfm["R_Score"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5, 4, 3, 2, 1]
)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"],
    5,
    labels=[1, 2, 3, 4, 5]
)

print("\nRFM Scores:")
print(rfm.head(10))

# Combine RFM scores into one score

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)

print("\nCombined RFM Scores:")
print(rfm[[
    "Recency",
    "Frequency",
    "Monetary",
    "R_Score",
    "F_Score",
    "M_Score",
    "RFM_Score"
]].head(10))

# Create customer segments

def segment_customer(row):
    r = int(row["R_Score"])
    f = int(row["F_Score"])
    m = int(row["M_Score"])

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    elif r >= 4 and f >= 3:
        return "Loyal Customers"

    elif r >= 4 and m >= 4:
        return "Potential Loyalists"

    elif r <= 2 and f >= 4:
        return "At Risk"

    elif r <= 2 and m >= 4:
        return "High Value At Risk"

    elif r >= 4:
        return "Recent Customers"

    else:
        return "Needs Attention"


rfm["Segment"] = rfm.apply(segment_customer, axis=1)

# Save RFM customer analysis for the dashboard
rfm.to_csv("data/rfm_customers.csv", index=True)

print("\nRFM customer data saved successfully!")
print("Saved file: data/rfm_customers.csv")


print("\nCustomer Segments:")
print(rfm["Segment"].value_counts())

print("\nSample Segmented Customers:")
print(
    rfm[
        [
            "Recency",
            "Frequency",
            "Monetary",
            "RFM_Score",
            "Segment"
        ]
    ].head(15)
)
# Segment distribution

segment_summary = (
    rfm.groupby("Segment")
    .agg(
        Customers=("Segment", "count"),
        Total_Revenue=("Monetary", "sum"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean")
    )
    .sort_values("Customers", ascending=False)
)

print("\nSegment Summary:")
print(segment_summary)

# K-Means Customer Clustering

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Select RFM features
rfm_features = rfm[["Recency", "Frequency", "Monetary"]]

# Scale the features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

# Create K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Train model and assign clusters
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

print("\nK-Means Clustering:")
print(rfm[[
    "Recency",
    "Frequency",
    "Monetary",
    "Cluster"
]].head(15))

print("\nCluster Distribution:")
print(rfm["Cluster"].value_counts().sort_index())

# Analyze K-Means clusters

cluster_summary = (
    rfm.groupby("Cluster")
    .agg(
        Customers=("Cluster", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean")
    )
    .sort_index()
)

print("\nK-Means Cluster Summary:")

print(
    cluster_summary.to_string()
)
# Improved K-Means using log-transformed RFM data

import numpy as np

# Apply log transformation
rfm_log = np.log1p(
    rfm[["Recency", "Frequency", "Monetary"]]
)

# Scale transformed data
scaler_log = StandardScaler()
rfm_scaled_log = scaler_log.fit_transform(rfm_log)

# Create improved K-Means model
kmeans_log = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Train model
rfm["ML_Cluster"] = kmeans_log.fit_predict(rfm_scaled_log)

# Save final RFM data after K-Means clustering
rfm.to_csv("data/rfm_customers.csv", index=True)

print("\nRFM customer data saved successfully!")
print("Saved file: data/rfm_customers.csv")
print("Columns saved:", rfm.columns.tolist())


# Analyze improved clusters
improved_cluster_summary = (
    rfm.groupby("ML_Cluster")
    .agg(
        Customers=("ML_Cluster", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean")
    )
    .sort_index()
)

print("\nImproved K-Means Cluster Summary:")
print(improved_cluster_summary.to_string())
# Create SQLite database

import sqlite3

# Create a copy for SQLite
sql_df = df.copy()

# Convert Pandas Period column to text
if "Month" in sql_df.columns:
    sql_df["Month"] = sql_df["Month"].astype(str)

# Connect to SQLite database
conn = sqlite3.connect("data/ecommerce.db")

# Store data in SQLite
sql_df.to_sql(
    "transactions",
    conn,
    if_exists="replace",
    index=False
)

print("\nSQLite database created successfully!")
print("Table 'transactions' created successfully.")

conn.close()
# SQL Analysis

conn = sqlite3.connect("data/ecommerce.db")

query = """
SELECT
    Country,
    SUM(Revenue) AS Total_Revenue
FROM transactions
GROUP BY Country
ORDER BY Total_Revenue DESC
LIMIT 10;
"""

top_countries_sql = pd.read_sql_query(query, conn)

print("\nTop 10 Countries by Revenue - SQL:")
print(top_countries_sql.to_string())

conn.close()
# Monthly Revenue Analysis using SQL

conn = sqlite3.connect("data/ecommerce.db")

query = """
SELECT
    Month,
    SUM(Revenue) AS Total_Revenue
FROM transactions
GROUP BY Month
ORDER BY Month;
"""

monthly_revenue_sql = pd.read_sql_query(query, conn)

print("\nMonthly Revenue - SQL:")
print(monthly_revenue_sql.to_string())

conn.close()
# Top 10 Customers by Revenue using SQL

conn = sqlite3.connect("data/ecommerce.db")

query = """
SELECT
    CustomerID,
    COUNT(DISTINCT InvoiceNo) AS Total_Orders,
    SUM(Revenue) AS Total_Revenue
FROM transactions
WHERE CustomerID IS NOT NULL
GROUP BY CustomerID
ORDER BY Total_Revenue DESC
LIMIT 10;
"""

top_customers_sql = pd.read_sql_query(query, conn)

print("\nTop 10 Customers by Revenue - SQL:")
print(top_customers_sql.to_string())

conn.close()


import plotly.express as px

# Top 10 Countries by Revenue
chart_data = top_countries_sql.sort_values(
    "Total_Revenue", ascending=True
)

fig = px.bar(
    chart_data,
    x="Total_Revenue",
    y="Country",
    orientation="h",
    title="Top 10 Countries by Revenue",
    labels={
        "Total_Revenue": "Total Revenue (£)",
        "Country": "Country"
    },
    text=chart_data["Total_Revenue"].apply(
        lambda x: f"£{x:,.0f}"
    ),
    template="plotly_white"
)

fig.update_layout(
    xaxis_type="log",
    xaxis_title="Total Revenue (£, logarithmic scale)",
    yaxis_title="",
    showlegend=False,
    height=500
)

fig.update_traces(textposition="outside")

fig.show()
