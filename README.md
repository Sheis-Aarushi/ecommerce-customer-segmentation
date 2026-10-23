# 🛍️ E-Commerce Customer & Sales Intelligence System

An end-to-end data analytics and machine learning project that transforms retail transaction data into actionable business insights through sales analysis, customer segmentation, and an interactive Streamlit dashboard.

## 📌 Project Overview

The **E-Commerce Customer & Sales Intelligence System** analyzes online retail transactions to understand sales performance, identify valuable customers, and discover customer purchasing patterns.

The project combines Python, SQL, data analytics, machine learning, and interactive visualization in a single application.

## 🎯 Objectives

- Analyze sales and revenue trends over time.
- Identify top-performing countries and products.
- Segment customers using Recency, Frequency, and Monetary (RFM) analysis.
- Apply K-Means clustering to discover customer groups.
- Build an interactive dashboard for exploring business performance.
- Export filtered transaction and customer analysis data as CSV files.

## ✨ Key Features

### 1. Data Cleaning and Preparation

- Handle duplicate transactions.
- Exclude cancelled invoices and invalid transaction quantities or prices.
- Process dates and calculate transaction revenue.
- Prepare data for SQL queries and customer analytics.

### 2. Sales Analytics Using Python and SQL

- Calculate total revenue and order counts.
- Analyze monthly revenue trends.
- Identify top countries by revenue.
- Rank products based on revenue.
- Store transaction data in an SQLite database.

### 3. RFM Customer Segmentation

Customers are analyzed using three metrics:

| Metric | Meaning |
|---|---|
| Recency | How recently a customer purchased |
| Frequency | How often a customer purchased |
| Monetary | How much a customer spent |

Rule-based RFM scoring categorizes customers into segments such as Champions, Loyal Customers, Recent Customers, At Risk, and Needs Attention.

### 4. Machine Learning with K-Means

- Transform RFM features using `log1p` to reduce skewness.
- Standardize features using `StandardScaler`.
- Apply K-Means clustering with four clusters.
- Compare clusters using average recency, frequency, and monetary value.

### 5. Interactive Streamlit Dashboard

- Date-range and country filters for sales analysis.
- KPI cards for revenue, orders, customers, and products.
- Monthly revenue trend visualization.
- Top countries and products by revenue.
- RFM customer segment charts.
- K-Means cluster comparison.
- Customer and transaction data tables.
- Downloadable CSV reports.

## 🧰 Technology Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Database:** SQLite, SQL
- **Machine Learning:** Scikit-learn, K-Means
- **Visualization:** Plotly
- **Dashboard:** Streamlit
- **Dataset Format:** Excel
- **Development Environment:** Visual Studio Code

## 📂 Project Structure

```text
ecommerce-customer-segmentation/
├── app.py
├── dashboard.py
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    ├── Online Retail.xlsx
    ├── ecommerce.db
    └── rfm_customers.csv
```

The Excel dataset and generated data files are kept locally and excluded from GitHub using `.gitignore`.

## 📊 Dataset

This project uses the Online Retail dataset containing transactions from a UK-based online retailer.

The original dataset contains approximately 542,000 transaction records with columns including InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country.

The cleaned transaction dataset contains approximately 525,000 rows after the project's cleaning steps.

## 🚀 Getting Started

### Prerequisites

- Python installed on your computer
- Visual Studio Code or another Python editor
- The Online Retail dataset saved as `data/Online Retail.xlsx`

### 1. Clone the repository

```bash
git clone https://github.com/Sheis-Aarushi/ecommerce-customer-segmentation.git
cd ecommerce-customer-segmentation
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Add the dataset and run the analysis

Place `Online Retail.xlsx` inside the `data/` folder, then