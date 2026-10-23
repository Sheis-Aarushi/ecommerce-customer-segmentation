
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="E-Commerce Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "ecommerce.db"
RFM_PATH = BASE_DIR / "data" / "rfm_customers.csv"

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(120, 120, 120, 0.07);
        border: 1px solid rgba(120, 120, 120, 0.18);
        padding: 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_transactions():
    with sqlite3.connect(DB_PATH) as conn:
        data = pd.read_sql_query(
            "SELECT * FROM transactions", conn
        )

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )
    data["Revenue"] = pd.to_numeric(
        data["Revenue"], errors="coerce"
    )
    data["Quantity"] = pd.to_numeric(
        data["Quantity"], errors="coerce"
    )

    data = data.dropna(subset=["InvoiceDate", "Revenue"])
    return data


@st.cache_data
def load_rfm():
    if not RFM_PATH.exists():
        return pd.DataFrame()

    data = pd.read_csv(RFM_PATH)

    # Handle CSV files saved with the customer ID as the index.
    if "CustomerID" not in data.columns:
        for column in ["Unnamed: 0", "index"]:
            if column in data.columns:
                data = data.rename(
                    columns={column: "CustomerID"}
                )
                break

    return data


if not DB_PATH.exists():
    st.error(
        "Database not found. Please check that "
        "data/ecommerce.db exists inside your project folder."
    )
    st.stop()

try:
    df = load_transactions()
    rfm = load_rfm()
except Exception as error:
    st.error(f"Could not load project data: {error}")
    st.stop()

if df.empty:
    st.warning("No transaction data is available.")
    st.stop()

# =====================================================
# HEADER
# =====================================================

st.title("📊 E-Commerce Intelligence")
st.markdown(
    "Sales performance · Customer behavior · Machine learning"
)
st.caption(
    "An interactive analytics dashboard built with "
    "Python, SQL, Pandas, Plotly and Streamlit."
)

# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.title("Dashboard Filters")
st.sidebar.caption("Choose the data you want to analyze.")

countries = sorted(
    df["Country"].dropna().unique().tolist()
)

selected_countries = st.sidebar.multiselect(
    "Countries",
    options=countries,
    default=countries,
    help="Select one or more countries.",
)

min_available_date = df["InvoiceDate"].min().date()
max_available_date = df["InvoiceDate"].max().date()

st.sidebar.markdown("### Date Range")

start_date = st.sidebar.date_input(
    "Start date",
    value=min_available_date,
    min_value=min_available_date,
    max_value=max_available_date,
)

end_date = st.sidebar.date_input(
    "End date",
    value=max_available_date,
    min_value=min_available_date,
    max_value=max_available_date,
)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

if not selected_countries:
    st.warning("Select at least one country in the sidebar.")
    st.stop()

# The end date is inclusive.
filtered_df = df[
    df["Country"].isin(selected_countries)
    & (df["InvoiceDate"].dt.date >= start_date)
    & (df["InvoiceDate"].dt.date <= end_date)
].copy()

st.sidebar.caption(
    f"{len(filtered_df):,} transactions match your filters."
)

if filtered_df.empty:
    st.warning(
        "No transactions match these filters. "
        "Try another country or date range."
    )
    st.stop()

# =====================================================
# BUSINESS OVERVIEW - KPI CARDS
# =====================================================

st.divider()
st.header("Business Overview")

total_revenue = filtered_df["Revenue"].sum()
total_orders = filtered_df["InvoiceNo"].nunique()
total_products = filtered_df["StockCode"].nunique()
total_customers = filtered_df["CustomerID"].nunique()

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Total Revenue", f"£{total_revenue:,.0f}")

with k2:
    st.metric("Total Orders", f"{total_orders:,}")

with k3:
    st.metric("Unique Customers", f"{total_customers:,}")

with k4:
    st.metric("Unique Products", f"{total_products:,}")

st.caption(
    f"Showing {start_date:%d %b %Y} to {end_date:%d %b %Y}"
)

# =====================================================
# SALES ANALYTICS
# =====================================================

st.divider()
st.header("📈 Sales Analytics")

monthly_revenue = (
    filtered_df.assign(
        Month=filtered_df["InvoiceDate"].dt.to_period("M").astype(str)
    )
    .groupby("Month", as_index=False)["Revenue"]
    .sum()
    .sort_values("Month")
)

country_revenue = (
    filtered_df.groupby("Country", as_index=False)["Revenue"]
    .sum()
    .sort_values("Revenue", ascending=False)
    .head(10)
    .sort_values("Revenue", ascending=True)
)

left_chart, right_chart = st.columns(2, gap="large")

with left_chart:
    st.subheader("Monthly Revenue Trend")

    fig_monthly = px.line(
        monthly_revenue,
        x="Month",
        y="Revenue",
        markers=True,
        labels={
            "Month": "Month",
            "Revenue": "Revenue (£)",
        },
        template="plotly_white",
    )

    fig_monthly.update_traces(
        line=dict(width=3),
        hovertemplate="%{x}<br>Revenue: £%{y:,.2f}<extra></extra>",
    )

    fig_monthly.update_layout(
        height=400,
        margin=dict(l=10, r=15, t=20, b=10),
        xaxis_title="Month",
        yaxis_title="Revenue (£)",
    )

    st.plotly_chart(
        fig_monthly, use_container_width=True,
        key="monthly_revenue_chart",
    )

with right_chart:
    st.subheader("Top 10 Countries by Revenue")

    fig_country = px.bar(
        country_revenue,
        x="Revenue",
        y="Country",
        orientation="h",
        labels={
            "Revenue": "Revenue (£)",
            "Country": "",
        },
        template="plotly_white",
    )

    fig_country.update_traces(
        marker_color="#636EFA",
        text=country_revenue["Revenue"].map(
            lambda value: f"£{value:,.0f}"
        ),
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>Revenue: £%{x:,.2f}<extra></extra>",
    )

    fig_country.update_layout(
        height=400,
        margin=dict(l=10, r=65, t=20, b=10),
        xaxis_title="Revenue (£)",
        yaxis_title="",
    )

    st.plotly_chart(
        fig_country, use_container_width=True,
        key="country_revenue_chart",
    )

# =====================================================
# PRODUCT ANALYTICS
# =====================================================

st.divider()
st.header("🛍️ Product Insights")

product_revenue = (
    filtered_df.groupby("Description", as_index=False)
    .agg(
        Total_Revenue=("Revenue", "sum"),
        Units_Sold=("Quantity", "sum"),
    )
    .dropna(subset=["Description"])
    .sort_values("Total_Revenue", ascending=False)
    .head(10)
    .sort_values("Total_Revenue", ascending=True)
)

st.subheader("Top 10 Products by Revenue")

fig_products = px.bar(
    product_revenue,
    x="Total_Revenue",
    y="Description",
    orientation="h",
    labels={
        "Total_Revenue": "Revenue (£)",
        "Description": "Product",
    },
    template="plotly_white",
)

fig_products.update_layout(
    height=470,
    margin=dict(l=10, r=35, t=20, b=10),
)

st.plotly_chart(
    fig_products, use_container_width=True,
    key="product_revenue_chart",
)

# =====================================================
# CUSTOMER INTELLIGENCE - RFM
# =====================================================

st.divider()
st.header("👥 Customer Intelligence")
st.caption(
    "RFM analysis groups customers by how recently, "
    "how frequently, and how much they purchase."
)

if rfm.empty:
    st.warning(
        "RFM data was not found. Run app.py to generate "
        "data/rfm_customers.csv."
    )
else:
    if "CustomerID" not in rfm.columns:
        st.warning(
            "CustomerID is missing from the RFM CSV. "
            "Check how the file was saved in app.py."
        )
    elif "Segment" not in rfm.columns:
        st.warning(
            "The Segment column is missing from the RFM CSV."
        )
    else:
        rfm["CustomerID"] = rfm["CustomerID"].astype(str)

        for column in [
            "Recency", "Frequency", "Monetary", "ML_Cluster"
        ]:
            if column in rfm.columns:
                rfm[column] = pd.to_numeric(
                    rfm[column], errors="coerce"
                )

        segment_summary = (
            rfm.dropna(subset=["Segment"])
            .groupby("Segment", as_index=False)
            .agg(
                Customers=("CustomerID", "nunique"),
                Avg_Monetary=("Monetary", "mean"),
            )
            .sort_values("Customers", ascending=False)
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Customers in RFM Analysis", f"{len(rfm):,}")
        c2.metric("Customer Segments", f"{rfm['Segment'].nunique()}")
        c3.metric(
            "Average Customer Value",
            f"£{rfm['Monetary'].mean():,.2f}"
            if "Monetary" in rfm.columns else "N/A",
        )

        seg_left, seg_right = st.columns(2, gap="large")

        with seg_left:
            st.subheader("Customer Distribution")

            fig_segments = px.bar(
                segment_summary.sort_values("Customers"),
                x="Customers",
                y="Segment",
                orientation="h",
                text="Customers",
                labels={"Customers": "Customers", "Segment": ""},
                template="plotly_white",
            )

            fig_segments.update_layout(
                height=400,
                margin=dict(l=10, r=15, t=20, b=10),
            )

            st.plotly_chart(
                fig_segments, use_container_width=True,
                key="customer_segment_chart",
            )

        with seg_right:
            st.subheader("Segment Share")

            fig_segment_share = px.pie(
                segment_summary,
                names="Segment",
                values="Customers",
                hole=0.5,
                template="plotly_white",
            )

            fig_segment_share.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=20, b=10),
                legend_title_text="Customer Segment",
            )

            st.plotly_chart(
                fig_segment_share, use_container_width=True,
                key="customer_segment_pie",
            )

        st.subheader("Average Revenue by Segment")

        segment_value = segment_summary.sort_values(
            "Avg_Monetary", ascending=False
        )

        fig_segment_value = px.bar(
            segment_value,
            x="Segment",
            y="Avg_Monetary",
            text_auto=".2s",
            labels={
                "Segment": "Customer Segment",
                "Avg_Monetary": "Average Revenue (£)",
            },
            template="plotly_white",
        )

        fig_segment_value.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
        )

        st.plotly_chart(
            fig_segment_value, use_container_width=True,
            key="segment_value_chart",
        )

        # =================================================
        # MACHINE LEARNING - K-MEANS CLUSTERS
        # =================================================

        st.divider()
        st.header("🤖 Machine Learning Insights")

        if "ML_Cluster" in rfm.columns:
            cluster_data = rfm.dropna(
                subset=["ML_Cluster"]
            ).copy()

            cluster_summary = (
                cluster_data.groupby("ML_Cluster", as_index=False)
                .agg(
                    Customers=("CustomerID", "nunique"),
                    Avg_Recency=("Recency", "mean"),
                    Avg_Frequency=("Frequency", "mean"),
                    Avg_Monetary=("Monetary", "mean"),
                )
            )

            st.caption(
                "Clusters are discovered using K-Means. "
                "Their numbers are identifiers, not rankings."
            )

            fig_clusters = px.scatter(
                cluster_summary,
                x="Avg_Recency",
                y="Avg_Monetary",
                size="Customers",
                color="ML_Cluster",
                hover_data=[
                    "Customers",
                    "Avg_Frequency",
                ],
                labels={
                    "Avg_Recency": "Average Recency (days)",
                    "Avg_Monetary": "Average Customer Revenue (£)",
                    "ML_Cluster": "Cluster",
                    "Customers": "Customers",
                },
                template="plotly_white",
            )

            fig_clusters.update_layout(height=430)

            st.plotly_chart(
                fig_clusters, use_container_width=True,
                key="kmeans_cluster_chart",
            )

            st.subheader("K-Means Cluster Summary")
            st.dataframe(
                cluster_summary.round(2),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "ML_Cluster was not found. Run app.py and save "
                "the RFM CSV after K-Means clustering."
            )

        # =================================================
        # CUSTOMER EXPLORER
        # =================================================

        st.divider()
        st.header("🔎 Customer Explorer")

        segment_options = sorted(
            rfm["Segment"].dropna().unique().tolist()
        )

        selected_segment = st.selectbox(
            "Filter by RFM segment",
            ["All Segments"] + segment_options,
        )

        customer_view = rfm.copy()

        if selected_segment != "All Segments":
            customer_view = customer_view[
                customer_view["Segment"] == selected_segment
            ]

        available_columns = [
            column for column in [
                "CustomerID",
                "Recency",
                "Frequency",
                "Monetary",
                "R_Score",
                "F_Score",
                "M_Score",
                "Segment",
                "ML_Cluster",
            ]
            if column in customer_view.columns
        ]

        customer_view = customer_view[available_columns]

        st.dataframe(
            customer_view,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download Customer Analysis CSV",
            data=customer_view.to_csv(index=False).encode("utf-8"),
            file_name="customer_analysis.csv",
            mime="text/csv",
        )

# =====================================================
# TRANSACTION EXPLORER AND DOWNLOAD
# =====================================================

st.divider()
st.header("📋 Transaction Explorer")

st.caption(
    "The table follows the selected country and date filters."
)

st.dataframe(
    filtered_df.head(500),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "⬇️ Download Filtered Transactions CSV",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_transactions.csv",
    mime="text/csv",
)

st.caption(
    "E-Commerce Customer & Sales Intelligence | "
    "Python · SQL · Pandas · Plotly · Streamlit · Scikit-learn"
)

