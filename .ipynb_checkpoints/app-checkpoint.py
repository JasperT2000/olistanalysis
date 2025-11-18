# ============================================================
# CUSTOMER INTELLIGENCE DASHBOARD 
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from mlxtend.frequent_patterns import apriori, association_rules


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    "<h2 style='text-align:center;'>🧠 Customer Intelligence Dashboard</h2>",
    unsafe_allow_html=True
)


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

@st.cache_data
def load_data():
    cust_agg = pd.read_csv("data/cust_agg.csv")
    orders = pd.read_csv("data/orders.csv")
    order_items = pd.read_csv("data/order_items.csv")
    products = pd.read_csv("data/olist_products_dataset.csv")
    customers = pd.read_csv("data/olist_customers_dataset.csv")
    return cust_agg, orders, order_items, products, customers

cust_agg, orders, order_items, products, customers = load_data()


# ============================================================
# CLEANUP
# ============================================================

cust = cust_agg.copy().reset_index(drop=True)

# cust_agg was built on customer_unique_id originally
if "customer_unique_id" in cust.columns:
    cust.rename(columns={"customer_unique_id": "customer_id"}, inplace=True)
elif "customer_id" not in cust.columns:
    cust["customer_id"] = cust.index.astype(str)

cust["segment"] = cust.get("segment", 0).astype(str)
cust["churned"] = cust["churned"].astype(int)

# For mapping from unique customer to orders' customer_id
cust_map = customers[["customer_unique_id", "customer_id"]].drop_duplicates()
cust_map.rename(
    columns={"customer_unique_id": "unique_id", "customer_id": "order_customer_id"},
    inplace=True
)


# ============================================================
# REVENUE BY MONTH
# ============================================================

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"]
)
orders["order_month"] = (
    orders["order_purchase_timestamp"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

orders2 = orders.merge(
    order_items[["order_id", "price"]],
    on="order_id",
    how="left"
)
monthly_rev = (
    orders2.groupby("order_month")["price"]
    .sum()
    .reset_index(name="revenue")
)


# ============================================================
# CHURN MODEL
# ============================================================

feature_cols = [
    "frequency",
    "monetary",
    "avg_order_value",
    "freight_ratio",
    "avg_review_score",
]

feature_cols = [c for c in feature_cols if c in cust.columns]

X = cust[feature_cols]
y = cust["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

cust["churn_prob"] = rf.predict_proba(X)[:, 1]

y_prob_test = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob_test)
fpr, tpr, _ = roc_curve(y_test, y_prob_test)


# ============================================================
# CALCULATE REPURCHASE INTERVAL (BASED ON UNIQUE CUSTOMER)
# ============================================================

# Join orders with unique id from customers
orders_ext = orders.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left"
)

orders_ext["order_purchase_timestamp"] = pd.to_datetime(
    orders_ext["order_purchase_timestamp"]
)

# Median days between orders per unique customer
order_dates = (
    orders_ext
    .dropna(subset=["customer_unique_id"])
    .sort_values("order_purchase_timestamp")
    .groupby("customer_unique_id")["order_purchase_timestamp"]
    .apply(lambda s: s.diff().dt.days.median())
)

# Map from unique id (stored in cust['customer_id']) to repurchase interval
cust["repurchase_interval"] = cust["customer_id"].map(order_dates)

# Notify a bit *before* the usual repurchase time (e.g. at 75% of the interval)
cust["notify_day"] = (cust["repurchase_interval"] * 0.75).round()
cust["notify_day"] = cust["notify_day"].replace([np.inf, -np.inf], np.nan)


# ============================================================
# NEXT-BEST-OFFER (Product Rules, Top 120 SKUs)
# ============================================================

top_products = order_items["product_id"].value_counts().head(120).index
oi_top = order_items[order_items["product_id"].isin(top_products)].copy()

basket = (
    oi_top.assign(value=1)
    .pivot_table(
        index="order_id",
        columns="product_id",
        values="value",
        aggfunc="sum",
        fill_value=0,
    )
)

basket = (basket > 0).astype(int)

frequent = apriori(
    basket,
    min_support=0.0005,
    use_colnames=True,
    low_memory=True,
)

rules = association_rules(frequent, metric="lift", min_threshold=1.0)
rules = rules[rules["antecedents"] != rules["consequents"]].copy()


def fs_to_single(fs):
    return list(fs)[0] if len(fs) else None


rules["antecedent_pid"] = rules["antecedents"].apply(fs_to_single)
rules["consequent_pid"] = rules["consequents"].apply(fs_to_single)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "📊 Overview",
        "👥 Segments",
        "⚠️ Churn Model",
        "🛒 Customer Insights",
        "💎 High-Value Customers",
    ]
)


# ============================================================
# OVERVIEW TAB
# ============================================================

with tabs[0]:

    st.subheader("📊 Business Overview")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{len(cust):,}")
    k2.metric("Churn Rate", f"{cust['churned'].mean()*100:.1f}%")
    k3.metric("Avg Order Value", f"${cust['avg_order_value'].mean():,.2f}")
    k4.metric("Total Revenue", f"${cust['monetary'].sum():,.0f}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.histogram(
                cust,
                x="recency_days",
                nbins=30,
                title="Recency Distribution",
            ),
            use_container_width=True,
        )

    with c2:
        st.plotly_chart(
            px.histogram(
                cust,
                x="monetary",
                nbins=30,
                title="Monetary Distribution",
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        px.line(
            monthly_rev,
            x="order_month",
            y="revenue",
            title="Revenue Over Time",
        ),
        use_container_width=True,
    )


# ============================================================
# SEGMENTS TAB 
# ============================================================

with tabs[1]:

    st.markdown("## 👥 Customer Segments")

    st.markdown("### 📡 Segment Profiles (Radar Chart)")

    radar_features = [
        "recency_days",
        "frequency",
        "monetary",
        "avg_order_value",
        "freight_ratio",
        "avg_review_score",
    ]

    seg_options = sorted(cust["segment"].unique())
    default_selection = seg_options[:2] if len(seg_options) >= 2 else seg_options

    selected_segments = st.multiselect(
        "Select one or two segments to compare",
        options=seg_options,
        default=default_selection,
    )

    if len(selected_segments) == 0:
        st.info("Select at least one segment to display the radar chart.")
    else:
        seg_profile_full = cust.groupby("segment")[radar_features].mean()
        seg_norm = (seg_profile_full - seg_profile_full.min()) / (
            seg_profile_full.max() - seg_profile_full.min() + 1e-8
        )

        fig_radar = go.Figure()

        for seg in selected_segments:
            vals = seg_norm.loc[seg].tolist()
            vals += vals[:1]

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=radar_features + [radar_features[0]],
                    fill="toself",
                    name=f"Segment {seg}",
                    opacity=0.7,
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("### 🔥 Segment Feature Overview (Heatmap)")

    seg_profile = (
        cust.groupby("segment")[radar_features]
        .mean()
        .reset_index()
    )
    heatmap_df = seg_profile.set_index("segment")[radar_features]

    fig_heat = px.imshow(
        heatmap_df,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        labels=dict(color="Value"),
        title="Segment Feature Comparison",
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### 📊 Key Metrics by Segment")

    col1, col2, col3 = st.columns(3)

    fig_monetary = px.bar(
        cust.groupby("segment")["monetary"].mean().reset_index(),
        x="segment",
        y="monetary",
        title="Avg Monetary Value",
    )
    col1.plotly_chart(fig_monetary, use_container_width=True)

    fig_recency = px.bar(
        cust.groupby("segment")["recency_days"].mean().reset_index(),
        x="segment",
        y="recency_days",
        title="Avg Recency (days)",
    )
    col2.plotly_chart(fig_recency, use_container_width=True)

    fig_freq = px.bar(
        cust.groupby("segment")["frequency"].mean().reset_index(),
        x="segment",
        y="frequency",
        title="Avg Frequency",
    )
    col3.plotly_chart(fig_freq, use_container_width=True)

    st.markdown("### ⚠️ Churn Rate by Segment")

    seg_churn = cust.groupby("segment")["churned"].mean().reset_index()

    fig_seg_churn = px.bar(
        seg_churn,
        x="segment",
        y="churned",
        title="Churn Rate by Segment",
        labels={"churned": "Churn Rate"},
    )

    st.plotly_chart(fig_seg_churn, use_container_width=True)


# ============================================================
# CHURN MODEL TAB
# ============================================================

with tabs[2]:

    st.subheader("⚠️ Churn Prediction Model")

    roc_col, fi_col = st.columns(2)

    with roc_col:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Baseline",
                line=dict(dash="dash"),
            )
        )
        fig_roc.update_layout(title=f"ROC Curve (AUC = {auc:.3f})")
        st.plotly_chart(fig_roc, use_container_width=True)

    with fi_col:
        fi_df = (
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": rf.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
        )

        fig_fi = px.bar(
            fi_df,
            x="importance",
            y="feature",
            title="Feature Importance",
            orientation="h",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with st.expander("🔍 Inspect Individual Customer"):
        cid = st.selectbox(
            "Select Customer ID",
            sorted(cust["customer_id"].unique()),
        )
        row = cust[cust["customer_id"] == cid]
        if row.empty:
            st.error("Customer not found. Try another.")
        else:
            doc = row.iloc[0]
            st.json(
                {
                    "Customer ID": cid,
                    "Churn": int(doc["churned"]),
                    "Churn Probability": round(doc["churn_prob"], 3),
                    "Recency": doc["recency_days"],
                    "Frequency": doc["frequency"],
                    "Monetary": doc["monetary"],
                }
            )


# ============================================================
# CUSTOMER INSIGHTS TAB
# ============================================================

with tabs[3]:

    st.subheader("🛒 Customer Insights & Recommendations")

    cid = st.selectbox(
        "Select Customer",
        sorted(cust["customer_id"].unique()),
    )

    row = cust[cust["customer_id"] == cid]
    if row.empty:
        st.error("Customer not found. Try another.")
    else:
        cdf = row.iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{cdf['churn_prob']:.2f}")
        c2.metric("Recency (days)", int(cdf["recency_days"]))
        c3.metric("Total Spend", f"${cdf['monetary']:.2f}")

        st.markdown("### Last Purchased Items")

        related_ids = cust_map.loc[
            cust_map["unique_id"] == cid, "order_customer_id"
        ].unique()

        if len(related_ids) == 0:
            st.info("No purchase history found for this customer.")
            last_items = pd.DataFrame()
        else:
            customer_orders = orders.loc[
                orders["customer_id"].isin(related_ids), "order_id"
            ].unique()

            if len(customer_orders) == 0:
                st.info("No purchase history found for this customer.")
                last_items = pd.DataFrame()
            else:
                # MERGE ENGLISH CATEGORY NAME
                last_items = (
                    order_items[order_items["order_id"].isin(customer_orders)]
                    .merge(
                        products[["product_id", "product_category_name_english"]],
                        on="product_id",
                        how="left"
                    )
                )

                if last_items.empty:
                    st.info("No items found for this customer's orders.")
                else:
                    st.dataframe(
                        last_items.sort_values("order_id", ascending=False)[
                            ["order_id", "product_id",
                             "product_category_name_english", "price"]
                        ].head(5)
                    )

        st.markdown("### Recommended Cross-Sell Items")

        if last_items.empty:
            bought = set()
        else:
            bought = set(last_items["product_id"].unique())

        subset = pd.DataFrame()
        if not rules.empty and bought:
            subset = rules[rules["antecedent_pid"].isin(bought)][
                ["antecedent_pid", "consequent_pid", "lift", "confidence", "support"]
            ]

        if subset.empty:
            st.info("No direct cross-sell rules found. Showing fallback suggestions.")

            if bought:
                bought_cats = (
                    products[products["product_id"].isin(bought)][
                        "product_category_name_english"
                    ]
                    .dropna()
                    .unique()
                )

                fallback = products[
                    products["product_category_name_english"].isin(bought_cats)
                ][["product_id", "product_category_name_english"]].drop_duplicates().head(10)

                st.dataframe(fallback)

            else:
                top_global = (
                    order_items["product_id"].value_counts().head(10).reset_index()
                )
                top_global.columns = ["product_id", "order_count"]

                fallback = top_global.merge(
                    products[["product_id", "product_category_name_english"]],
                    on="product_id",
                    how="left",
                )

                st.dataframe(fallback)

        else:
            # MERGE ENGLISH NAME FOR RECOMMENDED PRODUCTS
            rec = subset.merge(
                products[["product_id", "product_category_name_english"]],
                left_on="consequent_pid",
                right_on="product_id",
                how="left",
            )

            st.dataframe(
                rec.sort_values("lift", ascending=False)[[
                    "consequent_pid",
                    "product_category_name_english",
                    "lift", "confidence", "support"
                ]].head(10)
            )

        # NOTIFICATION DAYS
        st.metric(
            "Recommended Notification Timing",
            f"{int(cdf['notify_day'])} days"
            if pd.notnull(cdf["notify_day"])
            else "N/A",
            help="Send a marketing message around this day to pre-empt churn.",
        )

        # CALCULATE NOTIFICATION DATE
        st.markdown("### 📅 Notification Date")

        last_order_date = orders.loc[
            orders["customer_id"].isin(related_ids),
            "order_purchase_timestamp"
        ].max()

        if pd.notnull(last_order_date) and pd.notnull(cdf["notify_day"]):
            notify_date = last_order_date + timedelta(days=int(cdf["notify_day"]))
            st.write(f"**Next notification should be sent on:** {notify_date.date()}")
        else:
            st.write("Not available.")


# ============================================================
# HIGH VALUE CUSTOMERS
# ============================================================

with tabs[4]:

    st.subheader("💎 Top 50 High-Value Customers")
    hv = cust.sort_values("monetary", ascending=False).head(50)
    st.dataframe(
        hv[
            [
                "customer_id",
                "monetary",
                "frequency",
                "recency_days",
                "churn_prob",
            ]
        ]
    )
