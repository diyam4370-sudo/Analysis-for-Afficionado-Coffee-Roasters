
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Afficionado Coffee Roasters", layout="wide")

DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
BUCKETS = ["Morning (6-11)", "Afternoon (12-16)", "Evening (17-21)", "Late (22-5)"]


@st.cache_data(show_spinner="Loading data...")
def load(source, anchor):
    df = pd.read_excel(source) if str(getattr(source, "name", source)).endswith(("xlsx", "xls")) else pd.read_csv(source)
    t = pd.to_datetime(df["transaction_time"].astype(str), format="%H:%M:%S")
    df["hour"] = t.dt.hour
    df = df.sort_values("transaction_id").reset_index(drop=True)
    reset = t.groupby(df["store_id"]).diff().dt.total_seconds() < -3600
    df["day_idx"] = reset.groupby(df["store_id"]).cumsum()
    df["date"] = pd.Timestamp(anchor) + pd.to_timedelta(df["day_idx"], unit="D")
    df["revenue"] = df["transaction_qty"] * df["unit_price"]
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    df["time_bucket"] = pd.cut(df["hour"], [-1, 5, 11, 16, 21, 23],
                               labels=["L1", BUCKETS[0], BUCKETS[1], BUCKETS[2], "L2"]).astype(str)
    df["time_bucket"] = df["time_bucket"].replace({"L1": BUCKETS[3], "L2": BUCKETS[3]})
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    return df


def validate(df):
    return {
        "Rows": f"{len(df):,}",
        "Missing values": int(df.isna().sum().sum()),
        "Duplicate transaction IDs": int(df["transaction_id"].duplicated().sum()),
        "Non-positive quantity": int((df["transaction_qty"] <= 0).sum()),
        "Non-positive price": int((df["unit_price"] <= 0).sum()),
        "Days reconstructed": int(df["day_idx"].max() + 1),
    }


# ---------------- Sidebar ----------------
st.sidebar.title("Filters")
up = st.sidebar.file_uploader("Dataset (xlsx/csv)", type=["xlsx", "csv"])
source = up if up else "Afficionado_Coffee_Roasters.xlsx"
anchor = st.sidebar.date_input("Date of first day in data", pd.Timestamp("2025-01-01"))
try:
    raw = load(source, str(anchor))
except Exception as e:
    st.error(f"Could not load data ({e}). Upload Afficionado_Coffee_Roasters.xlsx in the sidebar.")
    st.stop()

stores = st.sidebar.multiselect("Store location", sorted(raw.store_location.unique()),
                                default=sorted(raw.store_location.unique()))
days = st.sidebar.multiselect("Day of week", DOW, default=DOW)
h0, h1 = st.sidebar.slider("Hour range", 0, 23, (6, 20))
metric_label = st.sidebar.radio("Metric", ["Revenue", "Quantity"], horizontal=True)
metric, unit = ("revenue", "Revenue ($)") if metric_label == "Revenue" else ("transaction_qty", "Units sold")

df = raw[raw.store_location.isin(stores) & raw.day_of_week.isin(days) & raw.hour.between(h0, h1)]
if df.empty:
    st.warning("No data for the current filters.")
    st.stop()

st.title("Afficionado Coffee Roasters: when do sales happen?")
st.caption(f"Dates are reconstructed (data has no date column); day 1 = {anchor}. "
           "Weekday labels shift if the anchor is wrong.")

k = st.columns(4)
k[0].metric("Revenue", f"${df.revenue.sum():,.0f}")
k[1].metric("Transactions", f"{len(df):,}")
k[2].metric("Units", f"{df.transaction_qty.sum():,}")
k[3].metric("Avg ticket", f"${df.revenue.sum() / len(df):.2f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Sales trend", "Day of week", "Hourly demand", "Location comparison", "Data quality"])

# ---------------- Trend ----------------
with tab1:
    daily = df.groupby("date")[metric].sum().reset_index()
    daily["7-day avg"] = daily[metric].rolling(7, min_periods=1).mean()
    fig = px.line(daily, x="date", y=[metric, "7-day avg"], labels={"value": unit, "date": ""})
    fig.update_layout(legend_title="", title="Daily trend")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    wk = df.groupby("week").agg(Revenue=("revenue", "sum"), Transactions=("transaction_id", "count"),
                                Days=("date", "nunique")).reset_index()
    wk = wk[wk.Days == 7]  # drop partial weeks so the trend isn't distorted
    c1.plotly_chart(px.bar(wk, x="week", y="Revenue", title="Weekly revenue (full weeks only)"),
                    use_container_width=True)
    c2.plotly_chart(px.bar(wk, x="week", y="Transactions", title="Weekly transactions (full weeks only)"),
                    use_container_width=True)
    ws = df.groupby(["week", "store_location"])[metric].sum().reset_index()
    full = wk.week
    ws = ws[ws.week.isin(full)]
    st.plotly_chart(px.line(ws, x="week", y=metric, color="store_location",
                            title="Weekly trend by store", labels={metric: unit}), use_container_width=True)
    if len(daily) > 2:
        slope = np.polyfit(np.arange(len(daily)), daily[metric], 1)[0]
        st.info(f"Linear trend: {slope:+,.1f} {unit.split()[0].lower()} per day.")

# ---------------- Day of week ----------------
with tab2:
    perday = df.groupby(["date", "day_of_week"]).agg(revenue=("revenue", "sum"),
                                                     transactions=("transaction_id", "count"),
                                                     qty=("transaction_qty", "sum")).reset_index()
    col = {"revenue": "revenue", "transaction_qty": "qty"}[metric]
    avg = perday.groupby("day_of_week")[[col, "transactions"]].mean().reindex(DOW).dropna().reset_index()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(avg, x="day_of_week", y=col, title=f"Average daily {metric_label.lower()}",
                           labels={col: unit, "day_of_week": ""}), use_container_width=True)
    c2.plotly_chart(px.bar(avg, x="day_of_week", y="transactions", title="Average daily transactions",
                           labels={"day_of_week": ""}), use_container_width=True)
    perday["Type"] = np.where(perday.day_of_week.isin(["Saturday", "Sunday"]), "Weekend", "Weekday")
    wk_cmp = perday.groupby("Type")[[col, "transactions"]].mean().round(1)
    st.subheader("Weekday vs weekend (average per day)")
    st.dataframe(wk_cmp)
    spread = (avg[col].max() - avg[col].min()) / avg[col].mean()
    st.caption(f"Gap between best and worst day: {spread:.1%} of the average day.")

# ---------------- Hourly ----------------
with tab3:
    hr = df.groupby("hour").agg(val=(metric, "sum"), tx=("transaction_id", "count")).reset_index()
    n_days = df.date.nunique()
    hr["per_day"] = hr.tx / n_days
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(hr, x="hour", y="per_day", markers=True,
                            title="Average transactions per hour of the day",
                            labels={"per_day": "Transactions / day"}), use_container_width=True)
    c2.plotly_chart(px.bar(hr, x="hour", y="val", title=f"Total {metric_label.lower()} by hour",
                           labels={"val": unit}), use_container_width=True)
    heat = df.pivot_table(index="day_of_week", columns="hour", values=metric, aggfunc="sum").reindex(DOW)
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="YlOrBr",
                              title=f"{metric_label}: day of week x hour",
                              labels=dict(x="Hour", y="", color=unit)), use_container_width=True)
    tb = df.groupby("time_bucket")[metric].sum().reindex(BUCKETS).dropna()
    st.plotly_chart(px.pie(values=tb.values, names=tb.index, hole=.45, title="Share by time bucket"),
                    use_container_width=True)

# ---------------- Locations ----------------
with tab4:
    st.caption("Heatmaps show each store's own hourly profile as % of its total, so stores of different size compare fairly.")
    for s in sorted(df.store_location.unique()):
        g = df[df.store_location == s]
        heat = g.pivot_table(index="day_of_week", columns="hour", values=metric, aggfunc="sum").reindex(DOW)
        st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="YlOrBr",
                                  title=s, labels=dict(x="Hour", y="", color=unit)),
                        use_container_width=True)
    prof = df.groupby(["store_location", "hour"])[metric].sum().reset_index()
    prof["share"] = prof[metric] / prof.groupby("store_location")[metric].transform("sum") * 100
    st.plotly_chart(px.line(prof, x="hour", y="share", color="store_location", markers=True,
                            title="Hourly profile (% of store total)", labels={"share": "% of store total"}),
                    use_container_width=True)
    summ = df.groupby("store_location").agg(Revenue=("revenue", "sum"), Transactions=("transaction_id", "count"))
    summ["Avg ticket"] = (summ.Revenue / summ.Transactions).round(2)
    summ["Peak hour"] = df.groupby(["store_location", "hour"]).size().groupby(level=0).idxmax().map(lambda x: x[1])
    bk = df.pivot_table(index="store_location", columns="time_bucket", values=metric, aggfunc="sum")
    bk = (bk.div(bk.sum(axis=1), axis=0) * 100).round(1).add_suffix(" %")
    st.dataframe(summ.join(bk))

# ---------------- Data quality ----------------
with tab5:
    st.json(validate(raw))
    st.write("Late-hours bucket (22-5) is empty: the earliest sale is 06:00 and the latest 20:59.")
    st.dataframe(raw.head(100))