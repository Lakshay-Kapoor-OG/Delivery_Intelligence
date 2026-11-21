import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ---------- Page configuration ----------
st.set_page_config(
    page_title="Delivery Intelligence | Operations",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual system ----------
BG = "#080E19"
PANEL = "#111B2A"
PANEL_ALT = "#142235"
BORDER = "#26384D"
TEXT = "#EAF4FF"
MUTED = "#8DA4BB"
TEAL = "#20D9D2"
BLUE = "#16A7E8"
AMBER = "#20D9D2"
CORAL = "#38C7C1"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {{
        --bg: {BG};
        --panel: {PANEL};
        --border: {BORDER};
        --text: {TEXT};
        --muted: {MUTED};
        --cyan: #19C8F2;
        --teal: #20D9D2;
    }}
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}
    .stApp {{
        background:
          radial-gradient(ellipse at 78% 3%, rgba(0,176,222,0.12), transparent 31%),
          radial-gradient(ellipse at 14% 42%, rgba(22,167,232,0.045), transparent 36%),
          var(--bg);
        color: var(--text);
    }}
    [data-testid="stHeader"] {{
        background: rgba(8,14,25,0.82);
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0C1523 0%, #0A1320 100%);
        border-right: 1px solid #1C2C40;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.2rem;
    }}
    [data-testid="stSidebar"] h2 {{
        font-family: 'Manrope', sans-serif;
        font-size: 0.98rem;
        letter-spacing: -0.02em;
    }}
    h1, h2, h3 {{
        font-family: 'Manrope', sans-serif !important;
        color: #EAF4FF !important;
        letter-spacing: -0.035em;
    }}
    h1 {{
        font-size: clamp(2rem, 3vw, 2.65rem) !important;
        font-weight: 800 !important;
        line-height: 1.12 !important;
    }}
    h2 {{ font-size: 1.3rem !important; font-weight: 700 !important; }}
    h3 {{ font-size: 1rem !important; font-weight: 700 !important; }}
    p, label, .stCaption {{ color: var(--muted); }}

    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, rgba(22,36,55,0.96), rgba(14,25,40,0.96));
        border: 1px solid #2A3C53;
        border-radius: 13px;
        padding: 1rem 1.05rem;
        min-height: 116px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.13);
        position: relative;
        overflow: hidden;
    }}
    [data-testid="stMetric"]::after {{
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, #19C8F2, #20D9D2);
        opacity: 0.9;
    }}
    [data-testid="stMetricLabel"] {{
        color: #91A8BF !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }}
    [data-testid="stMetricValue"] {{
        color: #F4FAFF !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: clamp(1.45rem, 2vw, 1.9rem) !important;
        font-weight: 800 !important;
    }}
    [data-testid="stMetricDelta"] {{ font-size: 0.72rem !important; }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {{
        background: #111D2D;
        border-color: #2A3B50;
        border-radius: 9px;
    }}
    [data-baseweb="tag"] {{
        background-color: #17394B !important;
        color: #C8F7FF !important;
        border: 1px solid #24566A !important;
    }}
    .stButton button, .stFormSubmitButton button {{
        border-radius: 9px;
        border: 1px solid rgba(25,200,242,0.48);
        background: linear-gradient(135deg, #078DBB, #087F91);
        color: #F4FCFF;
        font-weight: 700;
        padding: 0.62rem 1.05rem;
        box-shadow: 0 4px 18px rgba(0,174,220,0.12);
        transition: all 0.18s ease;
    }}
    .stButton button:hover, .stFormSubmitButton button:hover {{
        border-color: #62E9F7;
        background: linear-gradient(135deg, #0BA8D7, #0B9C9B);
        color: white;
        transform: translateY(-1px);
    }}
    hr {{
        border-color: #233449;
        margin: 1.45rem 0;
    }}
    .eyebrow {{
        color: #20D9D2;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.69rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }}
    .hero-copy {{
        color: #8DA4BB;
        font-size: 0.94rem;
        max-width: 760px;
        line-height: 1.65;
        margin-top: 0.45rem;
    }}
    .section-kicker {{
        color: #20D9D2;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-size: 0.67rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }}
    .insight-card {{
        background: linear-gradient(135deg, rgba(17,43,61,0.92), rgba(15,28,45,0.96));
        border: 1px solid #254359;
        border-radius: 13px;
        padding: 1rem 1.1rem;
        color: #D8E8F4;
        min-height: 100px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
    }}
    .insight-label {{
        color: #20D9D2;
        font-size: 0.68rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        margin-bottom: 0.35rem;
    }}
    .insight-value {{
        color: #F2FAFF;
        font-family: 'Manrope', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }}
    .insight-detail {{
        color: #91A8BF;
        font-size: 0.8rem;
        line-height: 1.45;
    }}
    .prediction-result {{
        background: linear-gradient(120deg, rgba(9,92,119,0.32), rgba(17,41,58,0.72));
        border: 1px solid rgba(32,217,210,0.38);
        border-radius: 14px;
        padding: 1.2rem 1.35rem;
        margin-top: 1rem;
        box-shadow: 0 8px 28px rgba(0,0,0,0.12);
    }}
    .prediction-result .label {{
        color: #20D9D2;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.68rem;
        font-weight: 800;
    }}
    .prediction-result .value {{
        color: #F4FBFF;
        font-family: 'Manrope', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0.25rem 0;
    }}
    .prediction-result .note {{
        color: #91A8BF;
        font-size: 0.8rem;
    }}
    .footer-note {{
        color: #71869C;
        font-size: 0.75rem;
        line-height: 1.6;
    }}
    [data-testid="stPlotlyChart"] {{
        background: linear-gradient(145deg, rgba(15,25,40,0.42), rgba(10,17,29,0.18));
        border: 1px solid rgba(39,59,80,0.45);
        border-radius: 13px;
        padding: 0.25rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Data and model ----------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_DIR / "delivery_data_engineered.csv")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "hist_gradient_boosting_delivery_time.joblib")

df = load_data()
TARGET = "Time_taken (min)"

required_columns = [
    TARGET, "City", "Road_traffic_density", "Weather_conditions",
    "Order_Hour", "Type_of_order", "Type_of_vehicle", "Festival",
    "Delivery_person_Age", "Delivery_person_Ratings", "Vehicle_condition",
    "multiple_deliveries", "Day_of_Week", "Is_Weekend",
    "Estimated_Distance_KM",
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error("The dataset is missing required columns: " + ", ".join(missing_columns))
    st.stop()

# ---------- Sidebar filters ----------
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:0.15rem 0 1.1rem 0">
          <div style="font-size:1.45rem">📦</div>
          <div style="font-family:Manrope,sans-serif;font-size:1.12rem;font-weight:800;color:#F8FAFC">
            Delivery Intelligence
          </div>
          <div style="font-size:0.78rem;color:{MUTED};margin-top:0.2rem">
            Operations workspace
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### Refine the data")
    st.caption("Leave a filter empty to include every category.")

    cities = sorted(df["City"].dropna().astype(str).unique().tolist())
    traffic_options = sorted(df["Road_traffic_density"].dropna().astype(str).unique().tolist())
    weather_options = sorted(df["Weather_conditions"].dropna().astype(str).unique().tolist())

    selected_cities = st.multiselect("City", cities, default=[], placeholder="All cities")
    selected_traffic = st.multiselect("Traffic density", traffic_options, default=[], placeholder="All traffic levels")
    selected_weather = st.multiselect("Weather", weather_options, default=[], placeholder="All weather conditions")

    st.divider()
    st.markdown(
        f"""
        <div style="font-size:0.76rem;color:{MUTED};line-height:1.65">
          <b style="color:#DCE6F2">About this workspace</b><br>
          Historical delivery analytics and an ML-based time estimate.<br><br>
          <span style="color:{TEAL};font-weight:700">●</span> Public dataset prototype
        </div>
        """,
        unsafe_allow_html=True,
    )

filtered = df.copy()
if selected_cities:
    filtered = filtered[filtered["City"].astype(str).isin(selected_cities)]
if selected_traffic:
    filtered = filtered[filtered["Road_traffic_density"].astype(str).isin(selected_traffic)]
if selected_weather:
    filtered = filtered[filtered["Weather_conditions"].astype(str).isin(selected_weather)]

filtered = filtered.dropna(subset=[TARGET])
if filtered.empty:
    st.warning("No orders match these filters. Clear one or more selections and try again.")
    st.stop()

# ---------- Plotly theme helper ----------
def polish_chart(fig, height=330):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=12, t=22, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#CBD5E1", size=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"),
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#334155",
            font=dict(color="#F8FAFC"),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#263449",
        tickfont=dict(color="#A8B5C8"),
        title_font=dict(color="#A8B5C8"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(94,139,169,0.15)",
        zeroline=False,
        linecolor="#263449",
        tickfont=dict(color="#A8B5C8"),
        title_font=dict(color="#A8B5C8"),
    )
    return fig

# ---------- Header ----------
st.markdown('<div class="eyebrow">DELIVERY PERFORMANCE · ANALYTICS</div>', unsafe_allow_html=True)
st.title("Delivery Operations Intelligence")
st.markdown(
    '<div class="hero-copy">A clearer view of delivery performance, operating conditions, and estimated delivery duration across the historical dataset.</div>',
    unsafe_allow_html=True,
)

st.markdown("")
target = filtered[TARGET]
average = target.mean()
median = target.median()
over_30 = target.gt(30).mean() * 100
p90 = target.quantile(0.90)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Orders in view", f"{len(filtered):,}", help="Orders remaining after the selected filters.")
k2.metric("Average delivery", f"{average:.1f} min")
k3.metric("Median delivery", f"{median:.0f} min")
k4.metric("Over 30 minutes", f"{over_30:.1f}%", help="30 minutes is an analytical threshold, not a verified service-level agreement.")

# Insight strip
traffic_means = filtered.groupby("Road_traffic_density")[TARGET].mean().dropna()
weather_means = filtered.groupby("Weather_conditions")[TARGET].mean().dropna()
if not traffic_means.empty:
    slow_traffic = traffic_means.idxmax()
    slow_traffic_mean = traffic_means.max()
else:
    slow_traffic, slow_traffic_mean = "N/A", np.nan
if not weather_means.empty:
    slow_weather = weather_means.idxmax()
    slow_weather_mean = weather_means.max()
else:
    slow_weather, slow_weather_mean = "N/A", np.nan

st.markdown("")
i1, i2, i3 = st.columns(3)
with i1:
    st.markdown(
        f'<div class="insight-card"><div class="insight-label">90th percentile</div><div class="insight-value">{p90:.0f} min</div><div class="insight-detail">Nine in ten recorded deliveries were at or below this duration in the current view.</div></div>',
        unsafe_allow_html=True,
    )
with i2:
    st.markdown(
        f'<div class="insight-card"><div class="insight-label">Slowest traffic group</div><div class="insight-value">{slow_traffic}</div><div class="insight-detail">Average recorded duration: {slow_traffic_mean:.1f} minutes.</div></div>',
        unsafe_allow_html=True,
    )
with i3:
    st.markdown(
        f'<div class="insight-card"><div class="insight-label">Slowest weather group</div><div class="insight-value">{slow_weather}</div><div class="insight-detail">Average recorded duration: {slow_weather_mean:.1f} minutes.</div></div>',
        unsafe_allow_html=True,
    )

st.divider()
st.markdown('<div class="section-kicker">01 / Operating conditions</div>', unsafe_allow_html=True)
st.header("What affects delivery duration?")
left, right = st.columns(2, gap="large")

traffic_order = ["Low", "Medium", "High", "Jam"]
traffic_summary = (
    filtered.groupby("Road_traffic_density")[TARGET]
    .agg(["mean", "count"])
    .reset_index()
)
traffic_summary.columns = ["Traffic", "Average minutes", "Orders"]
traffic_summary["Traffic"] = traffic_summary["Traffic"].astype(str)
traffic_summary["_order"] = traffic_summary["Traffic"].map({v: i for i, v in enumerate(traffic_order)}).fillna(99)
traffic_summary = traffic_summary.sort_values("_order")

with left:
    st.subheader("Average time by traffic")
    fig = px.bar(
        traffic_summary,
        x="Traffic",
        y="Average minutes",
        color="Traffic",
        color_discrete_map={"Low": "#20D9D2", "Medium": "#19C8F2", "High": "#168FC0", "Jam": "#126B88"},
        hover_data={"Orders": ":,", "Average minutes": ":.1f", "_order": False},
        text=traffic_summary["Average minutes"].map(lambda x: f"{x:.1f} min"),
        category_orders={"Traffic": traffic_order},
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    fig.update_layout(showlegend=False, yaxis_title="Minutes", xaxis_title="")
    fig = polish_chart(fig, 340)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

weather_summary = (
    filtered.groupby("Weather_conditions")[TARGET]
    .agg(["mean", "count"])
    .reset_index()
)
weather_summary.columns = ["Weather", "Average minutes", "Orders"]
weather_summary = weather_summary.sort_values("Average minutes", ascending=True)

with right:
    st.subheader("Average time by weather")
    fig = px.bar(
        weather_summary,
        x="Average minutes",
        y="Weather",
        orientation="h",
        color="Average minutes",
        color_continuous_scale=["#A5B4FC", "#818CF8", "#FB7185"],
        hover_data={"Orders": ":,", "Average minutes": ":.1f"},
        text=weather_summary["Average minutes"].map(lambda x: f"{x:.1f}"),
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Minutes",
        yaxis_title="",
        yaxis=dict(categoryorder="array", categoryarray=weather_summary["Weather"].tolist()),
    )
    fig = polish_chart(fig, 340)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown('<div class="section-kicker">02 / Delivery patterns</div>', unsafe_allow_html=True)
st.header("When are deliveries taking longer?")
left, right = st.columns(2, gap="large")

with left:
    st.subheader("Delivery duration distribution")
    fig = px.histogram(
        filtered,
        x=TARGET,
        nbins=28,
        color_discrete_sequence=["#19C8F2"],
        labels={TARGET: "Delivery duration (minutes)", "count": "Orders"},
    )
    fig.update_traces(marker_line_color=BG, marker_line_width=1, opacity=0.92)
    fig.update_layout(xaxis_title="Minutes", yaxis_title="Number of orders", bargap=0.06)
    fig = polish_chart(fig, 350)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with right:
    st.subheader("Average time by order hour")
    hourly = (
        filtered.dropna(subset=["Order_Hour"])
        .groupby("Order_Hour")[TARGET]
        .agg(["mean", "count"])
        .reset_index()
    )
    hourly.columns = ["Order hour", "Average minutes", "Orders"]
    hourly["Order hour"] = pd.to_numeric(hourly["Order hour"], errors="coerce")
    hourly = hourly.dropna(subset=["Order hour"]).sort_values("Order hour")
    # Keep missing hours as gaps instead of drawing a misleading continuous line.
    all_hours = pd.DataFrame({"Order hour": range(24)})
    hourly_full = all_hours.merge(hourly, on="Order hour", how="left")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_full["Order hour"],
        y=hourly_full["Average minutes"],
        mode="lines+markers",
        connectgaps=False,
        line=dict(color="#19C8F2", width=2.5),
        marker=dict(size=7, color="#20D9D2", line=dict(color=BG, width=1.5)),
        customdata=hourly_full[["Orders"]].to_numpy(),
        hovertemplate="Hour: %{x}:00<br>Average: %{y:.1f} min<br>Orders: %{customdata[0]:,.0f}<extra></extra>",
    ))
    fig.update_xaxes(dtick=2, range=[0, 23], title="Order hour")
    fig.update_yaxes(title="Average minutes")
    fig = polish_chart(fig, 350)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with st.expander("View the data behind these charts"):
    st.caption("Summaries reflect the current filter selection. These are historical associations, not evidence of causation.")
    tab1, tab2, tab3 = st.tabs(["Traffic", "Weather", "Hourly"])
    with tab1:
        st.dataframe(traffic_summary.drop(columns=["_order"], errors="ignore"), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(weather_summary, use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(hourly, use_container_width=True, hide_index=True)

st.divider()
st.markdown('<div class="section-kicker">03 / Machine learning</div>', unsafe_allow_html=True)
st.header("Estimate a delivery")
st.markdown(
    "Adjust the order and operating conditions below to generate an estimate from the saved machine-learning model."
)

try:
    model = load_model()
    with st.container(border=True):
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3, gap="large")
            with c1:
                st.markdown("#### Delivery partner")
                age = st.number_input("Age", min_value=18, max_value=60, value=30, step=1)
                rating = st.number_input("Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1, format="%.1f")
                vehicle_condition = st.selectbox("Vehicle condition", [0, 1, 2, 3], index=2)
                vehicle_type = st.selectbox("Vehicle type", sorted(df["Type_of_vehicle"].dropna().astype(str).unique()))
            with c2:
                st.markdown("#### Order conditions")
                weather = st.selectbox("Weather", weather_options, index=weather_options.index("Sunny") if "Sunny" in weather_options else 0)
                traffic = st.selectbox("Traffic density", traffic_options, index=traffic_options.index("Medium") if "Medium" in traffic_options else 0)
                multiple_deliveries = st.selectbox("Number of deliveries", [0, 1, 2, 3], index=1)
                festival_options = sorted(df["Festival"].dropna().astype(str).unique())
                festival = st.selectbox("Festival", festival_options, index=festival_options.index("No") if "No" in festival_options else 0)
            with c3:
                st.markdown("#### Location & timing")
                order_type = st.selectbox("Order type", sorted(df["Type_of_order"].dropna().astype(str).unique()))
                city = st.selectbox("City", cities, index=cities.index("Metropolitan") if "Metropolitan" in cities else 0)
                day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day = st.selectbox("Day of week", day_options, index=day_options.index("Wednesday"))
                order_hour = st.slider("Order hour", 0, 23, 18, help="Hour in 24-hour time.")
                distance = st.number_input("Estimated straight-line distance (km)", min_value=0.1, max_value=25.0, value=5.0, step=0.5)

            submitted = st.form_submit_button("Generate delivery estimate", use_container_width=True)

    if submitted:
        prediction_input = pd.DataFrame([{
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": rating,
            "Weather_conditions": weather,
            "Road_traffic_density": traffic,
            "Vehicle_condition": vehicle_condition,
            "Type_of_order": order_type,
            "Type_of_vehicle": vehicle_type,
            "multiple_deliveries": multiple_deliveries,
            "Festival": festival,
            "City": city,
            "Day_of_Week": day,
            "Is_Weekend": int(day in ["Saturday", "Sunday"]),
            "Order_Hour": order_hour,
            "Estimated_Distance_KM": distance,
        }])
        estimate = float(model.predict(prediction_input)[0])
        st.markdown(
            f"""
            <div class="prediction-result">
              <div class="label">Estimated delivery duration</div>
              <div class="value">{estimate:.1f} <span style="font-size:1rem;font-weight:600;color:#A8B5C8">minutes</span></div>
              <div class="note">Model-generated estimate based on the selected inputs. Actual delivery time can differ.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
except Exception as exc:
    st.error(f"The prediction section could not load: {exc}")
    st.caption("Check that the saved model and the installed scikit-learn version are compatible.")

st.markdown("")
st.markdown(
    f"""
    <div class="footer-note">
      <b style="color:#A8B5C8">Method note</b><br>
      This is an analytical prototype using a public historical dataset. Distance is a straight-line estimate where coordinates were considered valid, not a road-route distance. The 30-minute threshold is analytical rather than a verified SLA. Historical associations do not establish causation, and model performance may differ in live operations.
    </div>
    """,
    unsafe_allow_html=True,
)
