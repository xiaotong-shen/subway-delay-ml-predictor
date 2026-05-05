import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TTC Delay Predictor",
    page_icon="subway",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sandy / Claude-inspired palette ───────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #faf8f4;
        color: #2c2825;
    }
    [data-testid="stMain"] { background-color: #faf8f4; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f2ede4 !important;
        border-right: 1px solid #e5ddd2;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p { color: #2c2825 !important; }

    /* Typography helpers */
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2c2825;
        letter-spacing: -0.02em;
        margin: 0 0 0.2rem 0;
    }
    .app-sub {
        font-size: 0.85rem;
        color: #6b6560;
        margin: 0 0 1.25rem 0;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid #e5ddd2;
    }
    .sec-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #9e9690;
        margin-bottom: 0.4rem;
        display: block;
    }

    /* Inline detail rows */
    .drow {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0;
        border-bottom: 1px solid #f0ebe3;
        font-size: 0.875rem;
    }
    .drow:last-child { border-bottom: none; }
    .dlabel { color: #6b6560; }
    .dval   { color: #2c2825; font-weight: 500; }

    /* Risk badge */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-low  { background: #e8f3ec; color: #3a7d55; }
    .badge-mod  { background: #fdf0e4; color: #b86b28; }
    .badge-high { background: #fce9e7; color: #b03a2e; }

    /* Progress bar */
    .pbar-wrap { background: #e5ddd2; border-radius: 4px; height: 7px;
                 margin: 0.6rem 0 1.1rem 0; overflow: hidden; }
    .pbar-fill { height: 100%; border-radius: 4px; }

    /* Sidebar hint box */
    .hint-box {
        margin-top: 1.5rem;
        padding: 0.75rem;
        background: #ede8df;
        border-radius: 8px;
        font-size: 0.78rem;
        color: #6b6560;
        line-height: 1.6;
    }
    .hint-box strong { color: #2c2825; }

    /* Footer */
    .footer {
        text-align: center;
        color: #9e9690;
        font-size: 0.78rem;
        padding: 1.5rem 0 0.25rem 0;
        border-top: 1px solid #e5ddd2;
        margin-top: 1.5rem;
    }

    /* Plotly container bg */
    .js-plotly-plot { background: transparent !important; }

    /* Streamlit widget tweaks */
    div[data-baseweb="select"] > div {
        background-color: #ffffff;
        border-color: #e5ddd2 !important;
    }
    .stMetric > div { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("src/routes/resources/enriched_predictions_full.csv")
        with open("src/routes/resources/station_data.json") as f:
            sd = json.load(f)
        return df, sd
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


# ── Map ───────────────────────────────────────────────────────────────────────
def create_delay_map(df, hour, month=None, dow=None):
    if df is None:
        return None

    filtered = df[df['hour'] == hour].copy()
    if month is not None:
        filtered = filtered[filtered['month'] == month]
    if dow is not None:
        filtered = filtered[filtered['day_of_week'] == dow]
    if filtered.empty:
        st.warning("No data for these filters.")
        return None

    agg = filtered.groupby(['station', 'latitude', 'longitude'], as_index=False).agg(
        likelihood_of_delay=('likelihood_of_delay', 'mean'),
        delay_severity=('delay_severity', lambda x: x.mode()[0]),
        delay_length=('delay_length', 'mean'),
    )

    mn, mx = agg['likelihood_of_delay'].min(), agg['likelihood_of_delay'].max()
    rng = mx - mn if mx > mn else 1
    agg['scaled_size'] = ((agg['likelihood_of_delay'] - mn) / rng * 40 + 8).clip(lower=4)

    fig = px.scatter_map(
        agg,
        lat='latitude', lon='longitude',
        size='scaled_size',
        color='likelihood_of_delay',
        hover_name='station',
        hover_data={
            'likelihood_of_delay': ':.0%',
            'delay_severity': True,
            'delay_length': ':.1f',
            'scaled_size': False,
            'latitude': False,
            'longitude': False,
        },
        color_continuous_scale=[
            [0.0, '#3a7d55'],
            [0.5, '#e8a944'],
            [1.0, '#c96442'],
        ],
        size_max=48,
        zoom=10.5,
        center={'lat': 43.6532, 'lon': -79.3832},
    )
    fig.update_layout(
        map_style='carto-positron',
        height=560,
        margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
        paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(
            title='Risk',
            tickformat='.0%',
            thickness=10, len=0.45,
            bgcolor='rgba(250,248,244,0.9)',
            bordercolor='#e5ddd2', borderwidth=1,
        ),
    )
    return fig


# ── Timeline ──────────────────────────────────────────────────────────────────
def create_timeline(df, station, month=None, dow=None):
    if df is None or station is None:
        return None

    data = df[df['station'] == station].copy()
    if month is not None:
        data = data[data['month'] == month]
    if dow is not None:
        data = data[data['day_of_week'] == dow]
    if data.empty:
        return None

    hourly = data.groupby('hour', as_index=False)['likelihood_of_delay'].mean().sort_values('hour')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly['hour'],
        y=hourly['likelihood_of_delay'],
        mode='lines+markers',
        line=dict(color='#c96442', width=2.5),
        marker=dict(size=6, color='#c96442', line=dict(color='#ffffff', width=1.5)),
        fill='tozeroy',
        fillcolor='rgba(201,100,66,0.07)',
        hovertemplate='%{x:02d}:00 — %{y:.0%}<extra></extra>',
    ))
    fig.update_layout(
        height=210,
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            tickmode='linear', tick0=6, dtick=3,
            gridcolor='#f0ebe3', showline=False, zeroline=False,
            tickfont=dict(size=10, color='#6b6560'),
        ),
        yaxis=dict(
            range=[0, 1], tickformat='.0%',
            gridcolor='#f0ebe3', showline=False, zeroline=False,
            tickfont=dict(size=10, color='#6b6560'),
        ),
        showlegend=False,
    )
    return fig


# ── Prediction lookup ─────────────────────────────────────────────────────────
def get_prediction(df, station, hour, month=None, dow=None):
    if df is None:
        return None
    mask = (df['station'] == station) & (df['hour'] == hour)
    if month is not None:
        mask &= df['month'] == month
    if dow is not None:
        mask &= df['day_of_week'] == dow
    rows = df[mask]
    if rows.empty:
        return None
    nums = rows.mean(numeric_only=True).to_dict()
    nums['station']        = station
    nums['delay_severity'] = rows['delay_severity'].mode()[0]
    return nums


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df, station_data = load_data()
    if df is None:
        st.error("Failed to load data.")
        return

    MONTHS   = ['January','February','March','April','May','June',
                'July','August','September','October','November','December']
    DAYS     = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    DAYS_S   = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<p class="app-title">TTC Delay Predictor</p>'
        '<p class="app-sub">Historical delay patterns for Toronto subway stations</p>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<span class="sec-label">Filters</span>', unsafe_allow_html=True)

        selected_hour = st.slider("Hour of day", 0, 23, 8, format="%02d:00")

        selected_month = st.selectbox(
            "Month",
            options=sorted(df['month'].unique()),
            format_func=lambda m: MONTHS[m - 1],
        )

        selected_day = st.selectbox("Day of week", DAYS)
        selected_dow = DAYS.index(selected_day)

        st.markdown('<span class="sec-label" style="margin-top:1rem;display:block">Station</span>', unsafe_allow_html=True)
        stations   = sorted(df['station'].unique())
        default_i  = stations.index('BLOOR-YONGE') if 'BLOOR-YONGE' in stations else 0
        selected_station = st.selectbox("Station", stations, index=default_i, label_visibility="collapsed")

        st.markdown(
            '<div class="hint-box"><strong>Legend</strong><br>'
            'Dot size and color both encode delay risk.<br>'
            '<span style="color:#3a7d55">&#9679;</span> Low &nbsp;'
            '<span style="color:#e8a944">&#9679;</span> Moderate &nbsp;'
            '<span style="color:#c96442">&#9679;</span> High</div>',
            unsafe_allow_html=True,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    col_map, col_detail = st.columns([3, 1], gap="large")

    with col_map:
        st.markdown('<span class="sec-label">Interactive Map</span>', unsafe_allow_html=True)
        map_fig = create_delay_map(df, selected_hour, selected_month, selected_dow)
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True, config={'displayModeBar': False})

    with col_detail:
        st.markdown('<span class="sec-label">Station Details</span>', unsafe_allow_html=True)

        pred = get_prediction(df, selected_station, selected_hour, selected_month, selected_dow)

        if pred:
            likelihood = pred['likelihood_of_delay']
            delay_len  = pred['delay_length']
            severity   = pred['delay_severity']

            if likelihood < 0.4:
                badge_cls, risk_label, bar_color = 'badge-low',  'Low Risk',      '#3a7d55'
            elif likelihood < 0.65:
                badge_cls, risk_label, bar_color = 'badge-mod',  'Moderate Risk', '#e8a944'
            else:
                badge_cls, risk_label, bar_color = 'badge-high', 'High Risk',     '#c96442'

            # Likelihood + badge
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(
                    f'<p style="font-size:2rem;font-weight:700;color:#2c2825;margin:0">'
                    f'{likelihood:.0%}</p>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div style="padding-top:0.6rem">'
                    f'<span class="badge {badge_cls}">{risk_label}</span></div>',
                    unsafe_allow_html=True,
                )

            # Progress bar
            st.markdown(
                f'<div class="pbar-wrap">'
                f'<div class="pbar-fill" style="width:{likelihood*100:.1f}%;background:{bar_color}"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Detail rows — one st.markdown per row to keep HTML simple
            rows = [
                ("Station",   selected_station),
                ("Time",      f"{selected_hour:02d}:00"),
                ("Day",       f"{DAYS_S[selected_dow]}, {MONTHS[selected_month-1][:3]}"),
                ("Severity",  severity),
                ("Avg delay", f"{delay_len:.1f} min"),
            ]
            for label, val in rows:
                st.markdown(
                    f'<div class="drow">'
                    f'<span class="dlabel">{label}</span>'
                    f'<span class="dval">{val}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<p style="color:#6b6560;font-size:0.875rem">No data for these filters.</p>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<span class="sec-label" style="margin-top:1rem;display:block">Daily pattern</span>',
            unsafe_allow_html=True,
        )
        tl = create_timeline(df, selected_station, selected_month, selected_dow)
        if tl:
            st.plotly_chart(tl, use_container_width=True, config={'displayModeBar': False})

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer">TTC Subway Delay Predictor &middot; '
        'Random Forest model trained on 2021–2024 TTC delay data</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
