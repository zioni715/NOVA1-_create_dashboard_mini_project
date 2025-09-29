# app.py
from typing import Optional
from pathlib import Path
import json

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# --------------------------------------
# Page config (다크테마는 config.toml로 적용됨)
# --------------------------------------
st.set_page_config(
    page_title="Consumption Analysis Dashboard",
    page_icon="💲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 제목 이탤릭
st.markdown(
    "<h1 style='font-style: italic; margin: 0 0 .75rem 0;'>💲Consumption Analysis Dashboard💲</h1>",
    unsafe_allow_html=True,
)

# --------------------------------------
# Seoul GeoJSON (공식 경로) - 자동 다운로드
# --------------------------------------
SEOUL_GEO_URL = (
    "https://raw.githubusercontent.com/southkorea/seoul-maps/master/"
    "kostat/2013/json/seoul_municipalities_geo_simple.json"
)
LOCAL_GEO_PATH = "data/seoul_municipalities_geo_simple.json"

def ensure_seoul_geojson(local_path: str = LOCAL_GEO_PATH) -> Optional[str]:
    """로컬에 GeoJSON이 없으면 공식 저장소에서 자동 다운로드."""
    p = Path(local_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            r = requests.get(SEOUL_GEO_URL, timeout=20)
            r.raise_for_status()
            p.write_bytes(r.content)
        return str(p)
    except Exception as e:
        st.error(f"GeoJSON auto-download failed: {e}")
        return None

# --------------------------------------
# 데이터 로딩 (CSV 또는 MySQL)
# --------------------------------------
@st.cache_resource
def _mysql_engine(host: str, port: int, user: str, password: str, database: str):
    url = URL.create(
        "mysql+pymysql",
        username=user,
        port=port,
        password=password,
        host=host,
        database=database,
    )
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={"charset": "utf8mb4"},
    )

def load_data_from_mysql(host, port, user, password, database, query) -> pd.DataFrame:
    eng = _mysql_engine(host, port, user, password, database)
    return pd.read_sql(query, eng)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    required = ["CTY_RGN_NM", "ADMI_CTY_NM", "AGE_VAL", "AMT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    df = df.copy()
    df["AMT"] = pd.to_numeric(df["AMT"], errors="coerce")
    df = df.dropna(subset=["AMT"])
    df["CTY_RGN_NM"] = df["CTY_RGN_NM"].astype(str).str.strip()
    df["ADMI_CTY_NM"] = df["ADMI_CTY_NM"].astype(str).str.strip()
    df["AGE_VAL"] = df["AGE_VAL"].astype(str).str.strip()
    return df

# --------------------------------------
# Aggregations & Charts
# --------------------------------------
def amt_by_city_region(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("CTY_RGN_NM", as_index=False)["AMT"]
        .mean()
        .rename(columns={"AMT": "Average Consumption(city region)"})
    )

def plot_amt_by_city_region(df: pd.DataFrame):
    data = amt_by_city_region(df)
    fig = px.bar(
        data,
        x="CTY_RGN_NM",
        y="Average Consumption(city region)",
        title="Average Consumption by City Region",
        labels={
            "CTY_RGN_NM": "City Region",
            "Average Consumption(city region)": "Average Consumption",
        },
        template="plotly_dark",
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.update_yaxes(tickformat=",.0f")
    return fig

def amt_by_dong(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("ADMI_CTY_NM", as_index=False)["AMT"]
        .mean()
        .rename(columns={"AMT": "Average Consumption(dong)"})
    )

def plot_amt_by_dong(df: pd.DataFrame):
    data = amt_by_dong(df)
    fig = px.line(
        data,
        x="ADMI_CTY_NM",
        y="Average Consumption(dong)",
        markers=True,
        title="Average Consumption by Dong",
        labels={"ADMI_CTY_NM": "Dong", "Average Consumption(dong)": "Average Consumption"},
        template="plotly_dark",
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.update_yaxes(tickformat=",.0f")
    return fig

def amt_by_age(df: pd.DataFrame) -> pd.DataFrame:
    return df[["AGE_VAL", "AMT"]].copy()

def plot_amt_by_age(df: pd.DataFrame):
    data = amt_by_age(df)
    fig = px.scatter(
        data,
        x="AGE_VAL",
        y="AMT",
        opacity=0.6,
        title="Consumption by Age",
        labels={"AGE_VAL": "Age Group", "AMT": "Consumption"},
        template="plotly_dark",
    )
    fig.update_yaxes(tickformat=",.0f")
    return fig

# --------------------------------------
# Seoul-only map (파랑 음영 + 경계 흰색 + 구 라벨: 검정)
# --------------------------------------
KOR_TO_ENG = {
    "강남구": "Gangnam-gu", "강동구": "Gangdong-gu", "강북구": "Gangbuk-gu", "강서구": "Gangseo-gu",
    "관악구": "Gwanak-gu", "광진구": "Gwangjin-gu", "구로구": "Guro-gu", "금천구": "Geumcheon-gu",
    "노원구": "Nowon-gu", "도봉구": "Dobong-gu", "동대문구": "Dongdaemun-gu", "동작구": "Dongjak-gu",
    "마포구": "Mapo-gu", "서대문구": "Seodaemun-gu", "서초구": "Seocho-gu", "성동구": "Seongdong-gu",
    "성북구": "Seongbuk-gu", "송파구": "Songpa-gu", "양천구": "Yangcheon-gu", "영등포구": "Yeongdeungpo-gu",
    "용산구": "Yongsan-gu", "은평구": "Eunpyeong-gu", "종로구": "Jongno-gu", "중구": "Jung-gu", "중랑구": "Jungnang-gu",
}
GU_CENTROIDS = {
    "종로구": (37.573, 126.979), "중구": (37.5636, 126.997), "용산구": (37.532, 126.990),
    "성동구": (37.563, 127.036), "광진구": (37.538, 127.082), "동대문구": (37.574, 127.039),
    "중랑구": (37.606, 127.092), "성북구": (37.589, 127.017), "강북구": (37.639, 127.025),
    "도봉구": (37.669, 127.046), "노원구": (37.654, 127.056), "은평구": (37.617, 126.936),
    "서대문구": (37.582, 126.938), "마포구": (37.563, 126.908), "양천구": (37.516, 126.866),
    "강서구": (37.560, 126.822), "구로구": (37.495, 126.887), "금천구": (37.457, 126.895),
    "영등포구": (37.520, 126.910), "동작구": (37.512, 126.939), "관악구": (37.475, 126.953),
    "서초구": (37.483, 127.032), "강남구": (37.517, 127.047), "송파구": (37.514, 127.106),
    "강동구": (37.530, 127.123)
}

def plot_seoul_blue_thematic_map(df: pd.DataFrame):
    local_path = ensure_seoul_geojson()
    if not local_path:
        return None
    with open(local_path, "r", encoding="utf-8") as f:
        geo = json.load(f)

    gu_sum = (
        df.groupby("CTY_RGN_NM", as_index=False)["AMT"]
        .sum()
        .rename(columns={"AMT": "Total Amount"})
    )

    geo_names = {str(ft["properties"].get("name", "")).strip() for ft in geo.get("features", [])}
    direct = gu_sum["CTY_RGN_NM"].astype(str).str.strip()
    eng = gu_sum["CTY_RGN_NM"].map(KOR_TO_ENG).fillna(gu_sum["CTY_RGN_NM"]).astype(str).str.strip()
    use_english = (eng.isin(geo_names).sum() > direct.isin(geo_names).sum())

    plot_df = gu_sum.copy()
    plot_df["_LOC_"] = eng if use_english else direct

    fig = px.choropleth(
        plot_df,
        geojson=geo,
        locations="_LOC_",
        featureidkey="properties.name",
        color="Total Amount",
        color_continuous_scale="Blues",
        labels={"Total Amount": "Total Amount"},
        template="plotly_dark",
        title="Total Amount by City Region (Map)",
    )
    fig.update_traces(marker_line_color="rgba(255,255,255,0.95)", marker_line_width=1.5)
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=False,
        showcoastlines=False,
        showland=False,
        showframe=False,
        projection_type="mercator",
    )
    fig.update_coloraxes(colorbar_tickformat=",.0f", colorbar_title="Amount")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))

    centers = (
        pd.DataFrame.from_dict(GU_CENTROIDS, orient="index", columns=["lat", "lon"])
        .reset_index()
        .rename(columns={"index": "CTY_RGN_NM"})
    )
    centers = centers[centers["CTY_RGN_NM"].isin(set(gu_sum["CTY_RGN_NM"]))]

    fig.add_trace(
        go.Scattergeo(
            lon=centers["lon"],
            lat=centers["lat"],
            mode="text",
            text=centers["CTY_RGN_NM"],
            textfont=dict(
                size=12,
                color="black",
                family="Noto Sans KR, Malgun Gothic, Arial, sans-serif",
            ),
            hoverinfo="skip",
        )
    )
    return fig

# --------------------------------------
# App
# --------------------------------------
def main():
    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio("Choose data source", ["CSV", "MySQL"], horizontal=True, index=0)

    df = None
    if data_source == "CSV":
        csv_file = st.sidebar.file_uploader("201906.csv", type=["csv"])
        if csv_file is None:
            st.info("Upload the CSV to start.")
            st.stop()
        df = preprocess(pd.read_csv(csv_file))

    else:  # MySQL
        st.sidebar.markdown("**MySQL connection**")
        host = st.sidebar.text_input("Host / IP", value="127.0.0.1")
        port = st.sidebar.number_input("Port", min_value=1, max_value=65535, value=3306, step=1)
        user = st.sidebar.text_input("User", value="root")
        password = st.sidebar.text_input("Password", type="password")
        database = st.sidebar.text_input("Database", value="solodb")

        # ✅ 기본 쿼리를 실제 테이블명으로 세팅
        default_sql = (
            "SELECT CTY_RGN_NM, ADMI_CTY_NM, AGE_VAL, AMT, "
            "COALESCE(CNT, 1) AS CNT "
            "FROM s_data_table;"
        )
        # 필요 시 완전 수식으로 사용 (예: FROM `201906`.`s_data_table`;)
        query = st.sidebar.text_area("SQL to load data", value=default_sql, height=120)
        load_btn = st.sidebar.button("Connect & Load")

        if not load_btn:
            st.info("Fill in your MySQL info and click **Connect & Load**.")
            st.stop()

        try:
            df = preprocess(load_data_from_mysql(host, port, user, password, database, query))
            st.success("MySQL data loaded.")
        except Exception as e:
            st.error(f"MySQL load error: {e}")
            st.stop()

    # -------- Filters (구/동 Multiple 기본) --------
    st.sidebar.header("Filters and Options")

    gu_options = sorted(df["CTY_RGN_NM"].unique().tolist())
    gu_mode = st.sidebar.radio("City Region selection type",
                               ["Multiple", "Single (Dropdown)"], horizontal=True, index=0)
    if gu_mode == "Single (Dropdown)":
        selected_gus = [st.sidebar.selectbox("Select City Region", options=gu_options)]
    else:
        selected_gus = st.sidebar.multiselect("Select City Regions", options=gu_options, default=gu_options)

    base_for_dong = df[df["CTY_RGN_NM"].isin(selected_gus)] if selected_gus else df
    dong_options = sorted(base_for_dong["ADMI_CTY_NM"].unique().tolist())
    dong_mode = st.sidebar.radio("Dong selection type",
                                 ["Multiple", "Single (Dropdown)"], horizontal=True, index=0)
    if dong_mode == "Single (Dropdown)":
        selected_dongs = [st.sidebar.selectbox("Select Dong", options=dong_options)]
    else:
        selected_dongs = st.sidebar.multiselect("Select Dongs", options=dong_options, default=dong_options)

    age_options = sorted(df["AGE_VAL"].unique().tolist())
    selected_ages = st.sidebar.multiselect("Select Age Groups", options=age_options, default=age_options)

    # -------- Apply filters --------
    f = df.copy()
    if selected_gus:
        f = f[f["CTY_RGN_NM"].isin(selected_gus)]
    if selected_dongs:
        f = f[f["ADMI_CTY_NM"].isin(selected_dongs)]
    if selected_ages:
        f = f[f["AGE_VAL"].isin(selected_ages)]

    if f.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    # -------- KPI --------
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Total Amount (총액, ₩)", f"{int(f['AMT'].sum()):,}")
    with k2:
        txn_cnt = int(f["CNT"].sum()) if "CNT" in f.columns else len(f)
        st.metric("Transactions (거래건수)", f"{txn_cnt:,}")
    with k3:
        st.metric("Avg Amount per Txn (건당 평균금액, ₩)", f"{int(f['AMT'].mean()):,}")

    # -------- Map --------
    st.plotly_chart(plot_seoul_blue_thematic_map(f), use_container_width=True)

    # -------- Charts (왼쪽 City Region, 오른쪽 Age, 아래 Dong) --------
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_amt_by_city_region(f), use_container_width=True)
    with c2:
        st.plotly_chart(plot_amt_by_age(f), use_container_width=True)

    st.plotly_chart(plot_amt_by_dong(f), use_container_width=True)

if __name__ == "__main__":
    main()

# 좀 더 하면 좋을 것들
# 연령별 소비 상권 분석
# 사용자가 지역을 선택하고 하고 싶은 업종을 고르면 예상 매출액을 분석
# 혹은 타겟 소비자 분석