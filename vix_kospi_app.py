import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="VIX 기반 KOSPI 전략 분석", layout="wide")
st.title("📈 주식시장의 변동성을 활용한 VIX 기반 KOSPI 투자 전략 분석")

# ✅ 1. 파일 불러오기 함수
@st.cache_data
def load_data():
    vix = pd.read_csv("CBOE Volatility Index 과거 데이터.csv", encoding="cp949")
    kospi = pd.read_csv("코스피지수 과거 데이터 (1).csv", encoding="cp949")

    vix["날짜"] = pd.to_datetime(vix["날짜"])
    kospi["날짜"] = pd.to_datetime(kospi["날짜"])
    kospi["종가"] = kospi["종가"].str.replace(",", "").astype(float)

    vix = vix.set_index("날짜").sort_index()
    kospi = kospi.set_index("날짜").sort_index()

    df = kospi.join(vix, how="inner", lsuffix="_KOSPI", rsuffix="_VIX")
    return df

# ✅ 2. 전략 파라미터
threshold = st.slider("💡 VIX가 이 값보다 낮을 때 KOSPI를 보유", 10, 30, 15)

# ✅ 3. 데이터 불러오기 및 전략 계산
try:
    df = load_data()

    df["Signal"] = (df["종가_VIX"] < threshold).astype(int).shift(1)
    df["Return"] = df["종가_KOSPI"].pct_change()
    df["Strategy"] = df["Signal"] * df["Return"]
    df["누적수익률_보유"] = (1 + df["Return"]).cumprod()
    df["누적수익률_전략"] = (1 + df["Strategy"]).cumprod()

    # ✅ 4. 시각화
    st.subheader("📊 누적 수익률 비교")
    st.line_chart(df[["누적수익률_보유", "누적수익률_전략"]])

    final = df[["누적수익률_보유", "누적수익률_전략"]].dropna().iloc[-1]
    st.success(f"✅ 단순 보유 수익률: **{final[0]:.2f}배**")
    st.success(f"✅ VIX 전략 수익률: **{final[1]:.2f}배**")

    with st.expander("📄 분석 데이터 보기"):
        st.dataframe(df[["종가_KOSPI", "종가_VIX", "Signal", "Return", "Strategy"]].dropna())

except Exception as e:
    st.error(f"❌ 분석 중 오류 발생: {e}")
