import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# ✅ 2. 전략 파라미터 입력
threshold = st.slider("💡 VIX가 이 값보다 낮을 때 KOSPI를 보유", 10, 30, 15)

# ✅ 3. 데이터 불러오기 및 전략 계산
try:
    df = load_data()

    # 전략 신호 및 수익률 계산
    df["Signal"] = (df["종가_VIX"] < threshold).astype(int).shift(1)
    df["Return"] = df["종가_KOSPI"].pct_change()
    df["Strategy"] = df["Signal"] * df["Return"]
    df["누적수익률_보유"] = (1 + df["Return"]).cumprod()
    df["누적수익률_전략"] = (1 + df["Strategy"]).cumprod()

    # ✅ 4. 전략 수익률 시각화
    st.subheader("📊 누적 수익률 비교")
    st.line_chart(df[["누적수익률_보유", "누적수익률_전략"]])

    final = df[["누적수익률_보유", "누적수익률_전략"]].dropna().iloc[-1]
    st.success(f"✅ 단순 보유 수익률: **{final[0]:.2f}배**")
    st.success(f"✅ VIX 전략 수익률: **{final[1]:.2f}배**")

    with st.expander("📄 전략 데이터 보기"):
        st.dataframe(df[["종가_KOSPI", "종가_VIX", "Signal", "Return", "Strategy"]].dropna())

    # ✅ 5. VIX 구간별 다음날 수익률 분석
    st.subheader("📈 VIX 구간별 다음날 KOSPI 수익률 분석")

    df["다음날수익률"] = df["종가_KOSPI"].pct_change().shift(-1)

    bins = [0, 12, 15, 18, 21, 24, 27, 30, 100]
    labels = ["0–12", "12–15", "15–18", "18–21", "21–24", "24–27", "27–30", "30+"]
    df["VIX구간"] = pd.cut(df["종가_VIX"], bins=bins, labels=labels, right=False)

    vix_grouped = df.groupby("VIX구간")["다음날수익률"].mean().reset_index()
    vix_grouped.columns = ["VIX 구간", "평균 수익률 (다음날)"]

    st.dataframe(vix_grouped.style.format({"평균 수익률 (다음날)": "{:.4%}"}))

    # ✅ 6. 바 차트 시각화
    fig, ax = plt.subplots()
    plt.bar(vix_grouped["VIX 구간"], vix_grouped["평균 수익률 (다음날)"], color='skyblue')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("VIX 구간별 다음날 KOSPI 평균 수익률")
    plt.ylabel("평균 수익률")
    plt.xlabel("VIX 구간")
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ 분석 중 오류 발생: {e}")
