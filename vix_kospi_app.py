import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="VIX 기반 KOSPI 전략 분석", layout="wide")
st.title("📈 주식시장의 변동성을 활용한 VIX 기반 투자 전략 분석")

# 1. GitHub 원본 CSV 주소 입력
st.subheader("🔗 GitHub에서 CSV 데이터 불러오기")
vix_url = st.text_input("VIX 파일 주소", "https://raw.githubusercontent.com/사용자명/저장소명/main/CBOE%20Volatility%20Index%20과거%20데이터.csv")
kospi_url = st.text_input("KOSPI 파일 주소", "https://raw.githubusercontent.com/사용자명/저장소명/main/코스피지수%20과거%20데이터%20(1).csv")

# 2. 전략 조건 설정
threshold = st.slider("💡 KOSPI 보유 조건: VIX가 이 값보다 낮을 때 매수", 10, 30, 15)

if vix_url and kospi_url:
    try:
        # 데이터 불러오기
        vix = pd.read_csv(vix_url, encoding="cp949")
        kospi = pd.read_csv(kospi_url, encoding="cp949")

        # 날짜 처리
        vix["날짜"] = pd.to_datetime(vix["날짜"])
        kospi["날짜"] = pd.to_datetime(kospi["날짜"])
        kospi["종가"] = kospi["종가"].str.replace(",", "").astype(float)

        # 병합
        vix = vix.set_index("날짜").sort_index()
        kospi = kospi.set_index("날짜").sort_index()
        df = kospi.join(vix, how="inner", lsuffix="_KOSPI", rsuffix="_VIX")

        # 전략 적용
        df["Signal"] = (df["종가_VIX"] < threshold).astype(int).shift(1)
        df["Return"] = df["종가_KOSPI"].pct_change()
        df["Strategy"] = df["Signal"] * df["Return"]
        df["누적수익률_보유"] = (1 + df["Return"]).cumprod()
        df["누적수익률_VIX전략"] = (1 + df["Strategy"]).cumprod()

        # 결과 출력
        st.subheader("📊 누적 수익률 비교")
        st.line_chart(df[["누적수익률_보유", "누적수익률_VIX전략"]])

        final = df[["누적수익률_보유", "누적수익률_VIX전략"]].dropna().iloc[-1]
        st.success(f"✅ 단순 보유 전략 수익률: **{final[0]:.2f}배**")
        st.success(f"✅ VIX 전략 수익률: **{final[1]:.2f}배**")

        with st.expander("📄 데이터 테이블 보기"):
            st.dataframe(df[["종가_KOSPI", "종가_VIX", "Signal", "Return", "Strategy"]].dropna())

    except Exception as e:
        st.error(f"❌ 데이터 로딩 또는 분석 중 오류 발생: {e}")
