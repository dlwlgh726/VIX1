import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.font_manager as fm
import os

# ------------------------
# 0. 한글 폰트 설정
# ------------------------
def set_korean_font():
    font_path = "NanumGothic-Regular.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False
    else:
        print("❗ NanumGothic-Regular.ttf 아이콘을 찾을 수 없습니다.")

set_korean_font()
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

# ------------------------
# 1. 페이지 설정
# ------------------------
st.set_page_config(page_title="VIX/KOSPI + 금단율 산술문제", layout="centered")
st.title("📈 VIX 기능 + KOSPI 수익률 + 금단율-쥜가지 전략/예측")

# ------------------------
# 2. 데이터 로드
# ------------------------
@st.cache_data
def load_data():
    vix = pd.read_csv("CBOE Volatility Index 오거 데이터.csv", encoding='cp949')
    kospi = pd.read_csv("\ucf54\uc2a4\ud53c\uc9c0\uc218 \uacfc\uac70 \ub370\uc774\ud130 (1).csv", encoding='cp949')

    vix["날짜"] = pd.to_datetime(vix["날짜"])
    kospi["날짜"] = pd.to_datetime(kospi["날짜"])
    kospi["종가"] = kospi["종가"].str.replace(",", "").astype(float)

    vix = vix.set_index("날짜").sort_index()
    kospi = kospi.set_index("날짜").sort_index()
    df = kospi.join(vix, how="inner", lsuffix="_KOSPI", rsuffix="_VIX")

    df["Return"] = df["종가_KOSPI"].pct_change().shift(-1)

    bins = [0, 12, 15, 18, 21, 24, 27, 30, 100]
    labels = ["0–12", "12–15", "15–18", "18–21", "21–24", "24–27", "27–30", "30+"]
    df["VIX구간"] = pd.cut(df["종가_VIX"], bins=bins, labels=labels, right=False)

    return df

data = load_data()

# ------------------------
# 3. VIX 구간별 KOSPI 다음날 수익률 평균 배점
# ------------------------
vix_grouped = data.groupby("VIX구간")["Return"].mean().reset_index()
vix_grouped.columns = ["VIX 구간", "평균 수익률 (다음날)"]

st.subheader("📊 VIX 구간별 다음날 KOSPI 평균 수익률")
st.dataframe(vix_grouped.style.format({"평균 수익률 (다음날)": "{:.4%}"}))

fig, ax = plt.subplots()
plt.bar(vix_grouped["VIX 구간"], vix_grouped["평균 수익률 (다음날)"], color='skyblue')
plt.axhline(0, color='gray', linestyle='--')
plt.title("VIX 구간별 다음날 KOSPI 평균 수익률")
plt.ylabel("평균 수익률")
plt.xlabel("VIX 구간")
st.pyplot(fig)

# ------------------------
# 4. 기준금리 기반 아파트 가격 예측 파트 추가
# ------------------------
st.markdown("---")
st.header("🏠 기준금리 기반 아파트 평균가격 예측기")

apt_data = pd.read_csv("월별_아파트_기준금리_통합.csv")
apt_data["날짜"] = pd.to_datetime(apt_data["날짜"])
apt_data = apt_data.dropna(subset=["기준금리", "평균가격"])
apt_data["년월"] = apt_data["날짜"].dt.strftime("%Y년 %m월")

# 사용자 입력
st.sidebar.header("📌 사용자 설정")
regions = sorted(apt_data["지역"].unique())
selected_region = st.sidebar.selectbox("📍 지역 선택", regions)
ym_options = sorted(apt_data["년월"].unique())
def ym_to_date(ym_str):
    return pd.to_datetime(ym_str.replace("년 ", "-").replace("월", "-01"))
start_ym, end_ym = st.sidebar.select_slider("📅 분석 기간 설정 (연월)", options=ym_options, value=(ym_options[0], ym_options[-1]))
start_date = ym_to_date(start_ym)
end_date = ym_to_date(end_ym)
input_rate = st.sidebar.slider("📉 기준금리 입력 (%)", 0.0, 10.0, 3.5, step=0.1)
lag_months = st.sidebar.slider("⏱ 시차 (개월)", min_value=0, max_value=12, value=3)

# 시차 적용 및 필터링
apt_data = apt_data.sort_values(by=["지역", "날짜"])
apt_data["기준금리_시차"] = apt_data.groupby("지역")["기준금리"].shift(lag_months)
region_data = apt_data[(apt_data["지역"] == selected_region) & (apt_data["날짜"] >= start_date) & (apt_data["날짜"] <= end_date)]
region_data = region_data.dropna(subset=["기준금리_시차", "평균가격"])

if not region_data.empty and len(region_data) >= 3:
    region_data = region_data.copy()
    X = region_data[["기준금리_시차"]]
    y = region_data["평균가격"]
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, y)
    predicted_price = poly_model.predict(np.array([[input_rate]])).flatten()[0]

    corr = region_data["기준금리_시차"].corr(region_data["평균가격"])
    st.subheader(f"🔍 {selected_region} 지역 기준금리 {input_rate:.1f}%에 대한 예측")
    st.metric("📊 예상 평균 아파트 가격", f"{predicted_price:,.0f} 백만원")
    st.write(f"📈 기준금리(시차 {lag_months}개월)와 아파트 평균가격 간 상관계수: **{corr:.3f}**")
    st.caption(f"※ 선택된 기간: {start_ym} ~ {end_ym}, 총 {len(region_data)}개월")

    fig, ax = plt.subplots()
    sns.scatterplot(data=region_data, x="기준금리_시차", y="평균가격", ax=ax, s=40)
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred_curve = poly_model.predict(x_range)
    ax.plot(x_range, y_pred_curve, color='red', label="회귀 곡선")
    ax.scatter(input_rate, predicted_price, color="blue", s=100, label="예측값")
    ax.set_title(f"[ {selected_region} ] 기준금리(시차 {lag_months}개월)와 아파트 평균가격 관계 (비선형 회귀)")
    ax.set_xlabel(f"기준금리 (시차 {lag_months}개월)")
    ax.set_ylabel("평균 아파트 가격 (백만원)")
    ax.legend()
    st.pyplot(fig)

    fig2, ax1 = plt.subplots(figsize=(8, 4))
    color1 = "tab:blue"
    ax1.set_xlabel("날짜")
    ax1.set_ylabel("평균 아파트 가격", color=color1)
    ax1.plot(region_data["날짜"], region_data["평균가격"], marker='o', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("기준금리 (시차 적용)", color=color2)
    ax2.plot(region_data["날짜"], region_data["기준금리_시차"], marker='s', linestyle='--', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title(f"[ {selected_region} ] 월별 평균 아파트 가격 및 기준금리(시차 {lag_months}개월) 추이")
    fig2.tight_layout()
    st.pyplot(fig2)
else:
    st.warning("해당 지역의 데이터가 부족하거나 선택한 기간 내 정보가 충분하지 않습니다.")
