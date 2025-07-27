import streamlit as st
import yfinance as yf
import ta
import pandas as pd


# Şifre koruması (tüm sayfalarda ortak)
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    with st.form("giris_formu"):
        password = st.text_input("🔐 Erişim Şifresi", type="password")
        submitted = st.form_submit_button("Giriş Yap")
        if submitted:
            if password == "gizli123":  # ŞİFREN BURADA
                st.session_state.authenticated = True
                st.experimental_rerun()
            else:
                st.error("❌ Hatalı şifre")
    st.stop()

st.set_page_config(layout="wide")
st.title("📡 Otomatik Hisse Taraması")

bist30_list = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "JPM",
    "V",
    "MA",
    "UNH",
    "HD",
    "PEP",
    "KO",
    "PG",
    "LLY",
    "XOM",
    "CVX",
    "MRK",
    "ABBV",
    "JNJ",
    "DIS",
]

st.markdown("🔍 Son 1 Aylık Veriye Göre RSI ve EMA'ya Göre Tarama Yapılıyor...")

sinyal_listesi = []
progress = st.progress(0)

for i, hisse in enumerate(bist30_list):
    try:
        df = yf.download(hisse, period="1mo", interval="1d")
        close = df["Close"].dropna()

        if len(close) < 20:
            continue

        rsi = ta.momentum.RSIIndicator(close=close).rsi().iloc[-1]
        ema20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator().iloc[-1]
        ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator().iloc[-1]

        score = 0
        if rsi < 30:
            score += 1
        if ema20 > ema50:
            score += 1

        if score >= 2:
            sinyal_listesi.append((hisse, rsi, ema20, ema50))
    except:
        continue
    progress.progress((i + 1) / len(bist30_list))

if sinyal_listesi:
    st.success(f"📈 AL Fırsatı Olan {len(sinyal_listesi)} Hisse Bulundu:")
    df_sonuc = pd.DataFrame(sinyal_listesi, columns=["Hisse", "RSI", "EMA20", "EMA50"])
    st.dataframe(df_sonuc)
else:
    st.info("Bugün güçlü AL sinyali tespit edilmedi.")
