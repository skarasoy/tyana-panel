import streamlit as st
import yfinance as yf
import ta
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA


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


st.set_page_config(page_title="Tyana Panel", layout="wide")
st.title("📊 Tyana Hisse Analiz Paneli")

# ---------------- Favori Listesi -------------------
favori_dosya = "favoriler.txt"
if not os.path.exists(favori_dosya):
    with open(favori_dosya, "w") as f:
        f.write("ASELS.IS\nTHYAO.IS\nAAPL\nGOOGL\nMSFT")

with open(favori_dosya, "r") as f:
    favoriler = [line.strip() for line in f.readlines() if line.strip()]

with st.sidebar:
    st.markdown("### ⭐ Favori Hisseler")
    secilen_favori = st.selectbox("Favorilerden Seç", favoriler)
    yeni_favori = st.text_input("Yeni Hisse Ekle (örn. TSLA veya BIST: THYAO.IS)", "")
    silinecek_favori = st.selectbox("Favorilerden Sil", ["Seçiniz"] + favoriler)

    if yeni_favori:
        if yeni_favori not in favoriler:
            with open(favori_dosya, "a") as f:
                f.write(f"{yeni_favori}\n")
            favoriler.append(yeni_favori)
            st.success(f"{yeni_favori} favorilere eklendi.")
        else:
            st.info("Bu hisse zaten favorilerde.")

    if silinecek_favori != "Seçiniz":
        favoriler.remove(silinecek_favori)
        with open(favori_dosya, "w") as f:
            for fav in favoriler:
                f.write(f"{fav}\n")
        st.success(f"{silinecek_favori} favorilerden çıkarıldı.")
    st.markdown("---")

# ---------------- Hisse ve zaman aralığı seçimi -------------------
col1, col2 = st.columns(2)

with col1:
    hisse = st.text_input("Hisse Kodu (ASELS.IS, AAPL vs.)", secilen_favori)

with col2:
    zaman_araligi_label = st.selectbox(
        "Zaman Aralığı",
        ["1 Gün (24 Saat)", "1 Hafta", "1 Ay", "3 Ay", "6 Ay"],
    )
    zaman_araligi_map = {
        "1 Gün (24 Saat)": "1d",
        "1 Hafta": "5d",
        "1 Ay": "1mo",
        "3 Ay": "3mo",
        "6 Ay": "6mo",
    }
    zaman_araligi = zaman_araligi_map[zaman_araligi_label]

# ---------------- Veri çek -------------------
data = yf.download(hisse, period=zaman_araligi, interval="1d")

if data.empty:
    st.error("Veri alınamadı. Lütfen geçerli bir hisse kodu girin.")
    st.stop()

close = data["Close"]
high = data["High"]
low = data["Low"]
volume = data["Volume"] if "Volume" in data else pd.Series([None] * len(close))
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]
elif not isinstance(close, pd.Series):
    st.error("Kapanış verisi uygun formatta değil.")
    st.stop()

# ---------------- Şirket Bilgisi -------------------

try:
    info = yf.Ticker(hisse).info
    long_name = info.get("longName", "Şirket Bilgisi Yok")
    fiyat = info.get("regularMarketPrice", None)
    para_birimi = info.get("currency", "")
    logo_url = info.get("logo_url", None)
except:
    long_name = "Şirket Bilgisi Alınamadı"
    fiyat = None
    para_birimi = ""
    logo_url = None

st.markdown("---")
left, right = st.columns([0.8, 0.2])

with left:
    fiyat_goster = f" - {fiyat:.2f} {para_birimi}" if fiyat else ""
    st.subheader(f"🏢 {long_name}{fiyat_goster}")
with right:
    if logo_url:
        st.image(logo_url, width=60)

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

st.markdown("---")
st.subheader("🔮 5 Günlük Fiyat Tahminleri")
tab_arima, tab_xgb = st.tabs(["ARIMA", "XGBoost"])

future_days = 5
latest_price = close.iloc[-1]
data_np = close.values.reshape(-1, 1)

# ARIMA
with tab_arima:
    try:
        close_idx = pd.Series(
            close.values,
            index=pd.date_range(end=pd.Timestamp.today(), periods=len(close)),
        )
        arima_model = ARIMA(close_idx, order=(5, 1, 0)).fit()
        arima_forecast = arima_model.forecast(steps=future_days)
        st.metric(
            "Tahmin",
            f"{arima_forecast.iloc[-1]:.2f}",
            delta=f"{arima_forecast.iloc[-1] - latest_price:.2f}",
        )
    except Exception as e:
        st.warning(f"ARIMA modeli başarısız oldu: {e}")

# XGBoost
with tab_xgb:
    try:
        df_xgb = pd.DataFrame({"Close": close})
        df_xgb["Target"] = df_xgb["Close"].shift(-future_days)
        df_xgb = df_xgb.dropna()

        X = df_xgb[["Close"]].values
        y = df_xgb["Target"].values

        if len(X) < 10:
            raise ValueError("XGBoost için yeterli veri yok.")

        model_xgb = XGBRegressor(n_estimators=100)
        model_xgb.fit(X, y)

        xgb_input = np.array([[latest_price]])
        xgb_pred = model_xgb.predict(xgb_input)[0]
        st.metric("Tahmin", f"{xgb_pred:.2f}", delta=f"{xgb_pred - latest_price:.2f}")
    except Exception as e:
        st.warning(f"XGBoost modeli başarısız oldu: {e}")


# ---------------- Teknik Analiz -------------------
indicators = {}
signal_score = 0
signal_total = 0

# RSI
if len(close) >= 15:
    rsi = ta.momentum.RSIIndicator(close=close).rsi()
    latest_rsi = rsi.iloc[-1]
    indicators["RSI"] = latest_rsi
    signal_total += 1
    if latest_rsi < 30:
        signal_score += 1
    elif latest_rsi > 70:
        signal_score -= 1
else:
    rsi = pd.Series()
    latest_rsi = None

# MACD
macd_calc = ta.trend.MACD(close=close)
macd = macd_calc.macd()
macd_signal = macd_calc.macd_signal()
if pd.notna(macd.iloc[-1]) and pd.notna(macd_signal.iloc[-1]):
    indicators["MACD"] = macd.iloc[-1]
    indicators["MACD_Signal"] = macd_signal.iloc[-1]
    signal_total += 1
    signal_score += 1 if macd.iloc[-1] > macd_signal.iloc[-1] else -1

# EMA
ema20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
if pd.notna(ema20.iloc[-1]) and pd.notna(ema50.iloc[-1]):
    indicators["EMA20"] = ema20.iloc[-1]
    indicators["EMA50"] = ema50.iloc[-1]
    signal_total += 1
    signal_score += 1 if ema20.iloc[-1] > ema50.iloc[-1] else -1

# Bollinger Bands
bb_calc = ta.volatility.BollingerBands(close=close)
bb_upper = bb_calc.bollinger_hband()
bb_lower = bb_calc.bollinger_lband()
latest_price = close.iloc[-1]
if pd.notna(bb_upper.iloc[-1]) and pd.notna(bb_lower.iloc[-1]):
    signal_total += 1
    if latest_price < bb_lower.iloc[-1]:
        signal_score += 1
        indicators["Bollinger"] = "Alt Band → AL"
    elif latest_price > bb_upper.iloc[-1]:
        signal_score -= 1
        indicators["Bollinger"] = "Üst Band → SAT"
    else:
        indicators["Bollinger"] = "Orta Bölge → Nötr"

# ADX
try:
    adx_calc = ta.trend.ADXIndicator(high=high, low=low, close=close)
    adx = adx_calc.adx()
    if len(adx.dropna()) > 0:
        latest_adx = adx.iloc[-1]
        indicators["ADX"] = latest_adx
except Exception:
    indicators["ADX"] = None

# OBV
try:
    obv = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=volume
    ).on_balance_volume()
    indicators["OBV"] = obv.iloc[-1]
except:
    indicators["OBV"] = None

# CCI
try:
    cci = ta.trend.CCIIndicator(high=high, low=low, close=close).cci()
    indicators["CCI"] = cci.iloc[-1]
except:
    indicators["CCI"] = None

# Stochastic RSI
try:
    stoch_rsi = ta.momentum.StochRSIIndicator(close=close).stochrsi()
    indicators["StochRSI"] = stoch_rsi.iloc[-1]
except:
    indicators["StochRSI"] = None

# ---------------- Genel Sinyal -------------------

st.markdown("## 🔊 Genel Sinyal")
if signal_total == 0:
    st.warning("Yeterli teknik analiz verisi yok.")
elif signal_score >= 2:
    st.success(f"💹 AL Sinyali (%{int((signal_score/signal_total)*100)} güven)")
elif signal_score <= -1:
    st.error(f"📉 SAT Sinyali (%{abs(int((signal_score/signal_total)*100))} risk)")
else:
    st.info("🔍 BEKLE → Kararsız / Nötr Bölge")

# ---------------- Ayrıntılı Göstergeler -------------------
st.markdown("### 🔍 Ayrıntılı Göstergeler")

if latest_rsi is not None:
    if latest_rsi < 30:
        st.success(f"📈 RSI: {latest_rsi:.2f} → Aşırı Satış → AL sinyali ✅")
    elif latest_rsi > 70:
        st.error(f"📈 RSI: {latest_rsi:.2f} → Aşırı Alım → SAT sinyali ❌")
    else:
        st.info(f"📈 RSI: {latest_rsi:.2f} → Nötr → BEKLE 🔄")

if "MACD" in indicators:
    macd_val = indicators["MACD"]
    macd_sig = indicators["MACD_Signal"]
    if macd_val > macd_sig:
        st.success(
            f"📉 MACD: {macd_val:.2f} > Signal: {macd_sig:.2f} → Yukarı Kesişim → AL sinyali ✅"
        )
    else:
        st.error(
            f"📉 MACD: {macd_val:.2f} < Signal: {macd_sig:.2f} → Aşağı Kesişim → SAT sinyali ❌"
        )

if "EMA20" in indicators:
    e20 = indicators["EMA20"]
    e50 = indicators["EMA50"]
    if e20 > e50:
        st.success(
            f"📊 EMA20: {e20:.2f} > EMA50: {e50:.2f} → Trend Yukarı → AL sinyali ✅"
        )
    else:
        st.error(
            f"📊 EMA20: {e20:.2f} < EMA50: {e50:.2f} → Trend Aşağı → SAT sinyali ❌"
        )

if "Bollinger" in indicators:
    bb_info = indicators["Bollinger"]
    if "AL" in bb_info:
        st.success(f"🎯 Bollinger Bands: {bb_info} ✅")
    elif "SAT" in bb_info:
        st.error(f"🎯 Bollinger Bands: {bb_info} ❌")
    else:
        st.info(f"🎯 Bollinger Bands: {bb_info} 🔄")

if "ADX" in indicators and indicators["ADX"]:
    st.info(
        f"📶 ADX (Trend Gücü): {indicators['ADX']:.2f} → {'Zayıf Trend' if indicators['ADX'] < 20 else 'Güçlü Trend'}"
    )

if "OBV" in indicators and indicators["OBV"] is not None:
    st.info(f"📊 OBV: {indicators['OBV']:.2f} → Hacme göre fiyat yönü")

if "CCI" in indicators and indicators["CCI"] is not None:
    st.info(f"📉 CCI: {indicators['CCI']:.2f} → Ortalama sapma")

if "StochRSI" in indicators and indicators["StochRSI"] is not None:
    st.info(f"⚡ Stochastic RSI: {indicators['StochRSI']:.2f}")

# ---------------- Grafikler Sekmeli -------------------
st.markdown("---")
st.subheader("📈 Grafikler")

tab1, tab2, tab3, tab4 = st.tabs(
    ["💰 Kapanış Fiyatı", "📊 RSI", "📊 MACD", "📊 Teknik Sinyal Skoru"]
)

with tab1:
    st.line_chart(pd.DataFrame({"Kapanış": close}))

with tab2:
    if not rsi.empty:
        st.line_chart(pd.DataFrame({"RSI": rsi}))
    else:
        st.info("RSI göstermek için yeterli veri yok.")

with tab3:
    st.line_chart(pd.DataFrame({"MACD": macd, "Signal": macd_signal}))

with tab4:
    scores = {
        "RSI_Score": (
            1
            if latest_rsi and latest_rsi < 30
            else (-1 if latest_rsi and latest_rsi > 70 else 0)
        ),
        "MACD_Score": 1 if macd.iloc[-1] > macd_signal.iloc[-1] else -1,
        "EMA_Score": 1 if ema20.iloc[-1] > ema50.iloc[-1] else -1,
        "Bollinger_Score": (
            1
            if latest_price < bb_lower.iloc[-1]
            else (-1 if latest_price > bb_upper.iloc[-1] else 0)
        ),
    }
    st.bar_chart(pd.DataFrame(scores, index=["Bugün"]).T)
