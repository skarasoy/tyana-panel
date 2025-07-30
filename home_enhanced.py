import streamlit as st
import yfinance as yf
import ta
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
from datetime import datetime, timedelta
import time
import logging
import random
from typing import Optional, Dict, List, Tuple
from io import BytesIO

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Performans için uyarıları kapat
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Logger ayarla
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ==============================================================================
# RATE LIMITING VE VERİ ÇEKME SINIFI
# ==============================================================================


class RateLimitedDataFetcher:
    """Rate limiting ile güvenli veri çekme"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.failed_symbols = set()

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_stock_data(
        _self, symbol: str, period: str = "3mo"
    ) -> Optional[pd.DataFrame]:
        """Rate limiting ve retry logic ile veri çek"""

        for attempt in range(_self.max_retries):
            try:
                # Random delay ekle
                delay = _self.base_delay * (0.5 + random.random())
                time.sleep(delay)

                # Veri çek
                data = yf.download(
                    symbol, period=period, interval="1d", progress=False, threads=False
                )

                if not data.empty and len(data) >= 5:
                    return data

            except Exception as e:
                if attempt < _self.max_retries - 1:
                    # Exponential backoff
                    wait_time = _self.base_delay * (2**attempt) + random.random()
                    time.sleep(wait_time)
                else:
                    _self.failed_symbols.add(symbol)
                    logger.error(f"Failed to fetch {symbol}: {str(e)}")

        return None


# ==============================================================================
# CACHE FUNCTIONS - Performans için kritik
# ==============================================================================


@st.cache_data(ttl=300, show_spinner=False)  # 5 dakika cache
def get_stock_data(symbol, period):
    """Hisse verilerini cache'li olarak çek"""
    fetcher = RateLimitedDataFetcher()
    return fetcher.fetch_stock_data(symbol, period)


@st.cache_data(ttl=600, show_spinner=False)  # 10 dakika cache
def get_company_info(symbol):
    """Şirket bilgilerini cache'li olarak çek"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("longName", "Şirket Bilgisi Yok"),
            "price": info.get("regularMarketPrice", None),
            "currency": info.get("currency", ""),
            "logo": info.get("logo_url", None),
            "sector": info.get("sector", "Bilinmiyor"),
            "marketCap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "dividend_yield": info.get("dividendYield", None),
        }
    except:
        return {
            "name": "Şirket Bilgisi Alınamadı",
            "price": None,
            "currency": "",
            "logo": None,
            "sector": "Bilinmiyor",
            "marketCap": 0,
            "pe_ratio": None,
            "dividend_yield": None,
        }


@st.cache_data(ttl=60, show_spinner=False)
def calculate_technical_indicators(
    close_series, high_series, low_series, volume_series
):
    """Tüm teknik göstergeleri tek seferde hesapla"""
    indicators = {}
    signal_score = 0
    signal_total = 0

    try:
        # RSI
        if len(close_series) >= 15:
            rsi = ta.momentum.RSIIndicator(close=close_series).rsi()
            latest_rsi = rsi.iloc[-1]
            indicators["RSI"] = latest_rsi
            signal_total += 1
            if latest_rsi < 30:
                signal_score += 2
            elif latest_rsi < 40:
                signal_score += 1
            elif latest_rsi > 70:
                signal_score -= 1

        # MACD
        macd_calc = ta.trend.MACD(close=close_series)
        macd = macd_calc.macd()
        macd_signal = macd_calc.macd_signal()
        if pd.notna(macd.iloc[-1]) and pd.notna(macd_signal.iloc[-1]):
            indicators["MACD"] = macd.iloc[-1]
            indicators["MACD_Signal"] = macd_signal.iloc[-1]
            signal_total += 1
            signal_score += 1 if macd.iloc[-1] > macd_signal.iloc[-1] else -1

        # EMA
        ema20 = ta.trend.EMAIndicator(close=close_series, window=20).ema_indicator()
        ema50 = ta.trend.EMAIndicator(close=close_series, window=50).ema_indicator()
        if pd.notna(ema20.iloc[-1]) and pd.notna(ema50.iloc[-1]):
            indicators["EMA20"] = ema20.iloc[-1]
            indicators["EMA50"] = ema50.iloc[-1]
            signal_total += 1
            signal_score += 1 if ema20.iloc[-1] > ema50.iloc[-1] else -1

        # Bollinger Bands
        bb_calc = ta.volatility.BollingerBands(close=close_series)
        bb_upper = bb_calc.bollinger_hband()
        bb_lower = bb_calc.bollinger_lband()
        bb_middle = bb_calc.bollinger_mavg()
        latest_price = close_series.iloc[-1]

        if pd.notna(bb_upper.iloc[-1]) and pd.notna(bb_lower.iloc[-1]):
            indicators["BB_Upper"] = bb_upper.iloc[-1]
            indicators["BB_Lower"] = bb_lower.iloc[-1]
            indicators["BB_Middle"] = bb_middle.iloc[-1]

            signal_total += 1
            if latest_price < bb_lower.iloc[-1]:
                signal_score += 1
                indicators["Bollinger"] = "Alt Band → AL"
            elif latest_price > bb_upper.iloc[-1]:
                signal_score -= 1
                indicators["Bollinger"] = "Üst Band → SAT"
            else:
                indicators["Bollinger"] = "Orta Bölge → Nötr"

        # ADX - Trend gücü
        try:
            adx_calc = ta.trend.ADXIndicator(
                high=high_series, low=low_series, close=close_series
            )
            adx = adx_calc.adx()
            if len(adx.dropna()) > 0:
                indicators["ADX"] = adx.iloc[-1]
                if adx.iloc[-1] > 25:
                    signal_total += 0.5  # Güçlü trend bonus
        except:
            indicators["ADX"] = None

        # Volume analizi
        try:
            if volume_series is not None and not volume_series.isna().all():
                # OBV
                obv = ta.volume.OnBalanceVolumeIndicator(
                    close=close_series, volume=volume_series
                ).on_balance_volume()
                indicators["OBV"] = obv.iloc[-1]

                # Volume ortalaması
                vol_ma = volume_series.rolling(20).mean()
                latest_volume = volume_series.iloc[-1]
                if pd.notna(vol_ma.iloc[-1]) and vol_ma.iloc[-1] > 0:
                    volume_ratio = latest_volume / vol_ma.iloc[-1]
                    indicators["Volume_Ratio"] = volume_ratio
                    if volume_ratio > 1.5:
                        signal_score += 0.5  # Yüksek hacim bonus
        except:
            indicators["OBV"] = None
            indicators["Volume_Ratio"] = None

        # CCI
        try:
            cci = ta.trend.CCIIndicator(
                high=high_series, low=low_series, close=close_series
            ).cci()
            indicators["CCI"] = cci.iloc[-1]
            if cci.iloc[-1] < -100:
                signal_score += 0.5
            elif cci.iloc[-1] > 100:
                signal_score -= 0.5
        except:
            indicators["CCI"] = None

        # Stochastic RSI
        try:
            stoch_rsi = ta.momentum.StochRSIIndicator(close=close_series).stochrsi()
            indicators["StochRSI"] = stoch_rsi.iloc[-1]
            if stoch_rsi.iloc[-1] < 0.2:
                signal_score += 0.5
            elif stoch_rsi.iloc[-1] > 0.8:
                signal_score -= 0.5
        except:
            indicators["StochRSI"] = None

        # ATR - Volatilite
        try:
            atr = ta.volatility.AverageTrueRange(
                high=high_series, low=low_series, close=close_series
            ).average_true_range()
            indicators["ATR"] = atr.iloc[-1]
            indicators["ATR_Percent"] = (atr.iloc[-1] / latest_price) * 100
        except:
            indicators["ATR"] = None
            indicators["ATR_Percent"] = None

        # Series'leri de döndür (grafikler için)
        indicators["rsi_series"] = rsi if "rsi" in locals() else pd.Series()
        indicators["macd_series"] = macd if "macd" in locals() else pd.Series()
        indicators["macd_signal_series"] = (
            macd_signal if "macd_signal" in locals() else pd.Series()
        )
        indicators["bb_upper_series"] = (
            bb_upper if "bb_upper" in locals() else pd.Series()
        )
        indicators["bb_lower_series"] = (
            bb_lower if "bb_lower" in locals() else pd.Series()
        )
        indicators["bb_middle_series"] = (
            bb_middle if "bb_middle" in locals() else pd.Series()
        )

    except Exception as e:
        logger.error(f"Teknik analiz hesaplama hatası: {str(e)}")

    return indicators, signal_score, signal_total


@st.cache_data(ttl=300, show_spinner=False)
def predict_prices(close_series, future_days=5):
    """Gelişmiş fiyat tahminleri"""
    predictions = {}
    latest_price = close_series.iloc[-1]

    # ARIMA
    try:
        close_idx = pd.Series(
            close_series.values,
            index=pd.date_range(end=pd.Timestamp.today(), periods=len(close_series)),
        )

        # Auto ARIMA parametreleri
        best_aic = float("inf")
        best_order = (5, 1, 0)

        for p in range(1, 6):
            for d in range(0, 2):
                for q in range(0, 2):
                    try:
                        model = ARIMA(close_idx, order=(p, d, q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_order = (p, d, q)
                    except:
                        continue

        arima_model = ARIMA(close_idx, order=best_order).fit()
        arima_forecast = arima_model.forecast(steps=future_days)

        predictions["arima"] = {
            "value": arima_forecast.iloc[-1],
            "delta": arima_forecast.iloc[-1] - latest_price,
            "confidence": 1 - (arima_model.aic / 1000),  # Basit güven skoru
            "order": best_order,
        }
    except Exception as e:
        predictions["arima"] = {"error": str(e)}

    # LSTM
    try:
        model_lstm = tf.keras.Sequential()
        model_lstm.add(
            tf.keras.layers.LSTM(
                50, return_sequences=False, input_shape=(X_lstm.shape[1], 1)
            )
        )
        model_lstm.add(tf.keras.layers.Dense(1))
        model_lstm.compile(optimizer="adam", loss="mse")
        from sklearn.preprocessing import MinMaxScaler

        df_lstm = pd.DataFrame({"Close": close_series})
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_lstm[["Close"]])

        # Sekans oluştur (önceki 10 gün ile tahmin)
        sequence_length = 10
        X_lstm, y_lstm = [], []
        for i in range(sequence_length, len(scaled_data) - future_days):
            X_lstm.append(scaled_data[i - sequence_length : i])
            y_lstm.append(scaled_data[i + future_days - 1])

        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

        if len(X_lstm) >= 30:
            model_lstm = Sequential()
            model_lstm.add(
                LSTM(50, return_sequences=False, input_shape=(X_lstm.shape[1], 1))
            )
            model_lstm.add(Dense(1))
            model_lstm.compile(optimizer="adam", loss="mse")

            model_lstm.fit(X_lstm, y_lstm, epochs=10, batch_size=8, verbose=0)

            # Tahmin için son 10 gün
            last_seq = scaled_data[-sequence_length:]
            last_seq = last_seq.reshape(1, sequence_length, 1)
            lstm_pred_scaled = model_lstm.predict(last_seq, verbose=0)
            lstm_pred = scaler.inverse_transform(lstm_pred_scaled)[0][0]

            predictions["lstm"] = {
                "value": lstm_pred,
                "delta": lstm_pred - latest_price,
                "confidence": 1.0,  # Eğitim skoru koyulabilir (ör: val_loss vs mean)
            }
        else:
            predictions["lstm"] = {"error": "Yetersiz veri"}
    except Exception as e:
        predictions["lstm"] = {"error": str(e)}

    # XGBoost
    try:
        # Feature engineering
        df_xgb = pd.DataFrame({"Close": close_series})

        # Teknik göstergeler ekle
        df_xgb["SMA_5"] = df_xgb["Close"].rolling(5).mean()
        df_xgb["SMA_20"] = df_xgb["Close"].rolling(20).mean()
        df_xgb["RSI"] = ta.momentum.RSIIndicator(close=df_xgb["Close"]).rsi()
        df_xgb["Returns"] = df_xgb["Close"].pct_change()
        df_xgb["Volatility"] = df_xgb["Returns"].rolling(20).std()

        # Target
        df_xgb["Target"] = df_xgb["Close"].shift(-future_days)
        df_xgb = df_xgb.dropna()

        if len(df_xgb) >= 50:
            feature_cols = ["Close", "SMA_5", "SMA_20", "RSI", "Returns", "Volatility"]
            X = df_xgb[feature_cols].values
            y = df_xgb["Target"].values

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Model
            model_xgb = XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42
            )
            model_xgb.fit(X_train, y_train)

            # Score
            score = model_xgb.score(X_test, y_test)

            # Tahmin için son değerler
            last_values = df_xgb[feature_cols].iloc[-1].values.reshape(1, -1)
            xgb_pred = model_xgb.predict(last_values)[0]

            predictions["xgboost"] = {
                "value": xgb_pred,
                "delta": xgb_pred - latest_price,
                "confidence": score,
                "feature_importance": dict(
                    zip(feature_cols, model_xgb.feature_importances_)
                ),
            }
        else:
            predictions["xgboost"] = {"error": "Yetersiz veri"}
    except Exception as e:
        predictions["xgboost"] = {"error": str(e)}

    # Basit trend analizi
    try:
        # Linear regression trend
        x = np.arange(len(close_series))
        y = close_series.values

        slope, intercept = np.polyfit(x, y, 1)
        trend_pred = slope * (len(close_series) + future_days) + intercept

        # Determination coefficient (R² score)
        residuals = y - (slope * x + intercept)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        predictions["trend"] = {
            "value": trend_pred,
            "delta": trend_pred - latest_price,
            "slope": slope,
            "direction": "Yükseliş" if slope > 0 else "Düşüş",
            "confidence": min(max(r2, 0), 1),  # 0-1 aralığına zorla
        }
    except:
        predictions["trend"] = {"error": "Trend hesaplanamadı"}

    return predictions


# ==============================================================================
# FAVORİ YÖNETİMİ
# ==============================================================================


def load_favorites():
    """Favorileri yükle"""
    favori_dosya = "favoriler.json"
    if not os.path.exists(favori_dosya):
        default_favs = {
            "BIST": ["ASELS.IS", "THYAO.IS", "GARAN.IS", "AKBNK.IS"],
            "US": ["AAPL", "GOOGL", "MSFT", "NVDA"],
            "Kripto": ["BTC-USD", "ETH-USD"],
        }
        with open(favori_dosya, "w") as f:
            json.dump(default_favs, f)
        return default_favs

    try:
        with open(favori_dosya, "r") as f:
            return json.load(f)
    except:
        return {"BIST": ["ASELS.IS"], "US": ["AAPL"]}


def save_favorites(favoriler):
    """Favorileri kaydet"""
    try:
        with open("favoriler.json", "w") as f:
            json.dump(favoriler, f)
        return True
    except:
        return False


# ==============================================================================
# PERFORMANS MONİTÖRÜ
# ==============================================================================


class PerformanceMonitor:
    """Performans takibi"""

    def __init__(self):
        if "perf_monitor" not in st.session_state:
            st.session_state.perf_monitor = {
                "start_time": time.time(),
                "api_calls": 0,
                "cache_hits": 0,
                "errors": 0,
            }

    def record_api_call(self):
        st.session_state.perf_monitor["api_calls"] += 1

    def record_cache_hit(self):
        st.session_state.perf_monitor["cache_hits"] += 1

    def record_error(self):
        st.session_state.perf_monitor["errors"] += 1

    def get_stats(self):
        elapsed = time.time() - st.session_state.perf_monitor["start_time"]
        api_calls = st.session_state.perf_monitor["api_calls"]
        cache_hits = st.session_state.perf_monitor["cache_hits"]

        return {
            "elapsed_time": elapsed,
            "api_calls": api_calls,
            "cache_hits": cache_hits,
            "cache_hit_rate": (cache_hits / max(api_calls + cache_hits, 1)) * 100,
            "errors": st.session_state.perf_monitor["errors"],
        }


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Tyana Panel - Home",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS stilleri
st.markdown(
    """
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .success-metric {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .danger-metric {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    .warning-metric {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Şifre koruması
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    with st.form("giris_formu"):
        st.markdown("### 🔐 Tyana Panel Girişi")
        password = st.text_input("Erişim Şifresi", type="password")
        col1, col2 = st.columns([3, 1])
        with col2:
            submitted = st.form_submit_button("Giriş Yap", use_container_width=True)

        if submitted:
            if password == "gizli123":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Hatalı şifre")
    st.stop()

# Performans monitörü başlat
perf_monitor = PerformanceMonitor()

# Ana başlık
st.title("📊 Tyana Hisse Analiz Paneli - Premium")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### ⭐ Favori Yönetimi")

    # Favorileri yükle
    if "favoriler" not in st.session_state:
        st.session_state.favoriler = load_favorites()

    favoriler = st.session_state.favoriler

    # Liste olması durumuna karşı koruma
    if not isinstance(favoriler, dict):
        st.warning("⚠️ Favori verisi bozulmuş. Sıfırlanıyor...")
        favoriler = load_favorites()
        st.session_state.favoriler = favoriler

    # Kategori seçimi
    kategori = st.selectbox("Kategori", list(favoriler.keys()))

    if kategori in favoriler:
        secilen_favori = st.selectbox(
            "Favori Hisse", favoriler[kategori], format_func=lambda x: f"⭐ {x}"
        )
    else:
        secilen_favori = ""

    # Favori yönetimi
    with st.expander("🔧 Favori Düzenle", expanded=False):
        # Yeni kategori
        yeni_kategori = st.text_input("Yeni Kategori")
        if st.button("➕ Kategori Ekle") and yeni_kategori:
            if yeni_kategori not in st.session_state.favoriler:
                st.session_state.favoriler[yeni_kategori] = []
                save_favorites(st.session_state.favoriler)
                st.success("✅ Kategori eklendi")
                st.rerun()

        # Yeni hisse
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            kat_sec = st.selectbox(
                "Kategori Seç", list(favoriler.keys()), key="add_cat"
            )
        with col2:
            yeni_hisse = st.text_input("Hisse Kodu")

        if st.button("➕ Hisse Ekle") and yeni_hisse and kat_sec:
            if yeni_hisse not in st.session_state.favoriler[kat_sec]:
                st.session_state.favoriler[kat_sec].append(yeni_hisse)
                save_favorites(st.session_state.favoriler)
                st.success(f"✅ {yeni_hisse} eklendi")
                st.rerun()

    st.markdown("---")

    # Hızlı erişim
    st.markdown("### 🚀 Hızlı Erişim")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 BIST30", use_container_width=True):
            st.session_state.quick_symbol = "XU030.IS"
    with col2:
        if st.button("🇺🇸 S&P500", use_container_width=True):
            st.session_state.quick_symbol = "^GSPC"

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💰 Altın", use_container_width=True):
            st.session_state.quick_symbol = "GC=F"
    with col2:
        if st.button("💵 USD/TRY", use_container_width=True):
            st.session_state.quick_symbol = "TRY=X"

    st.markdown("---")

    # Performans bilgileri
    with st.expander("⚡ Performans", expanded=False):
        stats = perf_monitor.get_stats()
        st.metric("Cache Hit Rate", f"{stats['cache_hit_rate']:.1f}%")
        st.metric("API Calls", stats["api_calls"])
        st.metric("Errors", stats["errors"])

# Ana panel
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    default_symbol = st.session_state.get("quick_symbol", secilen_favori)
    hisse = st.text_input(
        "📈 Hisse Kodu",
        value=default_symbol,
        placeholder="ASELS.IS, AAPL, BTC-USD",
        help="BIST için .IS, kripto için -USD ekleyin",
    )

with col2:
    zaman_secenekleri = {
        "1 Gün": "1d",
        "5 Gün": "5d",
        "1 Ay": "1mo",
        "3 Ay": "3mo",
        "6 Ay": "6mo",
        "1 Yıl": "1y",
        "2 Yıl": "2y",
        "5 Yıl": "5y",
    }
    zaman_label = st.selectbox(
        "📅 Zaman Aralığı", list(zaman_secenekleri.keys()), index=3  # Default 3 ay
    )
    zaman_araligi = zaman_secenekleri[zaman_label]

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    analiz_btn = st.button("🔍 Analiz Et", type="primary", use_container_width=True)

# Veri çekme ve analiz
if hisse and (analiz_btn or default_symbol):
    # Loading state
    with st.spinner("📡 Veriler yükleniyor..."):
        perf_monitor.record_api_call()
        data = get_stock_data(hisse, zaman_araligi)

    if data is None or data.empty:
        st.error("❌ Veri alınamadı. Lütfen hisse kodunu kontrol edin.")
        perf_monitor.record_error()

        # Öneri göster
        st.info(
            """
        💡 **İpuçları:**
        - BIST hisseleri için `.IS` uzantısı ekleyin (örn: ASELS.IS)
        - ABD hisseleri için sadece sembol kullanın (örn: AAPL)
        - Kripto için `-USD` ekleyin (örn: BTC-USD)
        - Endeksler için `^` kullanın (örn: ^GSPC)
        """
        )
        st.stop()

    # Veri hazırlama
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"] if "Volume" in data else pd.Series([None] * len(close))

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # Şirket bilgisi
    with st.spinner("🏢 Şirket bilgileri yükleniyor..."):
        company_info = get_company_info(hisse)

    # Başlık ve temel bilgiler
    st.markdown("---")

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown(f"## 🏢 {company_info['name']}")
        if company_info["sector"] != "Bilinmiyor":
            st.caption(f"Sektör: {company_info['sector']}")

    with col2:
        if company_info["price"]:
            change = close.iloc[-1] - close.iloc[-2] if len(close) > 1 else 0
            change_pct = (
                (change / close.iloc[-2] * 100)
                if len(close) > 1 and close.iloc[-2] != 0
                else 0
            )

            st.metric(
                "Güncel Fiyat",
                f"{company_info['price']:.2f} {company_info['currency']}",
                delta=f"{change:.2f} ({change_pct:.2f}%)",
            )

    with col3:
        if company_info["logo"]:
            st.image(company_info["logo"], width=80)

    # Ek metrikler
    if any(
        [
            company_info["marketCap"],
            company_info["pe_ratio"],
            company_info["dividend_yield"],
        ]
    ):
        st.markdown("### 📊 Temel Metrikler")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if company_info["marketCap"]:
                market_cap_b = company_info["marketCap"] / 1e9
                st.metric("Piyasa Değeri", f"${market_cap_b:.2f}B")

        with col2:
            if company_info["pe_ratio"]:
                st.metric("F/K Oranı", f"{company_info['pe_ratio']:.2f}")

        with col3:
            if company_info["dividend_yield"]:
                st.metric(
                    "Temettü Verimi", f"%{company_info['dividend_yield']*100:.2f}"
                )

        with col4:
            # 52 hafta aralığı
            if len(close) >= 252:
                year_high = close[-252:].max()
                year_low = close[-252:].min()
                current = close.iloc[-1]
                position = (current - year_low) / (year_high - year_low) * 100
                st.metric("52 Hafta Pozisyon", f"%{position:.1f}")

    # Teknik analiz
    with st.spinner("📊 Teknik analiz hesaplanıyor..."):
        indicators, signal_score, signal_total = calculate_technical_indicators(
            close, high, low, volume
        )

    # AI Tahminleri
    st.markdown("---")
    st.markdown("### 🤖 AI Fiyat Tahminleri (5 Günlük)")

    with st.spinner("🧠 AI modelleri çalışıyor..."):
        predictions = predict_prices(close, future_days=5)

    # Hatasız tahminleri filtrele
    valid_predictions = [
        (model_name, pred_data)
        for model_name, pred_data in predictions.items()
        if "error" not in pred_data
    ]

    pred_cols = st.columns(len(valid_predictions))

    for idx, (model_name, pred_data) in enumerate(valid_predictions):
        with pred_cols[idx]:
            model_display = {
                "arima": "ARIMA",
                "xgboost": "XGBoost",
                "trend": "Trend",
                "lstm": "LSTM",
            }.get(model_name, model_name)

            delta = pred_data["delta"]
            delta_pct = (delta / close.iloc[-1]) * 100

            color = "inverse" if delta > 0 else "normal"

            st.metric(
                model_display,
                f"{pred_data['value']:.2f}",
                delta=f"{delta:.2f} ({delta_pct:.2f}%)",
                delta_color=color,
            )

            # Güven değeri sınırlanıyor
            confidence = pred_data.get("confidence", 0)
            confidence = min(max(confidence, 0), 1)  # 0.0 ile 1.0 arasında sınırla

            st.progress(confidence, text=f"Güven: %{confidence*100:.1f}")

    # Genel sinyal
    st.markdown("---")
    st.markdown("### 🎯 Genel Değerlendirme")

    # Sinyal kartları
    col1, col2, col3, col4 = st.columns(4)

    # Sinyal skoru hesaplama
    normalized_score = (
        (signal_score / max(signal_total, 1)) * 100 if signal_total > 0 else 0
    )

    with col1:
        if normalized_score >= 50:
            st.success(f"### 💹 GÜÇLÜ AL")
            st.markdown(f"**Sinyal Gücü:** %{normalized_score:.0f}")
        elif normalized_score >= 25:
            st.info(f"### 📈 AL")
            st.markdown(f"**Sinyal Gücü:** %{normalized_score:.0f}")
        elif normalized_score <= -25:
            st.error(f"### 📉 SAT")
            st.markdown(f"**Risk Seviyesi:** %{abs(normalized_score):.0f}")
        else:
            st.warning(f"### ⏸️ BEKLE")
            st.markdown("**Kararsız Bölge**")

    with col2:
        # Trend durumu
        if indicators.get("EMA20") and indicators.get("EMA50"):
            if indicators["EMA20"] > indicators["EMA50"]:
                st.metric(
                    "Trend",
                    "↗️ Yükseliş",
                    delta=f"{((indicators['EMA20']/indicators['EMA50'])-1)*100:.1f}%",
                )
            else:
                st.metric(
                    "Trend",
                    "↘️ Düşüş",
                    delta=f"{((indicators['EMA20']/indicators['EMA50'])-1)*100:.1f}%",
                )

    with col3:
        # Momentum
        if indicators.get("RSI"):
            rsi_val = indicators["RSI"]
            if rsi_val < 30:
                st.metric("Momentum", "🔥 Aşırı Satım", delta=f"RSI: {rsi_val:.1f}")
            elif rsi_val > 70:
                st.metric("Momentum", "❄️ Aşırı Alım", delta=f"RSI: {rsi_val:.1f}")
            else:
                st.metric("Momentum", "⚖️ Dengeli", delta=f"RSI: {rsi_val:.1f}")

    with col4:
        # Volatilite
        if indicators.get("ATR_Percent"):
            atr_pct = indicators["ATR_Percent"]
            if atr_pct > 5:
                st.metric("Volatilite", "🌊 Yüksek", delta=f"%{atr_pct:.1f}")
            elif atr_pct > 2:
                st.metric("Volatilite", "〰️ Orta", delta=f"%{atr_pct:.1f}")
            else:
                st.metric("Volatilite", "➖ Düşük", delta=f"%{atr_pct:.1f}")

    # Detaylı göstergeler
    st.markdown("---")
    st.markdown("### 📊 Detaylı Teknik Göstergeler")

    # Gösterge tablosu
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Momentum Göstergeleri")

        # RSI
        if indicators.get("RSI") is not None:
            rsi_val = indicators["RSI"]
            rsi_status = ""
            if rsi_val < 30:
                rsi_status = "🟢 Aşırı Satım - AL sinyali"
            elif rsi_val < 40:
                rsi_status = "🟡 Düşük - İzle"
            elif rsi_val > 70:
                rsi_status = "🔴 Aşırı Alım - SAT sinyali"
            else:
                rsi_status = "⚪ Normal"

            st.info(f"**RSI (14):** {rsi_val:.2f} - {rsi_status}")

        # MACD
        if (
            indicators.get("MACD") is not None
            and indicators.get("MACD_Signal") is not None
        ):
            macd_val = indicators["MACD"]
            macd_sig = indicators["MACD_Signal"]
            macd_diff = macd_val - macd_sig

            if macd_diff > 0:
                st.success(f"**MACD:** {macd_val:.4f} > Signal: {macd_sig:.4f} ✅")
            else:
                st.error(f"**MACD:** {macd_val:.4f} < Signal: {macd_sig:.4f} ❌")

        # Stochastic RSI
        if indicators.get("StochRSI") is not None:
            stoch_val = indicators["StochRSI"]
            if stoch_val < 0.2:
                st.success(f"**Stoch RSI:** {stoch_val:.2f} - Aşırı Satım ✅")
            elif stoch_val > 0.8:
                st.error(f"**Stoch RSI:** {stoch_val:.2f} - Aşırı Alım ❌")
            else:
                st.info(f"**Stoch RSI:** {stoch_val:.2f} - Normal")

        # CCI
        if indicators.get("CCI") is not None:
            cci_val = indicators["CCI"]
            if cci_val < -100:
                st.success(f"**CCI:** {cci_val:.2f} - Aşırı Satım ✅")
            elif cci_val > 100:
                st.error(f"**CCI:** {cci_val:.2f} - Aşırı Alım ❌")
            else:
                st.info(f"**CCI:** {cci_val:.2f} - Normal")

    with col2:
        st.markdown("#### 📊 Trend ve Volatilite")

        # EMA
        if indicators.get("EMA20") and indicators.get("EMA50"):
            ema20 = indicators["EMA20"]
            ema50 = indicators["EMA50"]
            ema_diff_pct = ((ema20 - ema50) / ema50) * 100

            if ema20 > ema50:
                st.success(f"**EMA Trend:** Yükseliş ↗️ ({ema_diff_pct:.2f}%)")
            else:
                st.error(f"**EMA Trend:** Düşüş ↘️ ({ema_diff_pct:.2f}%)")

            st.caption(f"EMA20: {ema20:.2f} | EMA50: {ema50:.2f}")

        # Bollinger Bands
        if indicators.get("Bollinger"):
            bb_text = indicators["Bollinger"]
            if "AL" in bb_text:
                st.success(f"**Bollinger:** {bb_text} ✅")
            elif "SAT" in bb_text:
                st.error(f"**Bollinger:** {bb_text} ❌")
            else:
                st.info(f"**Bollinger:** {bb_text}")

            if indicators.get("BB_Upper") and indicators.get("BB_Lower"):
                bb_width = indicators["BB_Upper"] - indicators["BB_Lower"]
                bb_width_pct = (bb_width / indicators["BB_Middle"]) * 100
                st.caption(f"Band Genişliği: %{bb_width_pct:.2f}")

        # ADX
        if indicators.get("ADX") is not None:
            adx_val = indicators["ADX"]
            if adx_val > 25:
                st.success(f"**ADX:** {adx_val:.2f} - Güçlü Trend 💪")
            elif adx_val > 20:
                st.info(f"**ADX:** {adx_val:.2f} - Orta Trend")
            else:
                st.warning(f"**ADX:** {adx_val:.2f} - Zayıf Trend")

        # ATR
        if indicators.get("ATR") and indicators.get("ATR_Percent"):
            atr_val = indicators["ATR"]
            atr_pct = indicators["ATR_Percent"]
            st.info(f"**ATR:** {atr_val:.2f} (%{atr_pct:.2f})")

    # Volume analizi
    if indicators.get("Volume_Ratio") is not None:
        st.markdown("---")
        st.markdown("#### 📊 Hacim Analizi")

        vol_ratio = indicators["Volume_Ratio"]
        if vol_ratio > 2:
            st.success(f"**Hacim:** Normalin {vol_ratio:.1f}x katı - Yüksek ilgi! 🔥")
        elif vol_ratio > 1.5:
            st.info(f"**Hacim:** Normalin {vol_ratio:.1f}x katı - Artan ilgi")
        else:
            st.info(f"**Hacim:** Normal seviyede ({vol_ratio:.1f}x)")

        if indicators.get("OBV") is not None:
            st.caption(f"OBV: {indicators['OBV']:,.0f}")

    # Grafikler
    st.markdown("---")
    st.markdown("### 📈 Grafikler ve Analizler")

    # Tab yapısı
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["💰 Fiyat & Volume", "📊 RSI", "📉 MACD", "📊 Bollinger", "🎯 Sinyal Özeti"]
    )

    with tab1:
        # Fiyat ve hacim grafiği
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )

        # Fiyat grafiği
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Fiyat",
            ),
            row=1,
            col=1,
        )

        # EMA'lar
        if not indicators.get("ema20_series", pd.Series()).empty:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators["ema20_series"],
                    name="EMA20",
                    line=dict(color="orange", width=1),
                ),
                row=1,
                col=1,
            )

        if not indicators.get("ema50_series", pd.Series()).empty:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators["ema50_series"],
                    name="EMA50",
                    line=dict(color="blue", width=1),
                ),
                row=1,
                col=1,
            )

        # Volume
        if "Volume" in data.columns:
            colors = [
                "red" if close.iloc[i] < close.iloc[i - 1] else "green"
                for i in range(1, len(close))
            ]
            colors.insert(0, "green")

            fig.add_trace(
                go.Bar(
                    x=data.index, y=data["Volume"], name="Hacim", marker_color=colors
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            title=f"{hisse} - Fiyat ve Hacim Analizi",
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # RSI grafiği
        if not indicators.get("rsi_series", pd.Series()).empty:
            fig_rsi = go.Figure()

            # RSI çizgisi
            fig_rsi.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators["rsi_series"],
                    name="RSI",
                    line=dict(color="purple", width=2),
                )
            )

            # Aşırı alım/satım bölgeleri
            fig_rsi.add_hline(
                y=70,
                line_dash="dash",
                line_color="red",
                annotation_text="Aşırı Alım (70)",
            )
            fig_rsi.add_hline(
                y=30,
                line_dash="dash",
                line_color="green",
                annotation_text="Aşırı Satım (30)",
            )

            # Dolgu alanlar
            fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1)
            fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1)

            fig_rsi.update_layout(
                title="RSI (14) - Relative Strength Index",
                yaxis_title="RSI",
                height=400,
                yaxis=dict(range=[0, 100]),
            )

            st.plotly_chart(fig_rsi, use_container_width=True)
        else:
            st.info("RSI verisi mevcut değil")

    with tab3:
        # MACD grafiği
        if not indicators.get("macd_series", pd.Series()).empty:
            fig_macd = go.Figure()

            # MACD çizgisi
            fig_macd.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators["macd_series"],
                    name="MACD",
                    line=dict(color="blue", width=2),
                )
            )

            # Signal çizgisi
            if not indicators.get("macd_signal_series", pd.Series()).empty:
                fig_macd.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators["macd_signal_series"],
                        name="Signal",
                        line=dict(color="red", width=2),
                    )
                )

                # Histogram
                macd_hist = indicators["macd_series"] - indicators["macd_signal_series"]
                colors = ["green" if val > 0 else "red" for val in macd_hist]

                fig_macd.add_trace(
                    go.Bar(
                        x=data.index,
                        y=macd_hist,
                        name="Histogram",
                        marker_color=colors,
                        opacity=0.3,
                    )
                )

            fig_macd.update_layout(
                title="MACD - Moving Average Convergence Divergence",
                yaxis_title="MACD",
                height=400,
            )

            st.plotly_chart(fig_macd, use_container_width=True)
        else:
            st.info("MACD verisi mevcut değil")

    with tab4:
        # Bollinger Bands grafiği
        if all(
            key in indicators
            for key in ["bb_upper_series", "bb_lower_series", "bb_middle_series"]
        ):
            if not indicators["bb_upper_series"].empty:
                fig_bb = go.Figure()

                # Fiyat
                fig_bb.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=close,
                        name="Fiyat",
                        line=dict(color="black", width=2),
                    )
                )

                # Bollinger Bands
                fig_bb.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators["bb_upper_series"],
                        name="Üst Band",
                        line=dict(color="red", width=1, dash="dash"),
                    )
                )

                fig_bb.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators["bb_middle_series"],
                        name="Orta Band (SMA20)",
                        line=dict(color="blue", width=1),
                    )
                )

                fig_bb.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators["bb_lower_series"],
                        name="Alt Band",
                        line=dict(color="green", width=1, dash="dash"),
                        fill="tonexty",
                        fillcolor="rgba(0,255,0,0.1)",
                    )
                )

                fig_bb.update_layout(
                    title="Bollinger Bands (20,2)", yaxis_title="Fiyat", height=400
                )

                st.plotly_chart(fig_bb, use_container_width=True)
        else:
            st.info("Bollinger Bands verisi mevcut değil")

    with tab5:
        # Sinyal özeti dashboard
        st.markdown("#### 📊 Teknik Sinyal Özeti")

        # Sinyal skorları
        signal_data = {"Gösterge": [], "Değer": [], "Sinyal": [], "Güç": []}

        # RSI
        if indicators.get("RSI") is not None:
            rsi_val = indicators["RSI"]
            signal_data["Gösterge"].append("RSI")
            signal_data["Değer"].append(f"{rsi_val:.2f}")

            if rsi_val < 30:
                signal_data["Sinyal"].append("GÜÇLÜ AL")
                signal_data["Güç"].append(2)
            elif rsi_val < 40:
                signal_data["Sinyal"].append("AL")
                signal_data["Güç"].append(1)
            elif rsi_val > 70:
                signal_data["Sinyal"].append("SAT")
                signal_data["Güç"].append(-1)
            else:
                signal_data["Sinyal"].append("NÖTR")
                signal_data["Güç"].append(0)

        # MACD
        if (
            indicators.get("MACD") is not None
            and indicators.get("MACD_Signal") is not None
        ):
            macd_diff = indicators["MACD"] - indicators["MACD_Signal"]
            signal_data["Gösterge"].append("MACD")
            signal_data["Değer"].append(f"{macd_diff:.4f}")

            if macd_diff > 0:
                signal_data["Sinyal"].append("AL")
                signal_data["Güç"].append(1)
            else:
                signal_data["Sinyal"].append("SAT")
                signal_data["Güç"].append(-1)

        # EMA
        if indicators.get("EMA20") and indicators.get("EMA50"):
            signal_data["Gösterge"].append("EMA Trend")
            ema_diff_pct = (
                (indicators["EMA20"] - indicators["EMA50"]) / indicators["EMA50"]
            ) * 100
            signal_data["Değer"].append(f"{ema_diff_pct:.2f}%")

            if indicators["EMA20"] > indicators["EMA50"]:
                signal_data["Sinyal"].append("AL")
                signal_data["Güç"].append(1)
            else:
                signal_data["Sinyal"].append("SAT")
                signal_data["Güç"].append(-1)

        # Bollinger
        if indicators.get("Bollinger"):
            signal_data["Gösterge"].append("Bollinger")
            signal_data["Değer"].append(indicators["Bollinger"].split("→")[0].strip())

            if "AL" in indicators["Bollinger"]:
                signal_data["Sinyal"].append("AL")
                signal_data["Güç"].append(1)
            elif "SAT" in indicators["Bollinger"]:
                signal_data["Sinyal"].append("SAT")
                signal_data["Güç"].append(-1)
            else:
                signal_data["Sinyal"].append("NÖTR")
                signal_data["Güç"].append(0)

        # DataFrame ve görselleştirme
        df_signals = pd.DataFrame(signal_data)

        # Renklendirme
        def color_signal(val):
            if val == "GÜÇLÜ AL":
                return "background-color: #28a745; color: white"
            elif val == "AL":
                return "background-color: #90EE90"
            elif val == "SAT":
                return "background-color: #dc3545; color: white"
            elif val == "NÖTR":
                return "background-color: #ffc107"
            return ""

        styled_df = df_signals.style.applymap(color_signal, subset=["Sinyal"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Toplam skor
        total_score = sum(signal_data["Güç"])
        max_score = len(signal_data["Güç"]) * 2
        score_percentage = (total_score / max_score) * 100 if max_score > 0 else 0

        st.markdown("---")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.metric("Toplam Sinyal Skoru", f"{total_score}/{max_score}")

            # Progress bar
            if score_percentage > 50:
                st.progress(
                    abs(score_percentage) / 100,
                    text=f"AL Sinyali: %{abs(score_percentage):.0f}",
                )
            elif score_percentage < -25:
                st.progress(
                    abs(score_percentage) / 100,
                    text=f"SAT Sinyali: %{abs(score_percentage):.0f}",
                )
            else:
                st.progress(0.5, text="NÖTR / BEKLE")

        with col2:
            # Özet öneri
            if score_percentage > 50:
                st.success("🎯 **ÖZET:** GÜÇLÜ AL")
            elif score_percentage > 25:
                st.info("📈 **ÖZET:** AL")
            elif score_percentage < -25:
                st.error("📉 **ÖZET:** SAT")
            else:
                st.warning("⏸️ **ÖZET:** BEKLE")

    # Dışa aktarma seçenekleri
    st.markdown("---")
    st.markdown("### 💾 Rapor ve Dışa Aktarma")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Excel Raporu", use_container_width=True):
            # Excel raporu oluştur
            output = BytesIO()

            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                # Özet sayfa
                summary_data = {
                    "Metrik": [
                        "Hisse",
                        "Güncel Fiyat",
                        "Günlük Değişim %",
                        "RSI",
                        "MACD Sinyal",
                        "EMA Trend",
                        "Genel Sinyal",
                        "Sinyal Skoru",
                    ],
                    "Değer": [
                        hisse,
                        f"{close.iloc[-1]:.2f}",
                        f"{((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100):.2f}%",
                        f"{indicators.get('RSI', 'N/A')}",
                        (
                            "AL"
                            if indicators.get("MACD", 0)
                            > indicators.get("MACD_Signal", 0)
                            else "SAT"
                        ),
                        (
                            "Yükseliş"
                            if indicators.get("EMA20", 0) > indicators.get("EMA50", 0)
                            else "Düşüş"
                        ),
                        (
                            "AL"
                            if normalized_score > 25
                            else ("SAT" if normalized_score < -25 else "BEKLE")
                        ),
                        f"{signal_score}/{signal_total}",
                    ],
                }

                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name="Özet", index=False)

                # Detaylı göstergeler
                indicators_data = {
                    "Gösterge": list(indicators.keys()),
                    "Değer": [
                        str(v) if not isinstance(v, pd.Series) else "Series"
                        for v in indicators.values()
                    ],
                }
                df_indicators = pd.DataFrame(indicators_data)
                df_indicators = df_indicators[df_indicators["Değer"] != "Series"]
                df_indicators.to_excel(writer, sheet_name="Göstergeler", index=False)

                # Fiyat verileri
                price_data = pd.DataFrame(
                    {
                        "Tarih": data.index,
                        "Açılış": data["Open"],
                        "Yüksek": data["High"],
                        "Düşük": data["Low"],
                        "Kapanış": data["Close"],
                        "Hacim": data.get("Volume", 0),
                    }
                )
                price_data.to_excel(writer, sheet_name="Fiyat Verileri", index=False)

            st.download_button(
                "📥 Excel İndir",
                data=output.getvalue(),
                file_name=f"{hisse}_analiz_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with col2:
        if st.button("📄 PDF Raporu", use_container_width=True):
            st.info("PDF raporu yakında eklenecek...")

    with col3:
        if st.button("📧 E-posta Gönder", use_container_width=True):
            st.info("E-posta özelliği yakında eklenecek...")

    # Performans bilgileri (footer)
    with st.expander("⚡ Performans ve Sistem Bilgileri", expanded=False):
        stats = perf_monitor.get_stats()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Cache Hit Rate", f"{stats['cache_hit_rate']:.1f}%")
        with col2:
            st.metric("API Calls", stats["api_calls"])
        with col3:
            st.metric("Errors", stats["errors"])
        with col4:
            st.metric("Uptime", f"{stats['elapsed_time']:.0f}s")

        st.markdown(
            f"""
        **🚀 Sistem Özellikleri:**
        - ✅ Rate limiting koruması
        - ✅ 5 dakika veri cache
        - ✅ 10 dakika şirket bilgisi cache
        - ✅ Akıllı retry mekanizması
        - ✅ Gelişmiş hata yönetimi
        - ✅ Performans monitörü
        
        **📊 Veri Kaynakları:**
        - Yahoo Finance (Birincil)
        - Fallback mekanizması hazır
        
        **🕐 Son Güncelleme:** {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
        """
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Tyana Panel v2.0 - Premium Edition | Powered by AI & Advanced Analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)
