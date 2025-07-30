import streamlit as st
import yfinance as yf
import ta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging
from datetime import datetime
import time
import random
import json
from typing import Dict, List, Optional, Tuple
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Performans için uyarıları kapat
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Logger ayarla
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ==============================================================================
# RATE LIMITED DATA FETCHER
# ==============================================================================


class RateLimitedBatchFetcher:
    """Rate limiting ile toplu veri çekme"""

    def __init__(
        self, max_retries: int = 3, base_delay: float = 0.5, batch_size: int = 20
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.batch_size = batch_size
        self.failed_symbols = set()
        self.success_count = 0
        self.error_count = 0

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_single_stock(
        _self, symbol: str, period: str = "3mo"
    ) -> Optional[pd.DataFrame]:
        """Tek hisse verisi çek"""
        for attempt in range(_self.max_retries):
            try:
                # Rate limiting delay
                delay = _self.base_delay * (0.5 + random.random())
                time.sleep(delay)

                data = yf.download(
                    symbol, period=period, interval="1d", progress=False, threads=False
                )

                if not data.empty and len(data) >= 30:
                    _self.success_count += 1
                    return data

            except Exception as e:
                if attempt < _self.max_retries - 1:
                    wait_time = _self.base_delay * (2**attempt) + random.random()
                    time.sleep(wait_time)
                else:
                    _self.failed_symbols.add(symbol)
                    _self.error_count += 1
                    logger.error(f"Failed to fetch {symbol}: {str(e)}")

        return None

    def fetch_batch_optimized(
        self, symbols: List[str], period: str = "3mo"
    ) -> Dict[str, pd.DataFrame]:
        """Optimize edilmiş batch veri çekme"""
        results = {}

        # Progress takibi
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

            col1, col2, col3 = st.columns(3)
            with col1:
                success_metric = st.empty()
            with col2:
                error_metric = st.empty()
            with col3:
                rate_metric = st.empty()

        # Batch'lere böl
        batches = [
            symbols[i : i + self.batch_size]
            for i in range(0, len(symbols), self.batch_size)
        ]

        total_processed = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(batches):
            batch_start = time.time()

            # Batch içinde paralel işleme
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_symbol = {
                    executor.submit(self.fetch_single_stock, symbol, period): symbol
                    for symbol in batch
                }

                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    total_processed += 1

                    try:
                        data = future.result()
                        if data is not None:
                            results[symbol] = data
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")

                    # Progress güncelle
                    progress = total_processed / len(symbols)
                    progress_bar.progress(progress)
                    status_text.text(
                        f"📊 İşleniyor: {symbol} ({total_processed}/{len(symbols)})"
                    )

                    # Metrikler
                    success_metric.metric("✅ Başarılı", self.success_count)
                    error_metric.metric("❌ Hata", self.error_count)

                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        rate = total_processed / elapsed
                        rate_metric.metric("⚡ Hız", f"{rate:.1f} hisse/s")

            # Batch arası bekleme
            if batch_idx < len(batches) - 1:
                batch_elapsed = time.time() - batch_start
                if batch_elapsed < 2:  # Minimum 2 saniye/batch
                    time.sleep(2 - batch_elapsed)

        # Progress temizle
        progress_container.empty()

        return results


# ==============================================================================
# TEKNİK ANALİZ FONKSİYONLARI
# ==============================================================================


@st.cache_data(ttl=600, show_spinner=False)
def calculate_enhanced_signals(symbol: str, data: pd.DataFrame) -> Optional[Dict]:
    """Gelişmiş sinyal hesaplama"""
    try:
        # DataFrame kontrolleri
        if "Close" not in data.columns or len(data) < 30:
            return None

        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        # NaN değerleri temizle
        close = close.dropna()
        if len(close) < 30:
            return None

        high = data["High"]
        low = data["Low"]

        # Sonuç dictionary
        result = {
            "symbol": symbol,
            "current_price": close.iloc[-1],
            "signals": [],
            "score": 0,
            "indicators": {},
        }

        # 1. RSI
        try:
            rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            rsi_val = rsi.iloc[-1]
            if pd.notna(rsi_val):
                result["indicators"]["rsi"] = rsi_val

                if rsi_val < 30:
                    result["score"] += 2
                    result["signals"].append("RSI Güçlü Aşırı Satım")
                elif rsi_val < 40:
                    result["score"] += 1
                    result["signals"].append("RSI Aşırı Satım")
                elif rsi_val > 70:
                    result["score"] -= 1
                    result["signals"].append("RSI Aşırı Alım")
        except:
            pass

        # 2. EMA
        try:
            ema20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
            ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()

            if pd.notna(ema20.iloc[-1]) and pd.notna(ema50.iloc[-1]):
                result["indicators"]["ema20"] = ema20.iloc[-1]
                result["indicators"]["ema50"] = ema50.iloc[-1]
                result["trend"] = (
                    "YUKARI" if ema20.iloc[-1] > ema50.iloc[-1] else "ASAGI"
                )

                if ema20.iloc[-1] > ema50.iloc[-1]:
                    result["score"] += 1
                    result["signals"].append("EMA Yükseliş Trendi")
                else:
                    result["score"] -= 1
                    result["signals"].append("EMA Düşüş Trendi")
        except:
            result["trend"] = "BILINMIYOR"

        # 3. MACD
        try:
            macd = ta.trend.MACD(close=close)
            macd_val = macd.macd().iloc[-1]
            macd_signal_val = macd.macd_signal().iloc[-1]

            if pd.notna(macd_val) and pd.notna(macd_signal_val):
                result["indicators"]["macd"] = macd_val
                result["indicators"]["macd_signal"] = macd_signal_val

                if macd_val > macd_signal_val:
                    result["score"] += 1
                    result["signals"].append("MACD Pozitif Kesişim")
                else:
                    result["score"] -= 1
        except:
            pass

        # 4. Bollinger Bands
        try:
            bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            bb_middle = bb.bollinger_mavg().iloc[-1]
            current_price = close.iloc[-1]

            if pd.notna(bb_upper) and pd.notna(bb_lower):
                result["indicators"]["bb_upper"] = bb_upper
                result["indicators"]["bb_lower"] = bb_lower
                result["indicators"]["bb_middle"] = bb_middle

                # Band pozisyonu
                if current_price < bb_lower:
                    result["indicators"]["bb_position"] = "Alt"
                    result["score"] += 1
                    result["signals"].append("Bollinger Alt Band")
                elif current_price > bb_upper:
                    result["indicators"]["bb_position"] = "Üst"
                    result["score"] -= 1
                    result["signals"].append("Bollinger Üst Band")
                else:
                    result["indicators"]["bb_position"] = "Orta"

                # Band genişliği (volatilite)
                band_width = (bb_upper - bb_lower) / bb_middle
                result["indicators"]["bb_width"] = band_width
        except:
            result["indicators"]["bb_position"] = "Bilinmiyor"

        # 5. Volume analizi
        try:
            if "Volume" in data.columns:
                volume = data["Volume"].dropna()
                if len(volume) >= 20:
                    vol_ma = volume.rolling(20).mean()
                    latest_volume = volume.iloc[-1]
                    avg_volume = vol_ma.iloc[-1]

                    if (
                        pd.notna(latest_volume)
                        and pd.notna(avg_volume)
                        and avg_volume > 0
                    ):
                        volume_ratio = latest_volume / avg_volume
                        result["indicators"]["volume_ratio"] = volume_ratio

                        if volume_ratio > 2.0:
                            result["indicators"]["volume_trend"] = "Çok Yüksek"
                            result["score"] += 1
                            result["signals"].append("Anormal Yüksek Hacim")
                        elif volume_ratio > 1.5:
                            result["indicators"]["volume_trend"] = "Yüksek"
                            result["score"] += 0.5
                            result["signals"].append("Yüksek Hacim")
                        else:
                            result["indicators"]["volume_trend"] = "Normal"
                    else:
                        result["indicators"]["volume_trend"] = "Normal"
                        result["indicators"]["volume_ratio"] = 1.0
        except:
            result["indicators"]["volume_trend"] = "Bilinmiyor"

        # 6. ADX (Trend Gücü)
        try:
            adx = ta.trend.ADXIndicator(high=high, low=low, close=close).adx()
            adx_val = adx.iloc[-1]
            if pd.notna(adx_val):
                result["indicators"]["adx"] = adx_val
                if adx_val > 25:
                    result["signals"].append("Güçlü Trend")
                    result["score"] += 0.5
        except:
            pass

        # 7. ATR (Volatilite)
        try:
            atr = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close
            ).average_true_range()
            atr_val = atr.iloc[-1]
            if pd.notna(atr_val):
                atr_pct = (atr_val / close.iloc[-1]) * 100
                result["indicators"]["atr"] = atr_val
                result["indicators"]["atr_percent"] = atr_pct
        except:
            pass

        # 8. Stochastic RSI
        try:
            stoch_rsi = ta.momentum.StochRSIIndicator(close=close).stochrsi()
            stoch_val = stoch_rsi.iloc[-1]
            if pd.notna(stoch_val):
                result["indicators"]["stoch_rsi"] = stoch_val
                if stoch_val < 0.2:
                    result["score"] += 0.5
                    result["signals"].append("Stoch RSI Aşırı Satım")
                elif stoch_val > 0.8:
                    result["score"] -= 0.5
        except:
            pass

        # 9. Price change
        try:
            price_change_1d = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
            price_change_5d = (
                ((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]) * 100
                if len(close) > 5
                else 0
            )
            price_change_20d = (
                ((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]) * 100
                if len(close) > 20
                else 0
            )

            result["indicators"]["change_1d"] = price_change_1d
            result["indicators"]["change_5d"] = price_change_5d
            result["indicators"]["change_20d"] = price_change_20d

            # Momentum skorlama
            if price_change_5d > 5 and price_change_1d > 0:
                result["score"] += 0.5
                result["signals"].append("Pozitif Momentum")
            elif price_change_5d < -5 and price_change_1d < 0:
                result["score"] -= 0.5
        except:
            pass

        # 10. Support/Resistance yakınlığı
        try:
            # Son 20 günün en yüksek ve en düşük değerleri
            recent_high = high[-20:].max()
            recent_low = low[-20:].min()
            current = close.iloc[-1]

            # Destek/direnç yakınlığı
            to_resistance = ((recent_high - current) / current) * 100
            to_support = ((current - recent_low) / current) * 100

            result["indicators"]["to_resistance"] = to_resistance
            result["indicators"]["to_support"] = to_support

            if to_support < 5:  # Desteğe %5'ten yakın
                result["score"] += 0.5
                result["signals"].append("Destek Bölgesinde")
            elif to_resistance < 5:  # Dirence %5'ten yakın
                result["score"] -= 0.5
                result["signals"].append("Direnç Bölgesinde")
        except:
            pass

        return result

    except Exception as e:
        logger.error(f"Error calculating signals for {symbol}: {str(e)}")
        return None


# ==============================================================================
# PERFORMANS MONİTÖRÜ
# ==============================================================================


class PerformanceMonitor:
    """Performans takibi ve raporlama"""

    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            "api_calls": 0,
            "cache_hits": 0,
            "errors": 0,
            "processed_symbols": 0,
        }

    def record_api_call(self):
        self.metrics["api_calls"] += 1

    def record_cache_hit(self):
        self.metrics["cache_hits"] += 1

    def record_error(self):
        self.metrics["errors"] += 1

    def record_processed_symbol(self):
        self.metrics["processed_symbols"] += 1

    def get_report(self) -> Dict:
        elapsed = time.time() - self.start_time

        return {
            "elapsed_time": elapsed,
            "api_calls": self.metrics["api_calls"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_hit_rate": self.metrics["cache_hits"]
            / max(self.metrics["api_calls"], 1)
            * 100,
            "errors": self.metrics["errors"],
            "processed_symbols": self.metrics[
                "processed_symbols"
            ],  # ← Bu satırı ekleyin
            "error_rate": self.metrics["errors"]
            / max(self.metrics["processed_symbols"], 1)
            * 100,
            "symbols_per_second": self.metrics["processed_symbols"] / max(elapsed, 1),
        }


# ==============================================================================
# HİSSE LİSTELERİ
# ==============================================================================

STOCK_LISTS = {
    "BIST30": [
        "AKBNK.IS",
        "ARCLK.IS",
        "ASELS.IS",
        "BIMAS.IS",
        "DOHOL.IS",
        "EKGYO.IS",
        "ENKAI.IS",
        "EREGL.IS",
        "FROTO.IS",
        "GARAN.IS",
        "GUBRF.IS",
        "HALKB.IS",
        "ISCTR.IS",
        "KCHOL.IS",
        "KOZAL.IS",
        "KRDMD.IS",
        "ODAS.IS",
        "PETKM.IS",
        "PGSUS.IS",
        "SAHOL.IS",
        "SASA.IS",
        "SISE.IS",
        "TAVHL.IS",
        "TCELL.IS",
        "THYAO.IS",
        "TKFEN.IS",
        "TOASO.IS",
        "TUPRS.IS",
        "VAKBN.IS",
        "YKBNK.IS",
    ],
    "BIST50_EXTRA": [
        "AEFES.IS",
        "AGHOL.IS",
        "AKSEN.IS",
        "ALARK.IS",
        "ALBRK.IS",
        "ALFAS.IS",
        "ANHYT.IS",
        "ANSGR.IS",
        "AYGAZ.IS",
        "BAGFS.IS",
        "BANVT.IS",
        "BIOEN.IS",
        "BIZIM.IS",
        "BRISA.IS",
        "BRYAT.IS",
        "BUCIM.IS",
        "CCOLA.IS",
        "CEMTS.IS",
        "CIMSA.IS",
        "DOAS.IS",
    ],
    "BIST_TEKNOLOJI": [
        "ALCTL.IS",
        "ARDYZ.IS",
        "ARENA.IS",
        "ARMDA.IS",
        "ARZUM.IS",
        "ASGYO.IS",
        "ASTOR.IS",
        "ATATP.IS",
        "AVGYO.IS",
        "AVTUR.IS",
        "AYCES.IS",
        "AYEN.IS",
        "AZTEK.IS",
        "BAKAB.IS",
        "BARMA.IS",
        "BASCM.IS",
        "BASGZ.IS",
        "BERA.IS",
        "BEYAZ.IS",
        "BIENY.IS",
    ],
    "US_MEGA_CAP": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "BRK-B",
        "JPM",
        "JNJ",
        "V",
        "UNH",
        "HD",
        "PG",
        "MA",
    ],
    "US_TECH": [
        "ORCL",
        "CRM",
        "CSCO",
        "ACN",
        "ADBE",
        "AVGO",
        "INTC",
        "AMD",
        "QCOM",
        "TXN",
        "INTU",
        "IBM",
        "NOW",
        "AMAT",
        "MU",
    ],
    "CRYPTO": [
        "BTC-USD",
        "ETH-USD",
        "BNB-USD",
        "XRP-USD",
        "ADA-USD",
        "DOGE-USD",
        "SOL-USD",
        "DOT-USD",
        "MATIC-USD",
        "AVAX-USD",
    ],
}

# ==============================================================================
# KULLANICI ARAYÜZÜ FONKSİYONLARI
# ==============================================================================


def create_filter_ui() -> Dict:
    """Gelişmiş filtre arayüzü"""
    filters = {}

    with st.expander("🎯 Gelişmiş Filtreler", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📊 Teknik Filtreler**")

            filters["min_score"] = st.slider(
                "Min Sinyal Skoru",
                min_value=-5,
                max_value=5,
                value=2,
                help="Minimum toplam sinyal skoru",
            )

            filters["rsi_range"] = st.slider(
                "RSI Aralığı",
                min_value=0,
                max_value=100,
                value=(20, 80),
                step=5,
                help="RSI değer aralığı",
            )

            filters["trend_filter"] = st.selectbox(
                "Trend Filtresi",
                ["Tümü", "Sadece Yükseliş", "Sadece Düşüş"],
                help="EMA bazlı trend filtresi",
            )

        with col2:
            st.markdown("**💰 Fiyat Filtreleri**")

            filters["price_change_1d"] = st.slider(
                "Günlük Değişim %",
                min_value=-10,
                max_value=10,
                value=(-10, 10),
                step=1,
                help="1 günlük fiyat değişimi",
            )

            filters["price_change_5d"] = st.slider(
                "5 Günlük Değişim %",
                min_value=-20,
                max_value=20,
                value=(-20, 20),
                step=1,
                help="5 günlük fiyat değişimi",
            )

            filters["exclude_penny"] = st.checkbox(
                "Penny Stock Hariç Tut",
                value=False,
                help="5 TL altı hisseleri hariç tut",
            )

        with col3:
            st.markdown("**📊 Hacim Filtreleri**")

            filters["volume_filter"] = st.selectbox(
                "Hacim Filtresi",
                ["Tümü", "Yüksek Hacim", "Çok Yüksek Hacim"],
                help="20 günlük ortalamaya göre",
            )

            filters["volume_ratio"] = st.number_input(
                "Min Hacim Çarpanı",
                min_value=1.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Ortalama hacmin kaç katı",
            )

            filters["bb_position"] = st.multiselect(
                "Bollinger Band Pozisyonu",
                ["Alt", "Orta", "Üst"],
                default=["Alt", "Orta", "Üst"],
                help="Bollinger band pozisyonları",
            )

    return filters


def apply_filters(results: List[Dict], filters: Dict) -> List[Dict]:
    """Filtreleri uygula"""
    filtered = []

    for result in results:
        # Sinyal skoru filtresi
        if result.get("score", 0) < filters["min_score"]:
            continue

        # RSI filtresi
        rsi = result.get("indicators", {}).get("rsi")
        if rsi is not None:
            if rsi < filters["rsi_range"][0] or rsi > filters["rsi_range"][1]:
                continue

        # Trend filtresi
        if (
            filters["trend_filter"] == "Sadece Yükseliş"
            and result.get("trend") != "YUKARI"
        ):
            continue
        elif (
            filters["trend_filter"] == "Sadece Düşüş" and result.get("trend") != "ASAGI"
        ):
            continue

        # Fiyat değişim filtreleri
        change_1d = result.get("indicators", {}).get("change_1d", 0)
        if (
            change_1d < filters["price_change_1d"][0]
            or change_1d > filters["price_change_1d"][1]
        ):
            continue

        change_5d = result.get("indicators", {}).get("change_5d", 0)
        if (
            change_5d < filters["price_change_5d"][0]
            or change_5d > filters["price_change_5d"][1]
        ):
            continue

        # Penny stock filtresi
        if filters["exclude_penny"] and result.get("current_price", 0) < 5:
            continue

        # Hacim filtresi
        volume_trend = result.get("indicators", {}).get("volume_trend", "Normal")
        volume_ratio = result.get("indicators", {}).get("volume_ratio", 1.0)

        if filters["volume_filter"] == "Yüksek Hacim" and volume_trend not in [
            "Yüksek",
            "Çok Yüksek",
        ]:
            continue
        elif (
            filters["volume_filter"] == "Çok Yüksek Hacim"
            and volume_trend != "Çok Yüksek"
        ):
            continue

        if volume_ratio < filters["volume_ratio"]:
            continue

        # Bollinger band filtresi
        bb_pos = result.get("indicators", {}).get("bb_position", "Bilinmiyor")
        if bb_pos != "Bilinmiyor" and bb_pos not in filters["bb_position"]:
            continue

        filtered.append(result)

    return filtered


def display_results(results: List[Dict], page_size: int = 20):
    """Sonuçları görüntüle"""
    if not results:
        st.warning("📭 Filtrelere uygun sonuç bulunamadı.")
        return

    # Sıralama seçenekleri
    col1, col2 = st.columns([3, 1])

    with col1:
        sort_by = st.selectbox(
            "Sıralama",
            [
                "Sinyal Skoru (Yüksek→Düşük)",
                "RSI (Düşük→Yüksek)",
                "Günlük Değişim (Yüksek→Düşük)",
                "Hacim (Yüksek→Düşük)",
                "Fiyat (Düşük→Yüksek)",
            ],
        )

    with col2:
        export_format = st.selectbox("Export", ["Görüntüle", "Excel", "CSV", "JSON"])

    # Sıralama uygula
    if sort_by == "Sinyal Skoru (Yüksek→Düşük)":
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
    elif sort_by == "RSI (Düşük→Yüksek)":
        results.sort(key=lambda x: x.get("indicators", {}).get("rsi", 100))
    elif sort_by == "Günlük Değişim (Yüksek→Düşük)":
        results.sort(
            key=lambda x: x.get("indicators", {}).get("change_1d", 0), reverse=True
        )
    elif sort_by == "Hacim (Yüksek→Düşük)":
        results.sort(
            key=lambda x: x.get("indicators", {}).get("volume_ratio", 0), reverse=True
        )
    else:  # Fiyat
        results.sort(key=lambda x: x.get("current_price", 0))

    # Export işlemleri
    if export_format != "Görüntüle":
        export_data(results, export_format)
        return

    # Özet istatistikler
    st.markdown("### 📊 Özet İstatistikler")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_count = len(results)
        st.metric("Toplam Sinyal", total_count)

    with col2:
        strong_signals = len([r for r in results if r.get("score", 0) >= 4])
        st.metric("Güçlü Sinyaller", strong_signals)

    with col3:
        uptrend_count = len([r for r in results if r.get("trend") == "YUKARI"])
        st.metric("Yükseliş Trendi", uptrend_count)

    with col4:
        high_volume = len(
            [
                r
                for r in results
                if r.get("indicators", {}).get("volume_trend")
                in ["Yüksek", "Çok Yüksek"]
            ]
        )
        st.metric("Yüksek Hacim", high_volume)

    with col5:
        avg_score = (
            sum(r.get("score", 0) for r in results) / len(results) if results else 0
        )
        st.metric("Ort. Skor", f"{avg_score:.2f}")

    # Sayfalama
    total_pages = (len(results) - 1) // page_size + 1

    if total_pages > 1:
        page = st.selectbox(
            "Sayfa",
            range(1, total_pages + 1),
            format_func=lambda x: f"Sayfa {x} / {total_pages}",
        )
    else:
        page = 1

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(results))

    page_results = results[start_idx:end_idx]

    # Ana tablo
    st.markdown("### 📈 Tarama Sonuçları")

    # DataFrame oluştur
    df_data = []
    for r in page_results:
        indicators = r.get("indicators", {})

        row = {
            "Hisse": r["symbol"],
            "Fiyat": f"{r.get('current_price', 0):.2f}",
            "Skor": r.get("score", 0),
            "RSI": (
                f"{indicators.get('rsi', 0):.1f}" if indicators.get("rsi") else "N/A"
            ),
            "1G %": (
                f"{indicators.get('change_1d', 0):.2f}%"
                if indicators.get("change_1d") is not None
                else "N/A"
            ),
            "5G %": (
                f"{indicators.get('change_5d', 0):.2f}%"
                if indicators.get("change_5d") is not None
                else "N/A"
            ),
            "Trend": "↗️" if r.get("trend") == "YUKARI" else "↘️",
            "Hacim": indicators.get("volume_trend", "N/A"),
            "BB": indicators.get("bb_position", "N/A"),
            "Sinyaller": ", ".join(r.get("signals", [])[:2]),  # İlk 2 sinyal
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)

    # Renklendirme fonksiyonları
    def color_score(val):
        try:
            if isinstance(val, str):
                return ""
            if val >= 4:
                return "background-color: #28a745; color: white"
            elif val >= 2:
                return "background-color: #90EE90"
            elif val < 0:
                return "background-color: #dc3545; color: white"
            return ""
        except:
            return ""

    def color_change(val):
        try:
            if isinstance(val, str) and "%" in val:
                num = float(val.replace("%", ""))
                if num > 5:
                    return "color: #28a745; font-weight: bold"
                elif num < -5:
                    return "color: #dc3545; font-weight: bold"
            return ""
        except:
            return ""

    # Stil uygula
    styled_df = df.style.applymap(color_score, subset=["Skor"]).applymap(
        color_change, subset=["1G %", "5G %"]
    )

    st.dataframe(styled_df, use_container_width=True, height=600)

    # Detaylı görünüm
    st.markdown("---")
    st.markdown("### 🔍 Detaylı Analiz")

    # En iyi 5 hisse detayı
    top_5 = page_results[:5]

    for idx, stock in enumerate(top_5, 1):
        with st.expander(
            f"🏆 {idx}. {stock['symbol']} - Skor: {stock.get('score', 0)}",
            expanded=(idx == 1),
        ):
            col1, col2, col3 = st.columns(3)

            indicators = stock.get("indicators", {})

            with col1:
                st.markdown("**📊 Fiyat & Değişim**")
                st.metric("Güncel Fiyat", f"{stock.get('current_price', 0):.2f}")

                change_1d = indicators.get("change_1d", 0)
                st.metric("1 Günlük", f"{change_1d:.2f}%", delta=f"{change_1d:.2f}%")

                change_5d = indicators.get("change_5d", 0)
                st.metric("5 Günlük", f"{change_5d:.2f}%", delta=f"{change_5d:.2f}%")

                if indicators.get("change_20d") is not None:
                    change_20d = indicators.get("change_20d", 0)
                    st.metric(
                        "20 Günlük", f"{change_20d:.2f}%", delta=f"{change_20d:.2f}%"
                    )

            with col2:
                st.markdown("**📈 Teknik Göstergeler**")

                # RSI
                if indicators.get("rsi") is not None:
                    rsi_val = indicators["rsi"]
                    rsi_text = (
                        "Aşırı Satım"
                        if rsi_val < 30
                        else ("Aşırı Alım" if rsi_val > 70 else "Normal")
                    )
                    st.info(f"**RSI:** {rsi_val:.1f} ({rsi_text})")

                # EMA Trend
                if indicators.get("ema20") and indicators.get("ema50"):
                    ema_diff = (
                        (indicators["ema20"] - indicators["ema50"])
                        / indicators["ema50"]
                    ) * 100
                    trend_icon = "↗️" if ema_diff > 0 else "↘️"
                    st.info(f"**EMA Trend:** {trend_icon} {ema_diff:.2f}%")

                # Bollinger
                if indicators.get("bb_position"):
                    bb_emoji = (
                        "🟢"
                        if indicators["bb_position"] == "Alt"
                        else ("🔴" if indicators["bb_position"] == "Üst" else "🟡")
                    )
                    st.info(
                        f"**Bollinger:** {bb_emoji} {indicators['bb_position']} Band"
                    )

                # ADX
                if indicators.get("adx") is not None:
                    adx_val = indicators["adx"]
                    adx_text = "Güçlü" if adx_val > 25 else "Zayıf"
                    st.info(f"**ADX:** {adx_val:.1f} ({adx_text} Trend)")

            with col3:
                st.markdown("**🎯 Sinyaller & Hacim**")

                # Aktif sinyaller
                st.success("**Aktif Sinyaller:**")
                for signal in stock.get("signals", []):
                    st.write(f"• {signal}")

                # Hacim durumu
                if indicators.get("volume_ratio") is not None:
                    vol_ratio = indicators["volume_ratio"]
                    vol_emoji = (
                        "🔥" if vol_ratio > 2 else ("📈" if vol_ratio > 1.5 else "📊")
                    )
                    st.info(f"**Hacim:** {vol_emoji} {vol_ratio:.1f}x ortalama")

                # ATR (Volatilite)
                if indicators.get("atr_percent") is not None:
                    atr_pct = indicators["atr_percent"]
                    vol_text = (
                        "Yüksek"
                        if atr_pct > 5
                        else ("Orta" if atr_pct > 2 else "Düşük")
                    )
                    st.info(f"**Volatilite:** {atr_pct:.1f}% ({vol_text})")

            # Mini grafik
            if st.checkbox(
                f"📊 {stock['symbol']} grafiğini göster", key=f"chart_{stock['symbol']}"
            ):
                # Veriyi çek (cache'li)
                fetcher = RateLimitedBatchFetcher()
                chart_data = fetcher.fetch_single_stock(stock["symbol"], period="1mo")

                if chart_data is not None and not chart_data.empty:
                    # Plotly grafik
                    fig = go.Figure()

                    # Candlestick
                    fig.add_trace(
                        go.Candlestick(
                            x=chart_data.index,
                            open=chart_data["Open"],
                            high=chart_data["High"],
                            low=chart_data["Low"],
                            close=chart_data["Close"],
                            name="Fiyat",
                        )
                    )

                    # Hacim
                    fig.add_trace(
                        go.Bar(
                            x=chart_data.index,
                            y=chart_data["Volume"],
                            name="Hacim",
                            yaxis="y2",
                            opacity=0.3,
                        )
                    )

                    # Layout
                    fig.update_layout(
                        title=f"{stock['symbol']} - Son 1 Ay",
                        yaxis=dict(title="Fiyat", side="left"),
                        yaxis2=dict(title="Hacim", side="right", overlaying="y"),
                        height=400,
                        xaxis_rangeslider_visible=False,
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Grafik verisi yüklenemedi")


def export_data(results: List[Dict], format: str):
    """Veriyi dışa aktar"""
    # DataFrame hazırla
    export_df = pd.DataFrame(
        [
            {
                "Hisse": r["symbol"],
                "Fiyat": r.get("current_price", 0),
                "Sinyal Skoru": r.get("score", 0),
                "RSI": r.get("indicators", {}).get("rsi", None),
                "1 Günlük Değişim %": r.get("indicators", {}).get("change_1d", None),
                "5 Günlük Değişim %": r.get("indicators", {}).get("change_5d", None),
                "20 Günlük Değişim %": r.get("indicators", {}).get("change_20d", None),
                "EMA20": r.get("indicators", {}).get("ema20", None),
                "EMA50": r.get("indicators", {}).get("ema50", None),
                "Trend": r.get("trend", "N/A"),
                "Hacim Trendi": r.get("indicators", {}).get("volume_trend", "N/A"),
                "Hacim Oranı": r.get("indicators", {}).get("volume_ratio", None),
                "Bollinger Pozisyon": r.get("indicators", {}).get("bb_position", "N/A"),
                "ADX": r.get("indicators", {}).get("adx", None),
                "ATR %": r.get("indicators", {}).get("atr_percent", None),
                "Stoch RSI": r.get("indicators", {}).get("stoch_rsi", None),
                "Sinyaller": "; ".join(r.get("signals", [])),
            }
            for r in results
        ]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format == "Excel":
        output = BytesIO()

        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # Ana veri
            export_df.to_excel(writer, sheet_name="Tarama Sonuçları", index=False)

            # Özet istatistikler
            summary_data = {
                "Metrik": [
                    "Toplam Hisse",
                    "Ortalama Skor",
                    "Güçlü Sinyal (Skor >= 4)",
                    "Orta Sinyal (2 <= Skor < 4)",
                    "Zayıf Sinyal (Skor < 2)",
                    "Yükseliş Trendi",
                    "Düşüş Trendi",
                    "Yüksek Hacimli",
                    "Ortalama RSI",
                    "Ortalama 1G Değişim %",
                ],
                "Değer": [
                    len(results),
                    f"{export_df['Sinyal Skoru'].mean():.2f}",
                    len(export_df[export_df["Sinyal Skoru"] >= 4]),
                    len(
                        export_df[
                            (export_df["Sinyal Skoru"] >= 2)
                            & (export_df["Sinyal Skoru"] < 4)
                        ]
                    ),
                    len(export_df[export_df["Sinyal Skoru"] < 2]),
                    len(export_df[export_df["Trend"] == "YUKARI"]),
                    len(export_df[export_df["Trend"] == "ASAGI"]),
                    len(
                        export_df[
                            export_df["Hacim Trendi"].isin(["Yüksek", "Çok Yüksek"])
                        ]
                    ),
                    (
                        f"{export_df['RSI'].mean():.2f}"
                        if export_df["RSI"].notna().any()
                        else "N/A"
                    ),
                    (
                        f"{export_df['1 Günlük Değişim %'].mean():.2f}%"
                        if export_df["1 Günlük Değişim %"].notna().any()
                        else "N/A"
                    ),
                ],
            }

            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name="Özet", index=False)

            # Formatting
            workbook = writer.book
            worksheet = writer.sheets["Tarama Sonuçları"]

            # Başlık formatı
            header_format = workbook.add_format(
                {"bold": True, "bg_color": "#D7E4BD", "border": 1}
            )

            for col_num, value in enumerate(export_df.columns.values):
                worksheet.write(0, col_num, value, header_format)

        st.download_button(
            "📥 Excel İndir",
            data=output.getvalue(),
            file_name=f"tyana_tarama_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    elif format == "CSV":
        csv = export_df.to_csv(index=False, encoding="utf-8-sig")

        st.download_button(
            "📥 CSV İndir",
            data=csv,
            file_name=f"tyana_tarama_{timestamp}.csv",
            mime="text/csv",
        )

    elif format == "JSON":
        json_str = json.dumps(results, ensure_ascii=False, indent=2)

        st.download_button(
            "📥 JSON İndir",
            data=json_str,
            file_name=f"tyana_tarama_{timestamp}.json",
            mime="application/json",
        )


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Tyana - Otomatik Tarama",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown(
    """
<style>
    .stProgress > div > div > div > div {
        background-color: #00cc88;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
    }
    .warning-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        margin: 10px 0;
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

# Ana başlık
st.title("📡 Tyana Otomatik Hisse Taraması - Premium")
st.markdown("🚀 **Gelişmiş AI destekli çoklu hisse analizi ve sinyal tespit sistemi**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Tarama Ayarları")

    # Piyasa seçimi
    st.markdown("#### 📊 Piyasa Seçimi")

    market_type = st.selectbox(
        "Piyasa Tipi", ["BIST", "US", "Kripto", "Özel Liste", "Tümü"]
    )

    # Alt liste seçimi
    symbol_list = []

    if market_type == "BIST":
        sub_market = st.selectbox(
            "BIST Alt Kategori",
            ["BIST30", "BIST50 Extra", "BIST Teknoloji", "BIST Tümü"],
        )

        if sub_market == "BIST30":
            symbol_list = STOCK_LISTS["BIST30"]
        elif sub_market == "BIST50 Extra":
            symbol_list = STOCK_LISTS["BIST50_EXTRA"]
        elif sub_market == "BIST Teknoloji":
            symbol_list = STOCK_LISTS["BIST_TEKNOLOJI"]
        else:  # BIST Tümü
            symbol_list = STOCK_LISTS["BIST30"] + STOCK_LISTS["BIST50_EXTRA"]

    elif market_type == "US":
        sub_market = st.selectbox(
            "US Alt Kategori", ["Mega Cap", "Technology", "US Tümü"]
        )

        if sub_market == "Mega Cap":
            symbol_list = STOCK_LISTS["US_MEGA_CAP"]
        elif sub_market == "Technology":
            symbol_list = STOCK_LISTS["US_TECH"]
        else:  # US Tümü
            symbol_list = STOCK_LISTS["US_MEGA_CAP"] + STOCK_LISTS["US_TECH"]

    elif market_type == "Kripto":
        symbol_list = STOCK_LISTS["CRYPTO"]

    elif market_type == "Özel Liste":
        custom_symbols = st.text_area(
            "Hisse Kodları (virgülle ayırın)",
            placeholder="ASELS.IS, THYAO.IS, AAPL, BTC-USD",
            height=100,
        )

        if custom_symbols:
            symbol_list = [s.strip() for s in custom_symbols.split(",") if s.strip()]

    else:  # Tümü
        symbol_list = []
        for key in STOCK_LISTS:
            symbol_list.extend(STOCK_LISTS[key])

    # Benzersiz yap ve sırala
    symbol_list = sorted(list(set(symbol_list)))

    st.info(f"📊 **{len(symbol_list)} hisse** taranacak")

    # Zaman aralığı
    st.markdown("---")
    period = st.selectbox(
        "📅 Analiz Periyodu",
        ["1mo", "3mo", "6mo", "1y"],
        index=1,
        format_func=lambda x: {
            "1mo": "1 Ay",
            "3mo": "3 Ay",
            "6mo": "6 Ay",
            "1y": "1 Yıl",
        }.get(x, x),
    )

    # Performans ayarları
    st.markdown("---")
    st.markdown("#### ⚡ Performans Ayarları")

    batch_size = st.slider(
        "Batch Boyutu",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        help="Aynı anda işlenecek hisse sayısı",
    )

    max_workers = st.slider(
        "Paralel İş Sayısı",
        min_value=1,
        max_value=10,
        value=5,
        help="Aynı anda çalışacak thread sayısı",
    )

    # Önbellek kontrolü
    st.markdown("---")
    if st.button("🗑️ Önbelleği Temizle", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Önbellek temizlendi!")

    # Bilgi
    with st.expander("ℹ️ Kullanım İpuçları", expanded=False):
        st.markdown(
            """
        **🎯 En İyi Sonuçlar İçin:**
        - RSI < 35 olan hisseleri arayın
        - Yüksek hacim filtresini kullanın
        - Güçlü sinyal (4+) arayın
        
        **⚡ Performans:**
        - Küçük batch = Daha stabil
        - Büyük batch = Daha hızlı
        - Rate limit'e dikkat!
        
        **📊 Filtreler:**
        - Çoklu filtre kullanın
        - Trend + RSI kombinasyonu
        - Volume confirmation
        """
        )

# Ana içerik
# Filtreler
filters = create_filter_ui()

# Tarama butonu ve sonuçlar
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    scan_button = st.button(
        "🚀 Taramayı Başlat",
        type="primary",
        use_container_width=True,
        disabled=(len(symbol_list) == 0),
    )

with col2:
    # Session state'ten önceki sonuçları kontrol et
    if "last_scan_results" in st.session_state and st.session_state.last_scan_results:
        if st.button("📋 Önceki Sonuçlar", use_container_width=True):
            scan_button = False  # Yeni tarama yapma
            results = st.session_state.last_scan_results
            filtered_results = apply_filters(results, filters)
            display_results(filtered_results)

with col3:
    # Otomatik yenileme
    auto_refresh = st.checkbox("🔄 Otomatik Yenile", value=False)
    if auto_refresh:
        refresh_interval = st.number_input(
            "Dakika", min_value=5, max_value=60, value=15
        )

# Tarama işlemi
if scan_button and symbol_list:
    st.markdown("---")

    # Performance monitor
    perf_monitor = PerformanceMonitor()

    # Fetcher
    fetcher = RateLimitedBatchFetcher(
        max_retries=2, base_delay=0.5, batch_size=batch_size
    )

    # Veri çekme
    with st.spinner(f"📡 {len(symbol_list)} hisse için veri çekiliyor..."):
        start_time = time.time()

        # Rate limiting uyarısı
        if len(symbol_list) > 50:
            st.warning(
                """
            ⚠️ **Büyük liste algılandı!**
            - Yahoo Finance rate limiting uygulayabilir
            - İşlem biraz uzun sürebilir
            - Sabırlı olun...
            """
            )

        # Batch halinde veri çek
        stock_data = fetcher.fetch_batch_optimized(symbol_list, period=period)

        fetch_time = time.time() - start_time

    # Başarı oranı
    success_rate = len(stock_data) / len(symbol_list) * 100

    if success_rate < 50:
        st.error(
            f"""
        ❌ **Düşük başarı oranı: %{success_rate:.1f}**
        
        **Olası Nedenler:**
        - Yahoo Finance rate limiting
        - İnternet bağlantı sorunları
        - Geçersiz hisse kodları
        
        **Çözüm:**
        - 5-10 dakika bekleyin
        - Daha küçük liste deneyin
        - Batch boyutunu azaltın
        """
        )
    else:
        st.success(
            f"""
        ✅ **Veri çekme tamamlandı!**
        - Başarı oranı: %{success_rate:.1f}
        - Süre: {fetch_time:.1f} saniye
        - Başarılı: {len(stock_data)} / {len(symbol_list)}
        """
        )

    # Teknik analiz
    if stock_data:
        with st.spinner("🧮 Teknik analiz yapılıyor..."):
            results = []

            analysis_progress = st.progress(0)
            analysis_status = st.empty()

            for idx, (symbol, data) in enumerate(stock_data.items()):
                # Progress güncelle
                progress = (idx + 1) / len(stock_data)
                analysis_progress.progress(progress)
                analysis_status.text(f"📊 Analiz ediliyor: {symbol}")

                # Sinyal hesapla
                result = calculate_enhanced_signals(symbol, data)
                if result:
                    results.append(result)
                    perf_monitor.record_processed_symbol()

            analysis_progress.empty()
            analysis_status.empty()

        # Sonuçları kaydet
        st.session_state.last_scan_results = results
        st.session_state.last_scan_time = datetime.now()

        # Filtreleri uygula
        filtered_results = apply_filters(results, filters)

        # Özet
        st.markdown("---")
        st.markdown("### 📊 Tarama Özeti")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Toplam Analiz",
                len(results),
                delta=f"{len(results) - len(filtered_results)} filtrelendi",
            )

        with col2:
            strong_count = len([r for r in filtered_results if r.get("score", 0) >= 4])
            st.metric(
                "Güçlü Sinyaller",
                strong_count,
                delta=(
                    f"%{(strong_count/len(filtered_results)*100):.0f}"
                    if filtered_results
                    else "0%"
                ),
            )

        with col3:
            avg_score = (
                sum(r.get("score", 0) for r in filtered_results) / len(filtered_results)
                if filtered_results
                else 0
            )
            st.metric(
                "Ortalama Skor",
                f"{avg_score:.2f}",
                delta="İyi" if avg_score > 2 else "Zayıf",
            )

        with col4:
            total_time = time.time() - start_time
            st.metric(
                "Toplam Süre",
                f"{total_time:.1f}s",
                delta=f"{len(symbol_list)/total_time:.1f} hisse/s",
            )

        # Sonuçları göster
        st.markdown("---")
        display_results(filtered_results)

        # Performans raporu
        with st.expander("📊 Performans Raporu", expanded=False):
            try:
                report = perf_monitor.get_report()

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("API Çağrısı", report.get("api_calls", 0))
                    st.metric(
                        "Cache Hit Rate", f"%{report.get('cache_hit_rate', 0):.1f}"
                    )

                with col2:
                    st.metric("İşlenen Sembol", len(results))
                    st.metric("Hata Oranı", f"%{report.get('error_rate', 0):.1f}")

                with col3:
                    st.metric(
                        "Ortalama Hız",
                        f"{report.get('symbols_per_second', 0):.2f} sembol/s",
                    )
                    st.metric("Toplam Süre", f"{report.get('elapsed_time', 0):.1f}s")

            except Exception as e:
                st.error(f"Performans raporu hatası: {str(e)}")

            # Başarısız semboller
            if fetcher.failed_symbols:
                st.warning(f"⚠️ {len(fetcher.failed_symbols)} hisse başarısız:")
                failed_str = ", ".join(sorted(fetcher.failed_symbols)[:20])
                if len(fetcher.failed_symbols) > 20:
                    failed_str += f" ... (+{len(fetcher.failed_symbols)-20} daha)"
                st.text(failed_str)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Tyana Otomatik Tarama v2.0 | Premium Edition</p>
        <p style='font-size: 12px;'>Powered by Advanced AI & Real-time Market Analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Otomatik yenileme
if auto_refresh and "last_scan_time" in st.session_state:
    elapsed = (datetime.now() - st.session_state.last_scan_time).seconds / 60
    if elapsed >= refresh_interval:
        st.rerun()
