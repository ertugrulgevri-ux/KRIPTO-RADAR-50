
import os
import time
import requests
import numpy as np
import pandas as pd

# =====================================================
# KRIPTO RADAR 50
# Binance Spot USDT | 50 Indicator | 5 Timeframes
# Automatic trading: DISABLED
# =====================================================

BASE_URL = "https://api.binance.com"

TIMEFRAMES = {
    "5m": 0.10,
    "15m": 0.20,
    "1h": 0.30,
    "4h": 0.25,
    "1d": 0.15
}

TOP_COINS = 50
INDICATOR_COUNT = 50
CANDLE_LIMIT = 250
MIN_CANDLES = 220
REQUEST_TIMEOUT = 20

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

session = requests.Session()
session.headers.update({"User-Agent": "KRIPTO-RADAR-50/1.0"})


# =====================================================
# 1. BINANCE API
# =====================================================

def api_get(path, params=None):

    url = BASE_URL + path

    for attempt in range(4):
        try:
            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code in (418, 429):
                wait = int(response.headers.get("Retry-After", 10))
                time.sleep(max(wait, 5))
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException:
            if attempt == 3:
                raise

            time.sleep(2 ** attempt)

    raise RuntimeError("Binance API bağlantısı başarısız.")


# =====================================================
# 2. COIN LISTESI
# =====================================================

def get_symbols():

    exchange = api_get("/api/v3/exchangeInfo")
    tickers = api_get("/api/v3/ticker/24hr")

    valid_symbols = set()

    for item in exchange["symbols"]:

        if item["status"] != "TRADING":
            continue

        if item["quoteAsset"] != "USDT":
            continue

        if not item.get("isSpotTradingAllowed", False):
            continue

        valid_symbols.add(item["symbol"])

    excluded = {
        "USDC", "FDUSD", "TUSD", "USDP",
        "BUSD", "DAI", "USDE", "USDS",
        "EUR", "TRY", "BRL"
    }

    ranked = []

    for item in tickers:

        symbol = item["symbol"]

        if symbol not in valid_symbols:
            continue

        base = symbol[:-4]

        if base in excluded:
            continue

        if base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue

        try:
            volume = float(item["quoteVolume"])
            price = float(item["lastPrice"])

            if volume <= 0 or price <= 0:
                continue

            ranked.append((symbol, volume))

        except (ValueError, TypeError):
            continue

    ranked.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in ranked[:TOP_COINS]]


# =====================================================
# 3. MUM VERILERI
# =====================================================

def get_candles(symbol, interval):

    data = api_get(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": CANDLE_LIMIT
        }
    )

    columns = [
        "time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(data, columns=columns)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Son mum tamamlanmadan sinyal üretme.
    df = df.iloc[:-1].copy()

    df.dropna(
        subset=["open", "high", "low", "close", "volume"],
        inplace=True
    )

    df.reset_index(drop=True, inplace=True)

    if len(df) < MIN_CANDLES:
        return None

    return df


# =====================================================
# 4. YARDIMCI HESAPLAMALAR
# =====================================================

def safe_rsi(close, period):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    result = result.mask(
        (avg_loss == 0) & (avg_gain > 0), 100
    )

    result = result.mask(
        (avg_gain == 0) & (avg_loss > 0), 0
    )

    result = result.mask(
        (avg_gain == 0) & (avg_loss == 0), 50
    )

    return result


def calculate_indicators(df):

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    volume = df["volume"]

    # -------------------------
    # EMA
    # -------------------------

    ema = {
        n: c.ewm(span=n, adjust=False).mean()
        for n in [9, 12, 13, 20, 21, 26, 50, 100, 200]
    }

    # -------------------------
    # SMA
    # -------------------------

    sma = {
        n: c.rolling(n).mean()
        for n in [20, 50, 100, 200]
    }

    # -------------------------
    # RSI
    # -------------------------

    rsi7 = safe_rsi(c, 7)
    rsi14 = safe_rsi(c, 14)
    rsi21 = safe_rsi(c, 21)

    # -------------------------
    # MACD
    # -------------------------

    macd = ema[12] - ema[26]

    macd_signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    macd_hist = macd - macd_signal

    # -------------------------
    # STOCHASTIC
    # -------------------------

    lowest = l.rolling(14).min()
    highest = h.rolling(14).max()

    stoch_k = (
        100 * (c - lowest)
        / (highest - lowest).replace(0, np.nan)
    )

    stoch_d = stoch_k.rolling(3).mean()

    # -------------------------
    # STOCH RSI
    # -------------------------

    rsi_low = rsi14.rolling(14).min()
    rsi_high = rsi14.rolling(14).max()

    stoch_rsi = (
        100 * (rsi14 - rsi_low)
        / (rsi_high - rsi_low).replace(0, np.nan)
    )

    stoch_rsi_d = stoch_rsi.rolling(3).mean()

    # -------------------------
    # ATR / ADX / DI
    # -------------------------

    tr = pd.concat(
        [
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0), 0
    )

    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0), 0
    )

    plus_di = (
        100 * plus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100 * minus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        / atr.replace(0, np.nan)
    )

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    adx = dx.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # -------------------------
    # CCI
    # -------------------------

    typical = (h + l + c) / 3

    tp_mean = typical.rolling(20).mean()

    mean_dev = typical.rolling(20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True
    )

    cci = (
        (typical - tp_mean)
        / (0.015 * mean_dev).replace(0, np.nan)
    )

    # -------------------------
    # WILLIAMS %R
    # -------------------------

    williams = (
        -100 * (highest - c)
        / (highest - lowest).replace(0, np.nan)
    )

    # -------------------------
    # ROC
    # -------------------------

    roc5 = c.pct_change(5) * 100
    roc10 = c.pct_change(10) * 100
    roc20 = c.pct_change(20) * 100

    # -------------------------
    # BOLLINGER BANDS
    # -------------------------

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()

    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    bb_percent = (
        (c - bb_lower)
        / (bb_upper - bb_lower).replace(0, np.nan)
    )

    bb_width = (
        (bb_upper - bb_lower)
        / bb_mid.replace(0, np.nan)
    ) * 100

    # -------------------------
    # KELTNER CHANNEL
    # -------------------------

    kc_mid = c.ewm(
        span=20,
        adjust=False
    ).mean()

    kc_upper = kc_mid + 2 * atr
    kc_lower = kc_mid - 2 * atr

    # -------------------------
    # DONCHIAN CHANNEL
    # -------------------------

    don_high = h.rolling(20).max()
    don_low = l.rolling(20).min()

    # -------------------------
    # ICHIMOKU
    # -------------------------

    tenkan = (
        h.rolling(9).max()
        + l.rolling(9).min()
    ) / 2

    kijun = (
        h.rolling(26).max()
        + l.rolling(26).min()
    ) / 2

    span_a = (tenkan + kijun) / 2

    span_b = (
        h.rolling(52).max()
        + l.rolling(52).min()
    ) / 2

    # -------------------------
    # OBV
    # -------------------------

    direction = np.sign(c.diff()).fillna(0)

    obv = (direction * volume).cumsum()

    obv_ema = obv.ewm(
        span=20,
        adjust=False
    ).mean()

    # -------------------------
    # ADL
    # -------------------------

    money_multiplier = (
        ((c - l) - (h - c))
        / (h - l).replace(0, np.nan)
    )

    money_volume = money_multiplier * volume

    adl = money_volume.fillna(0).cumsum()

    # -------------------------
    # CMF
    # -------------------------

    cmf = (
        money_volume.rolling(20).sum()
        / volume.rolling(20).sum().replace(0, np.nan)
    )

    # -------------------------
    # MFI
    # -------------------------

    raw_money_flow = typical * volume

    positive_flow = raw_money_flow.where(
        typical > typical.shift(), 0
    ).rolling(14).sum()

    negative_flow = raw_money_flow.where(
        typical < typical.shift(), 0
    ).rolling(14).sum()

    money_ratio = (
        positive_flow
        / negative_flow.replace(0, np.nan)
    )

    mfi = 100 - (100 / (1 + money_ratio))

    mfi = mfi.mask(
        (negative_flow == 0) & (positive_flow > 0), 100
    )

    mfi = mfi.mask(
        (positive_flow == 0) & (negative_flow > 0), 0
    )

    mfi = mfi.mask(
        (positive_flow == 0) & (negative_flow == 0), 50
    )

    # -------------------------
    # VWAP (20 PERIOD)
    # -------------------------

    vwap = (
        (typical * volume).rolling(20).sum()
        / volume.rolling(20).sum().replace(0, np.nan)
    )

    # -------------------------
    # FORCE INDEX
    # -------------------------

    force = (
        c.diff() * volume
    ).ewm(span=13, adjust=False).mean()

    # -------------------------
    # VPT
    # -------------------------

    vpt = (
        volume * c.pct_change()
    ).fillna(0).cumsum()

    # -------------------------
    # VOLUME RATIO
    # -------------------------

    volume_avg = volume.rolling(20).mean()

    volume_ratio = (
        volume / volume_avg.replace(0, np.nan)
    )

    # -------------------------
    # MOMENTUM
    # -------------------------

    momentum10 = c - c.shift(10)

    # -------------------------
    # TRIX
    # -------------------------

    trix_ema = c.ewm(span=15, adjust=False).mean()
    trix_ema = trix_ema.ewm(span=15, adjust=False).mean()
    trix_ema = trix_ema.ewm(span=15, adjust=False).mean()

    trix = trix_ema.pct_change() * 100

    # -------------------------
    # CURRENT VALUES
    # -------------------------

    p = c.iloc[-1]

    # =================================================
    # 50 TECHNICAL CHECKS
    # =================================================

    checks = [

        # EMA 1-7
        p > ema[9].iloc[-1],
        p > ema[12].iloc[-1],
        p > ema[20].iloc[-1],
        p > ema[21].iloc[-1],
        p > ema[50].iloc[-1],
        p > ema[100].iloc[-1],
        p > ema[200].iloc[-1],

        # EMA CROSS 8-10
        ema[9].iloc[-1] > ema[20].iloc[-1],
        ema[20].iloc[-1] > ema[50].iloc[-1],
        ema[50].iloc[-1] > ema[200].iloc[-1],

        # SMA 11-14
        p > sma[20].iloc[-1],
        p > sma[50].iloc[-1],
        p > sma[200].iloc[-1],
        sma[20].iloc[-1] > sma[50].iloc[-1],

        # RSI 15-18
        rsi7.iloc[-1] > 50,
        50 < rsi14.iloc[-1] < 70,
        rsi21.iloc[-1] > 50,
        rsi14.iloc[-1] > rsi14.iloc[-2],

        # MACD 19-21
        macd.iloc[-1] > macd_signal.iloc[-1],
        macd_hist.iloc[-1] > 0,
        macd_hist.iloc[-1] > macd_hist.iloc[-2],

        # STOCHASTIC 22-23
        stoch_k.iloc[-1] > stoch_d.iloc[-1],
        stoch_k.iloc[-1] > 50,

        # STOCH RSI 24-25
        stoch_rsi.iloc[-1] > stoch_rsi_d.iloc[-1],
        stoch_rsi.iloc[-1] > 50,

        # ADX / DI 26-27
        adx.iloc[-1] > 20,
        plus_di.iloc[-1] > minus_di.iloc[-1],

        # CCI 28
        cci.iloc[-1] > 0,

        # WILLIAMS 29
        williams.iloc[-1] > -50,

        # ROC 30-32
        roc5.iloc[-1] > 0,
        roc10.iloc[-1] > 0,
        roc20.iloc[-1] > 0,

        # MOMENTUM 33
        momentum10.iloc[-1] > 0,

        # BOLLINGER 34-35
        bb_percent.iloc[-1] > 0.5,
        bb_width.iloc[-1] > bb_width.iloc[-2],

        # KELTNER 36
        p > kc_mid.iloc[-1],

        # DONCHIAN 37
        p > don_high.iloc[-2],

        # ICHIMOKU 38-40
        tenkan.iloc[-1] > kijun.iloc[-1],
        p > span_a.iloc[-1],
        span_a.iloc[-1] > span_b.iloc[-1],

        # OBV 41-42
        obv.iloc[-1] > obv_ema.iloc[-1],
        obv.iloc[-1] > obv.iloc[-2],

        # CMF 43
        cmf.iloc[-1] > 0,

        # MFI 44
        mfi.iloc[-1] > 50,

        # VWAP 45
        p > vwap.iloc[-1],

        # FORCE INDEX 46
        force.iloc[-1] > 0,

        # ADL 47
        adl.iloc[-1] > adl.iloc[-2],

        # VPT 48
        vpt.iloc[-1] > vpt.iloc[-2],

        # VOLUME RATIO 49
        volume_ratio.iloc[-1] > 1,

        # TRIX 50
        trix.iloc[-1] > 0
    ]

    if len(checks) != INDICATOR_COUNT:
        raise ValueError("50 gösterge kontrolü tamamlanmadı!")

    # Eksik veya hesaplanamayan gösterge varsa
    # bu zaman dilimini geçerli puan olarak kabul etme.
    if not all(
        pd.notna(x)
        for x in checks
    ):
        return None

    positive = sum(bool(x) for x in checks)

    score = (positive / INDICATOR_COUNT) * 100

    return round(score, 2), positive


# =====================================================
# 5. TELEGRAM
# =====================================================

def send_telegram(message):

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram bilgileri henüz ayarlanmamış.")
        return False

    url = (
        "https://api.telegram.org/bot"
        + TELEGRAM_TOKEN
        + "/sendMessage"
    )

    try:

        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message
            },
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        return True

    except requests.RequestException as error:

        print("Telegram hatası:", error)
        return False


# =====================================================
# 6. COIN TARAMA
# =====================================================

def scan_coin(symbol):

    weighted_score = 0
    details = {}

    for timeframe, weight in TIMEFRAMES.items():

        try:

            df = get_candles(symbol, timeframe)

            if df is None:
                return None

            result = calculate_indicators(df)

            if result is None:
                return None

            score, positive = result

            details[timeframe] = score

            weighted_score += score * weight

            time.sleep(0.10)

        except Exception as error:

            print(
                symbol,
                timeframe,
                "HATA:",
                error
            )

            return None

    return {
        "symbol": symbol,
        "score": round(weighted_score, 2),
        "details": details
    }


# =====================================================
# 7. RAPOR
# =====================================================

def create_report(results):

    lines = [
        "🚀 KRİPTO RADAR 50",
        "Binance Spot USDT",
        "50 GÖSTERGE / 5 ZAMAN DİLİMİ",
        "250 KONTROL / COIN",
        "",
        "ZAMAN AĞIRLIKLARI:",
        "5m %10 | 15m %20 | 1h %30",
        "4h %25 | 1d %15",
        "",
        "EN YÜKSEK PUANLI COINLER:",
        ""
    ]

    for item in results[:15]:

        d = item["details"]

        lines.append(
            f"{item['symbol']} : {item['score']}/100\n"
            f"5m: {d['5m']} | "
            f"15m: {d['15m']} | "
            f"1h: {d['1h']} | "
            f"4h: {d['4h']} | "
            f"1d: {d['1d']}\n"
        )

    lines.append(
        "Not: Puanlar teknik gösterge uyumudur; "
        "yatırım tavsiyesi veya kesin alım sinyali değildir."
    )

    return "\n".join(lines)


# =====================================================
# 8. ANA PROGRAM
# =====================================================

def main():

    print("=" * 45)
    print("KRİPTO RADAR 50 BAŞLIYOR")
    print("=" * 45)

    try:

        symbols = get_symbols()

    except Exception as error:

        print("Coin listesi alınamadı:", error)
        return

    print("Taranacak coin sayısı:", len(symbols))

    results = []

    for index, symbol in enumerate(symbols, start=1):

        print(
            f"[{index}/{len(symbols)}] "
            f"{symbol} taranıyor..."
        )

        result = scan_coin(symbol)

        if result is not None:

            results.append(result)

            print(
                symbol,
                "PUAN:",
                result["score"]
            )

        else:

            print(
                symbol,
                "Eksik veri veya hesaplama hatası."
            )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    report = create_report(results)

    print(report)

    send_telegram(report)

    print("=" * 45)
    print("TARAMA TAMAMLANDI")
    print("Geçerli coin:", len(results))
    print("=" * 45)


if __name__ == "__main__":
    main()
