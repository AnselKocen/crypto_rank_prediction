"""
run_crypto_project_pipeline.py

1. Downloads OHLCV data for top cryptocurrencies using the Coindesk API.
2. Performs data cleaning and feature engineering (volume shock, momentum, volatility, etc).
3. Loads and cleans financial news articles, applies VADER sentiment analysis.
4. Merges sentiment features with crypto market features.
5. Visualizes sentiment trends via fear-greed gauge and word clouds.

Outputs cleaned datasets and visualizations into structured folders under the specified project week.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import requests

from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from typing import Optional
from collections import Counter

from tqdm import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from wordcloud import WordCloud
# ---------------------------------------------------------------------------
# Logging --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Directory helpers ----------------------------------------------------------
def _ensure_dir(root: Path, sub: str | Path) -> Path:
    """Return *root/sub* as Path, creating parents as needed.
        _ensure_dir(Path("week_crypto"), "results")
    """
    path = root / sub
    path.mkdir(parents=True, exist_ok=True)
    return path

def build_week_dirs(
        week_folder: str | Path = "5545project",
        results_folder: str = "stage1-2_results",
        data_sub: str = "clean_data",
        figure_sub: str = "figures"
) -> tuple[Path, Path]:
    """ Return *data_dir* inside <week_folder>/<results_folder>/… """
    project_root = Path.cwd().resolve()
    week_root = project_root if project_root.name == str(week_folder) else _ensure_dir(
        project_root, week_folder
    )
    results_root = _ensure_dir(week_root, results_folder)
    data_dir = _ensure_dir(results_root, data_sub)
    figure_dir = _ensure_dir(results_root, figure_sub)
    return data_dir, figure_dir


# -----------------------------------------------------------------
# 1. Stage 1 - ETL
# -----------------------------------------------------------------
# ----------------------- For crypto data -------------------------
# 1. API helpers
BASE_URL = "https://data-api.coindesk.com"

def _headers(api_key: str) -> dict[str, str]:
    return {"authorization": f"Apikey {api_key}"}

def get_top_coins(
        api_key: str,
        pages: List[int],
        limit: int = 100,
        sort_by: str = "CIRCULATING_MKT_CAP_USD",
) -> List[str]:
    """Return a list of coin symbols across *pages* sorted by *sort_by*."""
    coins: List[str] = []
    for page in pages:
        url = (
            f"{BASE_URL}/asset/v1/top/list?" 
            f"page={page}&page_size={limit}" 
            f"&sort_by={sort_by}&sort_direction=DESC" 
            "&groups=ID,BASIC,MKT_CAP"
        )
        resp = requests.get(url, headers=_headers(api_key), timeout=30)
        data = resp.json()

        if "Data" not in data or "LIST" not in data["Data"]:
            logging.warning("Page %d returned no data: %s", page, data.get("Message"))
            continue

        for coin in data["Data"]["LIST"]:
            coins.append(coin["SYMBOL"])
        logging.info("Collected %d symbols from page %d", len(data["Data"]["LIST"]), page)

    if not coins: # 在报告里，记得强调函数中的防止错误的部分。
        raise RuntimeError("No symbols retrieved. Check API key or parameters.")

    return coins

# 2. GET OHLCV
def get_daily_ohlcv(
        symbol: str,
        api_key: str,
        limit: int = 2000,
        currency: str = "USD",
        max_retries: int = 3,
        wait: float = 1.0,
        verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Download daily OHLCV for *symbol* from the Coindesk Crypto-Compare API.

    Parameters
    ----------
    symbol        : str   – e.g. 'BTC', 'ETH'
    api_key       : str   – your Coindesk/CC API key
    limit         : int   – how many days to return (<= 2000)
    currency      : str   – quote currency (default 'USD')
    max_retries   : int   – times to retry if TIMESTAMP missing or error
    wait          : float – seconds to wait between retries

    Returns
    -------
    pd.DataFrame indexed by ['symbol', 'date'] or None if all retries fail.
    """
    url = (
        f"{BASE_URL}/index/cc/v1/historical/days"
        f"?market=cadli&instrument={symbol}-{currency}"
        f"&limit={limit}&aggregate=1&fill=true&apply_mapping=true"
    )

    for attempt in range(1, max_retries + 1):
        try:
            safe_headers = {k: ('***' if k.lower() == 'authorization' else v)
                            for k, v in _headers(api_key).items()}
            logging.info("REQUEST -> GET %s | hdrs=%s", url, safe_headers)

            resp = requests.get(url, headers=_headers(api_key), timeout=30)

            # ----------------------- VERBOSE DIAGNOSTICS --------------------
            if verbose:
                safe_headers = {k: ("***" if k.lower() == "authorization" else v)
                                for k, v in resp.request.headers.items()}
                logging.info(
                    "[%s] HTTP %s  |  req-hdrs=%s  |  rate-remaining=%s",
                    symbol,
                    resp.status_code,
                    safe_headers,
                    resp.headers.get("x-ratelimit-remaining"),
                )
                logging.debug("[%s] raw-json=%s", symbol, resp.text[:500])
            # ----------------------------------------------------------------

            data = resp.json()

            # API-level error or missing payload
            if data.get("Response") == "Error" or "Data" not in data:
                logging.warning(
                    "No data for %s (attempt %d/%d): %s",
                    symbol,
                    attempt,
                    max_retries,
                    data.get("Message"),
                )
                raise ValueError("API response error")

            # Make sure TIMESTAMP exists; otherwise force retry
            if not data["Data"] or "TIMESTAMP" not in data["Data"][0]:
                logging.warning(
                    "TIMESTAMP missing for %s (attempt %d/%d) – retrying …",
                    symbol,
                    attempt,
                    max_retries,
                )
                raise KeyError("TIMESTAMP")

            # -----------------------------------------------------------------
            # Normal parsing path
            # -----------------------------------------------------------------
            df = pd.DataFrame(data["Data"])
            df["date"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
            df = df.rename(
                columns={
                    "OPEN": "open",
                    "HIGH": "high",
                    "LOW": "low",
                    "CLOSE": "close",
                    "VOLUME": "base_volume",
                    "QUOTE_VOLUME": "usd_volume",
                }
            )
            df = df[
                ["date", "open", "high", "low", "close", "usd_volume", "base_volume"]
            ].copy()

            # Convenience: express USD volume in millions
            df["usd_volume_mil"] = df["usd_volume"] / 1e6
            df["symbol"] = symbol
            df.set_index(["symbol", "date"], inplace=True)

            return df

        except (requests.RequestException, ValueError, KeyError) as exc:
            # Connection problem OR explicit retry trigger
            if attempt < max_retries:
                time.sleep(wait)
                continue
            logging.error("Failed to fetch OHLCV for %s: %s", symbol, exc)

    # All retries exhausted
    return None

# 3. stage1 main function of crypto data
def stage1_etl(
        api_key: str,
        pages: List[int],
        top_limit: int = 100,
        history_limit: int = 2000,
        currency: str = "USD",
        sleep_sec: float = 0.5,
        data_dir: Path | None = None, #
        filename: str = "stage_1_crypto_data.csv",
) -> pd.DataFrame:
    """
    Download OHLCV history for the top coins and return a tidy DataFrame.
    """
    logging.info("Fetching list of top coins …")
    symbols = get_top_coins(api_key, pages, top_limit)
    logging.info("Total symbols collected: %d", len(symbols))

    all_frames: List[pd.DataFrame] = []
    for sym in symbols:
        logging.info("Downloading history for %s", sym)
        df = get_daily_ohlcv(sym, api_key, history_limit, currency)
        if df is not None:
            all_frames.append(df)
        time.sleep(sleep_sec)

    if not all_frames:
        raise RuntimeError("No historical data retrieved.")

    data = pd.concat(all_frames).sort_index()

    if data_dir is not None:
        out_path = data_dir / filename
        data.to_csv(out_path)
        logging.info("Stage 1 CSV written to %s", out_path)

    return data
# ------------------------ For crypto news --------------------------
def fetch_news_range(
    api_key: str | None,
    start_dt: datetime,
    end_dt: datetime,
    lang: str = "EN",
) -> pd.DataFrame:
    """
    Pull CoinDesk news between *start_dt* and *end_dt* (inclusive).
    Logs the query date at each step so you can track progress.
    """
    url = "https://data-api.coindesk.com/news/v1/article/list"
    out: list[pd.DataFrame] = []

    while end_dt > start_dt:
        query_ts  = int(end_dt.timestamp()) #
        query_day = end_dt.strftime("%Y-%m-%d")
        logging.info("Requesting articles up to %s (UTC)", query_day)

        resp = requests.get(f"{url}?lang={lang}&to_ts={query_ts}")
        if not resp.ok:
            logging.error("Request failed with status %s", resp.status_code)
            break

        d = pd.DataFrame(resp.json()["Data"])
        if d.empty:
            logging.info("No data returned for %s – stopping loop.", query_day)
            break

        d["date"] = pd.to_datetime(d["PUBLISHED_ON"], unit="s")
        out.append(d[d["date"] >= start_dt])

        # step backward to the day before the earliest article we just received
        end_dt = datetime.utcfromtimestamp(d["PUBLISHED_ON"].min() - 1)

    news = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    logging.info("Fetched %d articles in total", len(news))

    return news
# 2. fetch news
def stage1_load_news(
    api_key: str | None,
    start_dt: datetime,
    end_dt: datetime,
    data_dir: Path,
    filename: str = "stage_1_news_raw.csv",
) -> pd.DataFrame:
    tic = time.time()
    logging.info("Stage 1 – downloading news …")

    df = fetch_news_range(api_key, start_dt, end_dt)

    # keep columns / rename like original script
    drop_cols = [
        "GUID",
        "PUBLISHED_ON_NS",
        "IMAGE_URL",
        "SUBTITLE",
        "AUTHORS",
        "URL",
        "UPVOTES",
        "DOWNVOTES",
        "SCORE",
        "CREATED_ON",
        "UPDATED_ON",
        "SOURCE_DATA",
        "CATEGORY_DATA",
        "STATUS",
        "SOURCE_ID",
        "TYPE",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df.columns = df.columns.str.lower()
    other = [c for c in df.columns if c not in ["date", "id"]]
    df = df[["date", "id"] + other]

    # POSITIVE field → numeric
    if "sentiment" in df.columns:
        df["positive"] = np.where(df["sentiment"].str.upper() == "POSITIVE", 1, 0)
        df = df.drop(columns="sentiment")
    else:
        df["positive"] = np.nan

    out = data_dir / filename
    df.to_csv(out, index=False)
    logging.info("Saved raw news -> %s (%.2f s)", out.name, time.time() - tic)
    return df


# 3. Clean News Text

_STOP = set(stopwords.words("english"))
_LEM = WordNetLemmatizer()

def _preprocess_text(txt: str) -> str:
    tokens = word_tokenize(str(txt))
    tokens_lower = [t.lower() for t in tokens]
    keep = [t for t in tokens_lower if t.isalpha() and t not in _STOP]
    lemmas = [_LEM.lemmatize(t) for t in keep]
    return " ".join(lemmas)

def _basic_tokens(txt: str) -> list[str]:
    tok = txt.split()
    return [w for w in tok if w.isalpha()]

def stage1_clean_text(
    df_raw: pd.DataFrame,
    data_dir: Path,
    max_common: int = 500,
    filename: str = "stage_1_news_clean.csv",
) -> pd.DataFrame:
    tic = time.time()
    logging.info("Stage 1 – cleaning text …")

    df = df_raw.copy()
    # choose body for analysis; rename to reviewText like original pattern
    df["reviewText"] = df["body"].progress_apply(_preprocess_text)

    # common words
    tokens = " ".join(df["reviewText"])
    counts = Counter(_basic_tokens(tokens)).most_common(max_common)
    pd.DataFrame(counts, columns=["word", "count"]).to_csv(
        data_dir / "stage_1_common_words.csv", index=False
    )

    out = data_dir / filename
    df.to_csv(out, index=False)
    logging.info("Saved cleaned data -> %s (%.2f s)", out.name, time.time() - tic)
    return df



# Stage 2 – feature engineering ---------------------------------------------

# ---------------------------------------------------------------------------
# 1.  crypto data
# ---------------------------------------------------------------------------
def stage2_feature_engineering(
        tidy_prices: pd.DataFrame | None = None,
        csv_path: Path | None = None,
        data_dir: Path | None = None,
        filename: str = "stage_2_crypto_data.csv",
) -> pd.DataFrame:
    """
    Create volume shocks, momentum, volatility, weekly returns, and
    save the cleaned weekly data set.
    """
    if tidy_prices is None:
        if csv_path is None:
            raise ValueError("Provide either tidy_prices or csv_path.")
        logging.info("Reading Stage 1 CSV from %s", csv_path)
        tidy_prices = pd.read_csv(
            csv_path, index_col=["symbol", "date"], parse_dates=["date"]

        )
    # Reset index for easier operations
    df = tidy_prices.reset_index().sort_values(["symbol", "date"]).copy()

    # Volume shocks ---------------------------------------------------------
    for m in [7, 14, 21, 28, 42]:
        rolling_mean = (
            df.groupby("symbol")["usd_volume"]
            .shift(1)
            .rolling(m, min_periods=m)
            .mean()
        )
        df["usd_volume"] = pd.to_numeric(df["usd_volume"], errors="coerce")
        df[f"v_{m}d"] = np.log(df["usd_volume"]) - np.log(rolling_mean)

    # Log returns -----------------------------------------------------------
    df["log_return"] = np.log1p(
        df.groupby("symbol")["close"].pct_change()
    )
    df = df.replace([-np.inf, np.inf], np.nan)  #

    lower = df["log_return"].quantile(0.01)
    upper = df["log_return"].quantile(0.99)

    df["log_return"] = df["log_return"].clip(lower, upper)

    df["return_sign_consistency_3d"] = (
        (df["log_return"] > 0)
        .groupby(df["symbol"])
        .rolling(3, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    candle_body_ratio = (df["close"] - df["open"]).abs() / (df["high"] - df["low"])
    long_candle_flag = candle_body_ratio > 0.7
    df["long_candle_ratio_5d"] = (
        long_candle_flag
        .groupby(df["symbol"])
        .rolling(5, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Momentum and volatility ----------------------------------------------
    for m in [14, 21, 28, 42, 90]:
        shifted = df.groupby("symbol")["log_return"].shift(7)
        df[f"momentum_{m}"] = (
                np.exp(
                    shifted.rolling(m, min_periods=m).sum()
                )
                - 1.0
        )
        df[f"volatility_{m}"] = (
                                    df.groupby("symbol")["log_return"]
                                    .rolling(m, min_periods=m)
                                    .std()
                                    .reset_index(level=0, drop=True)
                                ) * np.sqrt(365.0)

    # Short-term reversal proxy --------------------------------------------
    df["strev_daily"] = df["log_return"]

    df["date"] = pd.to_datetime(df["date"])
    dfw = (
        df.set_index("date")
        .groupby("symbol")
        .resample("W-WED")
        .last()
        .droplevel("symbol")
    )

    dfw["return"] = dfw.groupby("symbol")["close"].pct_change()

    lower = dfw["return"].quantile(0.01)
    upper = dfw["return"].quantile(0.99)
    dfw["return"] = dfw["return"].clip(lower, upper)
    dfw['strev_weekly'] = dfw["return"]

    dfw = dfw.reset_index()

    # ----------------------------------------------------------------------
    stable_tickers = [
        "USD", "USDT", "USDC", "TUSD", "BUSD", "PAX", "USDP", "GUSD",
        "DAI", "SUSD", "USDN", "FRAX", "USDX", "USDJ", "XUSD", "USDD",
        "UST", "USTC",
        "EUR", "EURT", "EURS", "EUROC", "SEUR", "SEUR", "SEUR", "SEUR",
        "AEUR", "EURC", "AGEUR", "PAR", "PAXG", "PYUSD", "USD1", "USDE"
    ]
    wrapped_tickers = [
        "WBTC", "WETH", "WBNB", "WSTETH", "WUSDC", "WUSDT",
        "WCRO", "WFTM", "WTRX", "WCELO", "WFIL", "WGLMR",
        "WXRP", "WLTC", "WSOL", "WADA",
    ]

    tickers_to_drop = {t.upper() for t in stable_tickers + wrapped_tickers}

    # Build masks
    is_exact_drop = dfw["symbol"].str.upper().isin(tickers_to_drop)
    has_usd_substr = dfw["symbol"].str.upper().str.contains("USD", na=False)

    # Keep rows that are **not** flagged by either rule
    dfw = dfw[~(is_exact_drop | has_usd_substr)].copy()

    # Basic cleaning --------------------------------------------------------
    dfw = dfw[dfw["return"] > -1.0]
    dfw = dfw.replace([-np.inf, np.inf], np.nan)

    col_order = [
        "date", "symbol", "return", "open", "high", "low", "close", "usd_volume",
        "base_volume", "v_7d",
        "v_14d", "v_21d", "v_28d", "v_42d", "momentum_14",
        "volatility_14", "momentum_21", "volatility_21", "momentum_28",
        "volatility_28", "momentum_42", "volatility_42", "momentum_90",
        "volatility_90", "return_sign_consistency_3d", "long_candle_ratio_5d",
        "strev_daily", "strev_weekly"
    ]
    dfw = dfw[[c for c in col_order if c in dfw.columns]].copy()
    dfw = dfw.fillna(0)

    # Save ------------------------------------------------------------------
    if data_dir is not None:
        out_path = data_dir / filename
        dfw.to_csv(out_path, index=False)
        logging.info("Stage 2 CSV written to %s", out_path)

    return dfw




# ---------------------------------------------------------------------------
# 2. VADER sentiment
# ---------------------------------------------------------------------------

crypto_lexicon = {
    # ─── General market tone ─────────────────────────────────────────────
    "bullish": 3.2,
    "bearish": -3.2,
    "rally": 2.4,
    "selloff": -2.6,
    "soar": 3.0,
    "plummet": -3.0,
    "skyrocket": 3.5,
    "tank": -2.8,
    "breakout": 2.6,
    "breakdown": -2.6,
    "recovery": 2.3,
    "capitulation": -3.3,
    "moon": 3.8,
    "moonshot": 3.5,
    "dip": -0.7,
    "buy_the_dip": 2.5,
    "crash": -3.2,
    "correction": -1.2,
    "bubble": -2.4,
    "dead_cat_bounce": -2.5,
    "all_time_high": 3.6,
    "all_time_low": -3.6,
    "bull_run": 3.4,
    "market_meltdown": -3.8,
    "flash_crash": -3.5,
    "volatility_spike": -1.5,
    "risk_on": 1.4,
    "risk_off": -1.4,
    "safe_haven": 1.1,
    # ─── Trader slang / emotions ────────────────────────────────────────
    "fomo": -0.8,
    "bagholder": -2.0,
    "whale": 1.0,
    "hodl": 2.1,
    "hodling": 2.0,
    "fear": -2.1,
    "greed": 1.8,
    "panic_sell": -3.0,
    "short_squeeze": 2.0,
    "pump": 1.8,
    "dump": -2.6,
    "pump_and_dump": -3.6,
    "rugpull": -3.5,
    # ─── Corporate / earnings language ───────────────────────────────────
    "earnings_beat": 2.4,
    "earnings_miss": -2.4,
    "guidance_raise": 2.2,
    "guidance_cut": -2.2,
    "profit_take": 1.5,
    "profit_warning": -2.6,
    "dividend_hike": 2.3,
    "dividend_cut": -2.3,
    "share_buyback": 1.8,
    "share_dilution": -1.8,
    "upgrade": 2.1,
    "downgrade": -2.1,
    "rating_boost": 1.8,
    "rating_cut": -1.8,
    # ─── Macro & policy terms ───────────────────────────────────────────
    "quantitative_easing": 0.5,
    "quantitative_tightening": -1.2,
    "interest_rate_hike": -1.3,
    "interest_rate_cut": 1.3,
    "inflation_surge": -2.4,
    "deflation": -1.0,
    "stagflation": -3.0,
    "recession": -3.4,
    "depression": -3.8,
    "soft_landing": 1.5,
    "hard_landing": -2.5,
    "economic_expansion": 2.4,
    "stimulus": 1.2,
    "credit_crunch": -3.1,
    "yield_curve_inversion": -2.7,
    # ─── Balance-sheet / distress ───────────────────────────────────────
    "default": -3.5,
    "bankruptcy": -4.0,
    "chapter_11": -3.6,
    "insolvency": -3.6,
    "liquidation": -3.0,
    "margin_call": -2.8,
    "asset_write_down": -2.5,
    "leverage_buyout": 0.2,
    # ─── Crypto-specific positive ───────────────────────────────────────
    "halving": 1.4,
    "hashrate_record": 2.3,
    "mainnet_launch": 2.5,
    "whitepaper_release": 1.6,
    "etf_approval": 2.8,
    "token_listed": 2.0,
    "layer2_scaling": 1.1,
    "gas_fee_drop": 1.7,
    "airdrop": 1.3,
    "alt_season": 2.0,
    "nft_boom": 1.9,
    "metaverse_boost": 2.0,
    "token_burn": 1.5,
    "yield_farming": 1.2,
    "liquidity_mining": 1.0,
    "stablecoin_recovery": 2.1,
    # ─── Crypto-specific negative ───────────────────────────────────────
    "crypto_winter": -3.3,
    "depeg": -3.1,
    "hack": -3.4,
    "exploit": -2.8,
    "bridge_exploit": -3.2,
    "smart_contract_bug": -2.5,
    "exit_scam": -4.0,
    "treasury_drain": -2.7,
    "gas_fee_spike": -1.7,
    "testnet_delay": -1.4,
    "token_delisted": -2.0,
    "pow_ban": -2.3,
    "regulatory_crackdown": -2.2,
    "etf_rejection": -2.8,
    "cease_and_desist": -2.8,
    "fraudulent": -4.0,
    "securities_violation": -3.0,
    "class_action": -2.9,
    "hard_fork": -0.2,
    "fork": 0.0,
    "soft_fork": 0.2,
    "ordinals_collapse": -1.5,
    "nft_crash": -1.9,
    "impermanent_loss": -1.6,
    "validator_slash": -2.0,
    # ─── Neutral / mild, but useful context ─────────────────────────────
    "contango": 0.1,
    "backwardation": -0.1,
    "open_interest_surge": 0.9,
    "funding_rate_flip": 0.3,
    "market_depth": 0.4,
    "thin_liquidity": -1.1,
    "governance_vote": 0.5,
    "license_grant": 2.2,
    "settlement_reached": 1.0,
    "bailout": -0.8,
    "flight_to_quality": 0.7,
    "esg_compliance": 1.0,
    # ── market / price action ──
    "onboarding": -0.4,         # onboarding paused → mildly negative :contentReference[oaicite:0]{index=0}
    "offboarded": -2.0,         # lost banking access → negative :contentReference[oaicite:1]{index=1}
    "listing": 1.5,             # exchange listing → positive :contentReference[oaicite:2]{index=2}
    "delisting": -1.5,
    "gainer": 2.2,              # top daily gainer :contentReference[oaicite:3]{index=3}
    "resistance": -0.8,         # price stuck below resistance :contentReference[oaicite:4]{index=4}
    "support_level": 0.8,
    "leg_up": 1.3,
    "momentum": 1.4,            # “building momentum” :contentReference[oaicite:5]{index=5}
    "turbulence": -1.5,
    "liquidations": -1.9,       # large liquidations :contentReference[oaicite:6]{index=6}
    "retrace": -0.5,
    "retest": 0.6,
    "parabolic": 3.0,
    "melt_up": 2.8,
    "freefall": -3.0,
    "oversold": 1.0,
    "overbought": -1.1,
    "grind_higher": 1.3,
    "slump": -2.0,
    "bounce": 1.3,
    "headwinds": -1.2,
    "tailwinds": 1.2,
    # ── trader / crypto slang ──
    "ape_in": 1.6,
    "diamond_hands": 2.4,
    "paper_hands": -1.8,
    "rekt": -3.2,
    "degen": -1.0,
    "flippening": 1.7,
    # ── risk & attack vectors ──
    "front_run": -2.2,
    "sandwich_attack": -2.6,
    "oracle_failure": -2.9,
    "slashing": -2.0,
    "unstake": -0.6,
    # ── DeFi / staking ──
    "staking_reward": 1.7,
    "overleveraged": -2.3,
    "hashwar": -2.5,
    "ghost_chain": -2.4,
    # ── corporate / deal flow ──
    "pivot": 0.7,
    "windfall": 3.0,
    "oversubscribed": 2.0,
    "shortfall": -2.1,
    # ── macro / policy ──
    "hawkish": -0.9,
    "dovish": 0.9,
    "taper": -0.7,
    "fiscal_cliff": -2.5,
    # ── compliance / regulation ──
    "whitelist": 1.2,
    "blacklist": -1.2,
    # ── gas & fees ──
    "gasless": 1.5,
    "peg_restore": 2.0
}

_VADER = SentimentIntensityAnalyzer()
_VADER.lexicon.update(crypto_lexicon)

# 单文本打分函数
def _vader_scores(txt: str) -> pd.Series:
    return pd.Series(_VADER.polarity_scores(txt))

def stage2_sentiment(
    df_clean: pd.DataFrame,
    data_dir: Path,
    fig_dir:Path,
    filename: str = "stage_2_news_sentiment.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tic = time.time()
    logging.info("Stage 3 – running VADER …")

    df = df_clean.copy()
    df[["neg", "neu", "pos", "compound"]] = df["reviewText"].progress_apply(
        _vader_scores
    )
    # | reviewText     | neg  | neu  | pos  | compound |
    # | -------------- | ---- | ---- | ---- | -------- |
    # | btc price jump | 0.00 | 0.52 | 0.48 | 0.65     |
    df["sentiment"] = (df["compound"] > 0.05).astype(int)
    df["date"] = pd.to_datetime(df["date"])
    out = data_dir / filename
    df.to_csv(out, index=False)

    df = df.set_index("date").sort_index()
    plt.figure(figsize=(8, 4))
    plt.hist(
        df[df["compound"] >= 0.05]["compound"],
        bins=50,
        alpha=0.7,
        density=True,
        color="green",
        label="compound ≥ 0.05",
    )
    plt.hist(
        df[df["compound"] < 0.05]["compound"],
        bins=50,
        alpha=0.7,
        density=True,
        color="red",
        label="compound < 0.05",
    )
    plt.title("Distribution of VADER Compound Scores")
    plt.xlabel("Compound score")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(fig_dir / "compound_distribution.png", bbox_inches="tight")
    plt.close()
    logging.info("Plots saved to %s (%.2f s)", fig_dir, time.time() - tic)

    q_lo, q_hi = df["compound"].quantile([0.10, 0.90])
    logging.info(f"Using dynamic thresholds: q10={q_lo:.2f}, q90={q_hi:.2f}")

    df["extremely_positive"] = (df["compound"] > q_hi).astype(int)
    df["extremely_negative"] = (df["compound"] < q_lo).astype(int)

    compound_weekly = (
        df["compound"]
        .rolling("7D")
        .mean()
        .resample("W-WED")
        .last()
        .rename("compound_weekly")
    )

    weekly_ext_pos_ratio = (
        df["extremely_positive"]
        .resample("W-WED")
        .mean()
        .rename("extremely_positive_ratio")
    )

    weekly_ext_neg_ratio = (
        df["extremely_negative"]
        .resample("W-WED")
        .mean()
        .rename("extremely_negative_ratio")
    )

    compound_weekly_df = pd.concat(
        [compound_weekly, weekly_ext_pos_ratio, weekly_ext_neg_ratio], axis=1
    ).reset_index()

    weekly_path = data_dir / "stage_2_sentiment_weekly.csv"
    compound_weekly_df.to_csv(weekly_path, index=False)
    logging.info("Saved sentiment data -> %s (%.2f s)", out.name, time.time() - tic)

    return df, compound_weekly


def generate_sentiment_wordclouds(df: pd.DataFrame, fig_dir: Path, end_day: datetime) -> None:
    sid = SentimentIntensityAnalyzer()
    sid.lexicon.update(crypto_lexicon)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["body"])

    end_day = pd.to_datetime(end_day)
    start_day = end_day - timedelta(days=7)
    recent_df = df[(df["date"] >= start_day) & (df["date"] <= end_day)]

    all_text = " ".join(recent_df["body"].astype(str).tolist())
    stop_words = set(stopwords.words("english"))
    tokens = [w.lower() for w in word_tokenize(all_text) if w.isalpha() and w.lower() not in stop_words]

    pos_tokens = [w for w in tokens if sid.polarity_scores(w)["compound"] >= 0.2]
    neg_tokens = [w for w in tokens if sid.polarity_scores(w)["compound"] <= -0.1]

    wc = WordCloud(width=1200, height=800, background_color="white", max_words=500)

    for tag, words in [("positive", pos_tokens), ("negative", neg_tokens)]:
        if words:
            wc.generate(" ".join(words)).to_file(fig_dir / f"wordcloud_{tag}.png")
            print(f"✅ {tag} wordcloud saved to {fig_dir / f'wordcloud_{tag}.png'}")
        else:
            continue


def merge_sentiment_feature(
    df_features: pd.DataFrame,
    df_sentiment: pd.DataFrame,
    save_path: str | Path
) -> pd.DataFrame:
    df_features = df_features.copy()
    df_features["date"] = pd.to_datetime(df_features["date"])
    df_sentiment = df_sentiment.copy()
    df_sentiment["date"] = pd.to_datetime(df_sentiment["date"])

    sentiment_cols = ["date", "compound_weekly", "extremely_positive_ratio", "extremely_negative_ratio"]

    df_merged = pd.merge(
        df_features,
        df_sentiment[sentiment_cols],
        on="date",
        how="left"
    )

    df_merged[["compound_weekly", "extremely_positive_ratio", "extremely_negative_ratio"]] = df_merged[
        ["compound_weekly", "extremely_positive_ratio", "extremely_negative_ratio"]
    ].fillna(0)

    df_merged.to_csv(Path(save_path) / "stage_2_merged.csv", index=False)

    return df_merged


def draw_fear_greed_gauge_from_latest(df: pd.DataFrame, fname: Path) -> None:
    latest_value = df["compound_weekly"].iloc[-1]
    score = (latest_value + 1.0) * 50.0

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection="polar"))
    colors = ["#8B0000", "#FF4500", "#FFD700", "#90EE90", "#006400"]
    bounds = [0, 20, 40, 60, 80, 100]

    for i in range(5):
        t0, t1 = np.pi * (bounds[i] / 100), np.pi * (bounds[i + 1] / 100)
        ax.fill_between(np.linspace(t0, t1, 20), 0.5, 1, color=colors[i], alpha=0.8)

    for sc in [0, 25, 50, 75, 100]:
        ang = np.pi * (sc / 100)
        ax.plot([ang, ang], [0.5, 0.55], "k-", lw=1)
        ax.text(ang, 0.6, f"{sc}", ha="center", va="center", fontsize=10)

    needle = np.pi * (score / 100)
    ax.plot([needle, needle], [0, 0.9], "k-", lw=8)
    ax.plot(needle, 0, "ko", ms=15)

    cat, col = _cat(score)
    ax.text(np.pi / 2, 0.2, f"{score:.0f}", ha="center", va="center", fontsize=60, weight="bold")
    plt.figtext(0.5, 0.15, "Latest Weekly Average", ha="center", fontsize=13)
    plt.figtext(0.5, 0.10, f"Current Status: {cat}", ha="center", fontsize=15, weight="bold", color=col)

    ax.set_ylim(0, 1.3)
    ax.set_xlim(0, np.pi)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(1)
    ax.grid(False)
    ax.set_rticks([])
    ax.set_thetagrids([])
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def _cat(v: float) -> tuple[str, str]:
    if v < 20: return "Extreme Fear", "#8B0000"
    if v < 40: return "Fear", "#FF4500"
    if v < 60: return "Neutral", "#FFD700"
    if v < 80: return "Greed", "#90EE90"
    return "Extreme Greed", "#006400"

################## finished #################

