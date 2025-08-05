"""
Execute crypto_project_pipeline.py
Works unchanged on macOS, Windows, or Linux.
"""
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from crypto_project_pipeline import (
    build_week_dirs,
    stage1_etl,
    stage1_load_news,
    stage1_clean_text,
    stage2_feature_engineering,
    stage2_sentiment,
    merge_sentiment_feature,
    draw_fear_greed_gauge_from_latest,
    generate_sentiment_wordclouds
)

# User-adjustable inputs
# ---------------------------------------------------
WEEK_FOLDER = "5545project"
API_KEY = ""                            # Coindesk API key
PAGES = [1,2]                           # which pages of the top-list to pull
TOP_LIMIT = 100                         # coins per page
HISTORY_LIMIT = 2000                    # days of history per coin
CURRENCY = "USD"                        # quote currency
# ---------------------------------------------------
END_DT = datetime.today()
START_DT = END_DT - timedelta(days=HISTORY_LIMIT)
BASE_DIR = Path(__file__).resolve().parent # current working directory
# ---------------------------------------------------------------------------
def main() -> None:
    """Run Stage 1 then Stage 2 with the constants above."""
    data_dir, figure_dir = build_week_dirs(WEEK_FOLDER)

    df_prices = stage1_etl(
        api_key=API_KEY,
        pages=PAGES,
        top_limit=TOP_LIMIT,
        history_limit=HISTORY_LIMIT,
        currency=CURRENCY,
        data_dir=data_dir,
    )

    stage2_feature_engineering(
        tidy_prices=df_prices,
        data_dir=data_dir,
    )

    raw = stage1_load_news(API_KEY, START_DT, END_DT, data_dir)

    clean = stage1_clean_text(raw, data_dir)

    sent = stage2_sentiment(clean, data_dir,figure_dir)

    # load df needed
    df_daily_news = pd.read_csv(data_dir / "stage_2_news_sentiment.csv")
    df_features = pd.read_csv(data_dir / "stage_2_crypto_data.csv")
    df_sentiment = pd.read_csv(data_dir / "stage_2_sentiment_weekly.csv")


    # merge table
    df_merged = merge_sentiment_feature(df_features, df_sentiment, save_path=data_dir)

    # plot
    draw_fear_greed_gauge_from_latest(df_sentiment, figure_dir / "fear_greed_gauge.png")
    generate_sentiment_wordclouds(df_daily_news,figure_dir,END_DT)

    print("Done!")
    print("Data ->", data_dir.resolve())



if __name__ == "__main__":  # required on Windows
    main()