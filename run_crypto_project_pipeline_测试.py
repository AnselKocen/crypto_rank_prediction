"""
Execute Stages 1 and 2 of the crypto data pipeline.
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

# ---------------------------------------------------------------------------
# User-adjustable inputs -----------------------------------------------------
# PLEASE MAKE SURE THAT stage_1_news_raw ( the file contain news to 2023 download from ed) is under cwd!!!
WEEK_FOLDER = "5545project"
API_KEY = ""          # required - Coindesk API 密钥，必须填写才能访问数据
PAGES = [1,2]                            # which pages of the top-list to pull - 指定抓取哪些页的币种榜单，例如 [1] 表示第 1 页
TOP_LIMIT = 100                         # coins per page - 每页获取多少币种，默认最多是 100
HISTORY_LIMIT = 2000                   # days of history per coin - 每个币最多下载多少天的数据（默认 2000 天）
CURRENCY = "USD"                       # quote currency - 计价币种，通常是 USD


# ---------------------------------------------------
END_DT = datetime.today()
START_DT = END_DT - timedelta(days=HISTORY_LIMIT)
BASE_DIR  = "."                 # current working directory


# ---------------------------------------------------------------------------


def main() -> None:
    """Run Stage 1 then Stage 2 with the constants above."""
    data_dir, figure_dir = build_week_dirs(WEEK_FOLDER)# 调用 build_week_dirs()，创建数据输出文件夹结构：./week5_crypto/results/clean_data/
    raw = pd.read_csv(data_dir / "stage_1_news_raw.csv")# 测试用的
    clean = stage1_clean_text(raw, data_dir)

    sent = stage2_sentiment(clean, data_dir,figure_dir)

    # 加载DataFrame

    df_daily_news = pd.read_csv(data_dir / "stage_2_news_sentiment.csv")
    df_features = pd.read_csv(data_dir / "stage_2_crypto_data.csv")
    df_sentiment = pd.read_csv(data_dir / "stage_2_sentiment_weekly.csv")


    # 调用合并函数
    df_merged = merge_sentiment_feature(df_features, df_sentiment, save_path=data_dir)

    # 仪表盘
    draw_fear_greed_gauge_from_latest(df_sentiment, figure_dir / "fear_greed_gauge.png")

    # 词云
    generate_sentiment_wordclouds(df_daily_news,figure_dir,END_DT)

    print("Done!")
    print("Data ->", data_dir.resolve())



if __name__ == "__main__":  # required on Windows
    main()