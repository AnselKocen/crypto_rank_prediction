from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

from crypto_project_pipeline import (
    build_week_dirs,
    stage1_etl,
    stage1_load_news,
    stage1_clean_text,
    stage2_feature_engineering,
    stage2_sentiment,
    merge_sentiment_feature
)

BASE_DIR = Path(__file__).resolve().parent

today = datetime.today()

news_start_date = today - timedelta(days=6)  # 包含今天在内，共7天
news_end_date = today + timedelta(days=1)    # 为了包含今天，加1天（有些API是exclusive）

print(f"Get news from {news_start_date.date()} to {news_end_date.date()}")

save_dir = BASE_DIR/"figures"
api_key = ""
df_news = stage1_load_news(api_key, news_start_date, news_end_date, save_dir)
df_clean = stage1_clean_text(df_news, save_dir)
stage2_sentiment(df_clean, save_dir, save_dir)

from crypto_project_pipeline import generate_sentiment_wordclouds
df_sent = pd.read_csv(save_dir / "stage_2_news_sentiment.csv")
generate_sentiment_wordclouds(df_sent, save_dir, news_end_date)

df_weekly_sent = pd.read_csv(save_dir / "stage_2_sentiment_weekly.csv")
from crypto_project_pipeline import draw_fear_greed_gauge_from_latest
draw_fear_greed_gauge_from_latest(df_weekly_sent, save_dir / "fear_greed_gauge.png")

update_time = datetime.now().strftime("%Y-%m-%d")
(Path(__file__).resolve().parent / "last_updated_wordcloud.txt").write_text(update_time)