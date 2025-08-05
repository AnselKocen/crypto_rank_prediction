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
# ==============================================
#  Step 0: 自动确定最近的周三（返回 datetime）
# ==============================================
def get_latest_wednesday(today: datetime) -> datetime:
    days_since_wed = (today.weekday() - 2) % 7
    return today - timedelta(days=days_since_wed)

# ==============================================
#  Step 1: 增量更新 df_merged（只下载新的一周）
# ==============================================
def update_df_merged(api_key: str, history_path: str, output_root = BASE_DIR.name):
    today = datetime.today()
    latest_wed = get_latest_wednesday(today)  #  保持为 datetime 类型

    # 读取历史数据并检查是否已更新
    df_hist = pd.read_csv(history_path)
    df_hist["date"] = pd.to_datetime(df_hist["date"])
    last_date = df_hist["date"].max().date()
    next_day = last_date + timedelta(days=1)

    # 创建目录结构
    data_dir, fig_dir = build_week_dirs(
        week_folder="网页构建",
        results_folder="客户查看",
        data_sub="clean_data",
        figure_sub="figures"
    )
    print(f"[debug] data_dir: {data_dir.resolve()}")
    print(f"[debug] fig_dir: {fig_dir.resolve()}")

    print(f"###NOTICE### Latest Date in Historical Data: {last_date}，Latest Wednesday: {latest_wed.date()}")
    if last_date >= latest_wed.date():
        print("###NOTICE### Data already includes the most recent week. No update needed.")
        return df_hist.copy(), latest_wed, Path(history_path).parent

    elif today.weekday() == 2:  # Wednesday
        print("###NOTICE### Today is Wednesday. Please wait until Thursday to ensure complete data is used for predicting next week.")
        return df_hist.copy(), latest_wed, Path(history_path).parent

    # === 拉取最近的价格数据并构建 market 特征 ===
    stage1_etl(
        api_key=api_key,
        pages=[1,2],
        top_limit=100,
        history_limit=100,
        currency="USD",
        data_dir=data_dir
    )
    # 去除 latest_wed 之后的数据，避免出现未来行情
    df_prices = pd.read_csv(data_dir / "stage_1_crypto_data.csv")
    df_prices.columns = [c.strip().lower() for c in df_prices.columns]

    df_prices.columns = [col.strip().lower() for col in df_prices.columns]
    print(f"是否包含 'date'：{'date' in df_prices.columns}")
    print('--------------------------------------------------------------------------------------------------------')

    df_prices["date"] = pd.to_datetime(df_prices["date"])
    df_prices = df_prices[df_prices["date"] <= latest_wed]

    df_features = stage2_feature_engineering(df_prices, data_dir)
    df_features["date"] = pd.to_datetime(df_features["date"]).dt.date  # 转为 date 方便比较
    df_features = df_features[df_features["date"] > last_date]

    # === 抓取新闻并计算 sentiment 特征 ===
    news_start_date = latest_wed - timedelta(days=6)
    news_end_date = latest_wed + timedelta(days=1)
    print(f"抓取新闻时间范围: {news_start_date.date()} 至 {news_end_date.date()}")
    df_news = stage1_load_news(api_key, news_start_date, news_end_date, data_dir)
    df_clean = stage1_clean_text(df_news, data_dir)
    df_sent, df_weekly_sent = stage2_sentiment(df_clean, data_dir, fig_dir)

    # 更新词云
    # from crypto_project_pipeline import generate_sentiment_wordclouds
    # generate_sentiment_wordclouds(df_sent, fig_dir, latest_wed)
    # 存到更新路径
    #renew_path = BASE_DIR / "figures"
    #generate_sentiment_wordclouds(df_sent, renew_path, latest_wed)
    # === 合并 market 和 sentiment 特征 ===
    df_new = merge_sentiment_feature(df_features, df_weekly_sent, save_path=data_dir)
    df_new["ret_lead1"] = df_new.groupby("symbol")["return"].shift(-1)

    # === 合并到历史数据并保存 ===
    df_combined = pd.concat([df_hist, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["date", "symbol"]).sort_values("date")
    print(f"⚠即将覆盖历史文件：{history_path}，保存最新的 df_merged，共 {df_combined['date'].nunique()} 周数据。")
    df_combined.to_csv(history_path, index=False)
    # 生成仪表盘
    # from crypto_project_pipeline import draw_fear_greed_gauge_from_latest
    # df_latest = pd.read_csv(BASE_DIR / "df_merged_history.csv")
    # draw_fear_greed_gauge_from_latest(df_latest, fig_dir / "fear_greed_gauge1.png")
    # 更新仪表盘
    #renew_path = BASE_DIR/"figures"
    #draw_fear_greed_gauge_from_latest(df_latest, renew_path / "fear_greed_gauge1.png")
    return df_combined, latest_wed, data_dir




# ==============================================
# Step 2: 模型训练 + 预测（只预测最新一周）
# ==============================================
def run_prediction_pipeline(api_key: str, history_path: str):
    df_merged, latest_wed, data_dir = update_df_merged(api_key, history_path)
    df_merged = df_merged[df_merged["date"] <= latest_wed].copy()

    weeks = sorted(df_merged["date"].unique())
    train_weeks = weeks[-53:-1]  # 52周训练
    test_week = weeks[-1]        # 最新一周预测

    df_train = df_merged[df_merged["date"].isin(train_weeks)].dropna(subset=["ret_lead1"])
    df_test = df_merged[df_merged["date"] == test_week]

    feature_sets = {
        "all": [col for col in df_train.columns if col not in ["date", "symbol", "return", "ret_lead1", "open", "high", "low", "close"]],
        "market": [col for col in df_train.columns if any(k in col for k in ["momentum", "volatility", "usd_volume", "base_volume", "return_sign", "long_candle", "strev"])]
    }

    fixed_params = {
        "enet_all": {"alpha": 0.01, "l1_ratio": 0.5},
        "enet_market": {"alpha": 0.01, "l1_ratio": 0.5},
        "extra_all": {"n_estimators": 200, "max_depth": 10, "max_features": "sqrt"},
        "extra_market": {"n_estimators": 200, "max_depth": 10, "max_features": "sqrt"}
    }

    results = {}

    for model_key in ["enet", "extra"]:
        for tag in ["all", "market"]:
            name = f"{model_key}_{tag}"
            print(f"\n 正在训练模型 {name} ...")

            X_train = df_train[feature_sets[tag]].copy()
            y_train = df_train["ret_lead1"].fillna(0)
            X_test = df_test[feature_sets[tag]].copy()

            imputer = SimpleImputer(strategy="mean")
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            if model_key == "enet":
                model = ElasticNet(max_iter=10000, random_state=42, **fixed_params[name])
            else:
                model = ExtraTreesRegressor(random_state=42, **fixed_params[name])

            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            df_out = df_test[["symbol"]].copy()
            df_out["y_pred"] = y_pred
            df_out = df_out.sort_values("y_pred", ascending=False)

            results[name] = df_out
            out_path = data_dir / f"pred_{name}.csv"
            df_out.to_csv(out_path, index=False)
            print(f" {name} 输出保存至 {out_path}")

    # === 打印推荐币种 ===
    for name, df in results.items():
        print(f"\n 模型: {name}")
        print("Top 20 推荐币种:")
        print(df.head(20).to_string(index=False))
        print("Bottom 20 做空建议:")
        print(df.tail(20).to_string(index=False))

        print(f">>>TOP20_{name}: " + ",      ".join(df.head(20)["symbol"].tolist()))
        print(f">>>BOT20_{name}: " + ",      ".join(df.tail(20)["symbol"].tolist()))


# ==============================================
#  执行主程序（客户只需运行此脚本）
# ==============================================
if __name__ == "__main__":
    API_KEY = ""  # ← 请替换为你自己的 Coindesk API 密钥
    HISTORY_PATH = BASE_DIR / "df_merged_history.csv"
    run_prediction_pipeline(API_KEY, str(HISTORY_PATH))