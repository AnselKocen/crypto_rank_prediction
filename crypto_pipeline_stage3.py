import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


def evaluate_sentiment_confusion(sent_df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sent_df["positive"] = pd.to_numeric(sent_df["positive"], errors="coerce")
    sent_df["sentiment"] = pd.to_numeric(sent_df["sentiment"], errors="coerce")

    if "positive" in sent_df.columns and sent_df["positive"].notna().any():
        cm = confusion_matrix(sent_df["positive"], sent_df["sentiment"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig(fig_dir / "confusion_matrix.png", bbox_inches="tight")
        plt.close()

        rep = classification_report(sent_df["positive"], sent_df["sentiment"])
        (fig_dir / "classification_report.txt").write_text(rep)
    else:
        print("⚠️ No usable 'positive' column in DataFrame.")


def plot_roc_curve(sent_df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    y_true = pd.to_numeric(sent_df["positive"], errors="coerce")
    y_score = pd.to_numeric(sent_df["compound"], errors="coerce")

    mask = y_true.notna() & y_score.notna()
    y_true = y_true[mask]
    y_score = y_score[mask]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png", bbox_inches="tight")
    plt.close()
    print(f"✅ ROC曲线保存至 {fig_dir / 'roc_curve.png'}")


def plot_pr_curve(sent_df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    y_true = pd.to_numeric(sent_df["positive"], errors="coerce")
    y_score = pd.to_numeric(sent_df["compound"], errors="coerce")

    mask = y_true.notna() & y_score.notna()
    y_true = y_true[mask]
    y_score = y_score[mask]

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR curve (AP = {avg_precision:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_dir / "pr_curve.png", bbox_inches="tight")
    plt.close()
    print(f"✅ PR曲线保存至 {fig_dir / 'pr_curve.png'}")


#####################################################################
import pandas as pd, numpy as np, os, joblib, ast
from pathlib import Path
from collections import Counter
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error

def rolling_weekly_prediction(df: pd.DataFrame, save_dir: Path, roll_window: int = 52):
    model_configs = {
        "enet": {
            "estimator": ElasticNet(max_iter=10000, random_state=42),
            "param_grid": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.1, 0.5, 0.9]
            }
        },
        "extra": {
            "estimator": ExtraTreesRegressor(random_state=42),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 10],
                "model__max_features": ["sqrt", "log2"]
            }
        }
    }

    def get_feature_sets(df: pd.DataFrame):
        exclude_cols = ["date", "symbol", "return", "ret_lead1", "open", "high", "low", "close"]
        market_features = [col for col in df.columns if any(k in col for k in ["momentum", "volatility", "usd_volume", "base_volume", "return_sign", "long_candle", "strev"])]
        all_features = [col for col in df.columns if col not in exclude_cols]
        return {
            "all": all_features,
            "market": market_features
        }

    df = df.copy()
    df["ret_lead1"] = df.groupby("symbol")["return"].shift(-1)
    weeks = sorted(df["date"].unique())
    feature_sets = get_feature_sets(df)

    for tag, feature_set in feature_sets.items():
        for model_key, config in model_configs.items():
            print(f"\n🚀 [{model_key.upper()}-{tag}] 正在调参并训练最终模型...")

            df_train_all = df.dropna(subset=["ret_lead1"])
            X_full = df_train_all[feature_set].copy()
            y_full = df_train_all["ret_lead1"].copy()
            imputer = SimpleImputer(strategy="mean")
            X_full_imputed = imputer.fit_transform(X_full)
            y_full_clean = y_full.fillna(0)

            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("model", config["estimator"])
            ])
            grid = GridSearchCV(pipe, config["param_grid"], scoring="neg_mean_squared_error", cv=3, n_jobs=-1)
            grid.fit(X_full_imputed, y_full_clean)

            best_estimator = grid.best_estimator_
            best_param_dict = grid.best_params_
            print(f"✅ 最优参数: {best_param_dict}")

            model_save_dir = Path("models") / f"{model_key}_{tag}"
            model_save_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_estimator, model_save_dir / "final_model.pkl")
            with open(model_save_dir / "best_params.txt", "w") as f:
                f.write(str(best_param_dict))
            print(f"📦 模型与参数保存于: {model_save_dir}")

            all_results, all_y_true, all_y_pred = [], [], []

            for i in range(roll_window, len(weeks) - 1):
                train_weeks = weeks[i - roll_window:i]
                test_week = weeks[i]

                df_train = df[df["date"].isin(train_weeks)].dropna(subset=["ret_lead1"])
                df_test = df[df["date"] == test_week]

                X_test = df_test[feature_set].copy()
                y_test = df_test["ret_lead1"].copy()
                X_test_imputed = imputer.transform(X_test)
                y_pred = best_estimator.predict(X_test_imputed)

                result = df_test[["date", "symbol"]].copy()
                result["y_pred"] = y_pred
                result["y_true"] = y_test.values
                result["model_tag"] = f"{model_key}_{tag}"
                result["rank"] = result.groupby("date")["y_pred"].rank(ascending=False, method="first")
                all_results.append(result)

                all_y_true.extend(y_test.dropna().values)
                all_y_pred.extend([p for t, p in zip(y_test, y_pred) if not pd.isna(t)])

            r2 = r2_score(all_y_true, all_y_pred)
            mse = mean_squared_error(all_y_true, all_y_pred)
            print(f"📊 [评估] R²: {r2:.4f} | MSE: {mse:.6f}")

            final_df = pd.concat(all_results, ignore_index=True)
            out_path = save_dir / f"pred_{model_key}_{tag}.csv"
            final_df.to_csv(out_path, index=False)
            print(f"📄 预测排名结果已保存: {out_path}")


#########################################################################
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal
import matplotlib.dates as mdates
import numpy as np
import warnings

def rolling_backtest_with_baseline_combined(
    pred_enet_df: pd.DataFrame,
    pred_extra_df: pd.DataFrame,
    strategy_tag: Literal["all", "market"],
    top_n: int,
    data_dir: Path,
    fig_dir: Path,
    df_merged: pd.DataFrame
):
    all_weeks = sorted(pred_enet_df["date"].unique())
    fusion_results, enet_results, extra_results, enet_EW_results, extra_EW_results = [], [], [], [], []

    # === baseline 策略（等权平均投资） ===
    df = df_merged.copy()
    df["ret_lead1"] = df.groupby("symbol")["return"].shift(-1)
    df.dropna(subset=["ret_lead1"], inplace=True)
    baseline_results = []
    for i in range(len(all_weeks) - 1):
        this_week = all_weeks[i]
        next_week = all_weeks[i + 1]
        df_this_week = df[df["date"] == this_week]
        df_next_week = df[df["date"] == next_week][["symbol", "ret_lead1"]]
        selected = df_this_week["symbol"]
        portfolio = pd.DataFrame({"symbol": selected})
        portfolio["weight"] = 1 / len(portfolio)
        merged = pd.merge(portfolio, df_next_week, on="symbol", how="left").dropna()
        merged["weighted_return"] = merged["weight"] * merged["ret_lead1"]
        baseline_results.append({"date": next_week, "return": merged["weighted_return"].sum()})
    df_baseline = pd.DataFrame(baseline_results)
    df_baseline["cum_return"] = (1 + df_baseline["return"]).cumprod()
    df_baseline["strategy"] = "baseline_EW"

    # === 策略回测部分 ===
    for i in range(len(all_weeks) - 1):
        this_week = all_weeks[i]
        next_week = all_weeks[i + 1]
        df_enet = pred_enet_df[pred_enet_df["date"] == this_week]
        df_extra = pred_extra_df[pred_extra_df["date"] == this_week]
        df_next = pred_enet_df[pred_enet_df["date"] == next_week][["symbol", "y_true"]]

        # 选择 top/bottom n 个币种
        enet_top = df_enet.sort_values("y_pred", ascending=False).head(top_n)["symbol"]
        enet_bot = df_enet.sort_values("y_pred", ascending=True).head(top_n)["symbol"]
        extra_top = df_extra.sort_values("y_pred", ascending=False).head(top_n)["symbol"]
        extra_bot = df_extra.sort_values("y_pred", ascending=True).head(top_n)["symbol"]

        required_syms = set(enet_top) | set(enet_bot) | set(extra_top) | set(extra_bot)
        if not required_syms.issubset(set(df_next["symbol"])):
            warnings.warn(f"Skipping week {this_week}: missing y_true for some required symbols.")
            continue

        def compute_portfolio_return(df_next, long_syms, short_syms=None):
            long_df = pd.DataFrame({"symbol": list(long_syms)})
            long_df["weight"] = 1.0 / len(long_df)
            if short_syms is None:
                portfolio = long_df
            else:
                short_df = pd.DataFrame({"symbol": list(short_syms)})
                short_df["weight"] = -1.0 / len(short_df)
                portfolio = pd.concat([long_df, short_df], ignore_index=True)
            merged = pd.merge(portfolio, df_next, on="symbol", how="left").dropna()
            if merged.empty or merged.shape[0] < (len(portfolio) * 0.8):
                return np.nan
            merged["weighted_return"] = merged["weight"] * merged["y_true"]
            return merged["weighted_return"].sum()

        enet_EW_return = compute_portfolio_return(df_next, enet_top)
        extra_EW_return = compute_portfolio_return(df_next, extra_top)
        enet_ls_return = compute_portfolio_return(df_next, enet_top, enet_bot)
        extra_ls_return = compute_portfolio_return(df_next, extra_top, extra_bot)

        if not np.isnan(enet_EW_return):
            enet_EW_results.append({"date": next_week, "return": enet_EW_return})
        if not np.isnan(extra_EW_return):
            extra_EW_results.append({"date": next_week, "return": extra_EW_return})
        if not np.isnan(enet_ls_return):
            enet_results.append({"date": next_week, "return": enet_ls_return})
        if not np.isnan(extra_ls_return):
            extra_results.append({"date": next_week, "return": extra_ls_return})

        # === fusion 策略 ===
        both_long = set(enet_top) & set(extra_top)
        only_long = (set(enet_top) | set(extra_top)) - both_long
        both_short = set(enet_bot) & set(extra_bot)
        only_short = (set(enet_bot) | set(extra_bot)) - both_short

        long_df = pd.DataFrame({"symbol": list(both_long) + list(only_long)})
        long_df["raw_weight"] = [0.65] * len(both_long) + [0.35] * len(only_long)
        long_df["weight"] = long_df["raw_weight"] / long_df["raw_weight"].sum()

        short_df = pd.DataFrame({"symbol": list(both_short) + list(only_short)})
        short_df["raw_weight"] = [-0.65] * len(both_short) + [-0.35] * len(only_short)
        short_df["weight"] = short_df["raw_weight"] / abs(short_df["raw_weight"].sum())

        fusion_portfolio = pd.concat([long_df, short_df], ignore_index=True)
        merged = pd.merge(fusion_portfolio, df_next, on="symbol", how="left").dropna()
        if merged.empty or merged.shape[0] < len(fusion_portfolio) * 0.8:
            warnings.warn(f"Skipping fusion week {this_week}: incomplete data for fusion portfolio.")
            continue
        merged["weighted_return"] = merged["weight"] * merged["y_true"]
        fusion_results.append({"date": next_week, "return": merged["weighted_return"].sum()})

    def build_df(results, label):
        df = pd.DataFrame(results).dropna()
        df["cum_return"] = (1 + df["return"]).cumprod()
        df["strategy"] = label
        return df

    df_fusion = build_df(fusion_results, "fusion_ls")
    df_enet = build_df(enet_results, "enet_ls")
    df_extra = build_df(extra_results, "extra_ls")
    df_enet_EW = build_df(enet_EW_results, "enet_EW")
    df_extra_EW = build_df(extra_EW_results, "extra_EW")
    df_all = pd.concat([df_fusion, df_enet, df_extra, df_enet_EW, df_extra_EW, df_baseline], ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all.to_csv(data_dir / f"rolling_backtest_{strategy_tag}_with_baseline.csv", index=False)

    color_map = {
        "fusion_ls": "mediumpurple",
        "enet_ls": "orange",
        "extra_ls": "deepskyblue",
        "enet_EW": "red",
        "extra_EW": "limegreen",  # 新增颜色
        "baseline_EW": "lightgray"
    }

    plt.figure(figsize=(12, 5))
    for strategy, group in df_all.groupby("strategy"):
        plt.plot(group["date"], group["cum_return"], label=strategy, linewidth=2, color=color_map.get(strategy, None))
    plt.title(f"Cumulative Return Comparison: {strategy_tag}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(fig_dir / f"cum_return_comparison_{strategy_tag}.png")
    plt.close()

    metrics = []
    for label, df in df_all.groupby("strategy"):
        ret = df["return"]
        max_drawdown = (df["cum_return"].cummax() - df["cum_return"]) / df["cum_return"].cummax()
        metrics.append({
            "strategy": label,
            "sharpe": ret.mean() / ret.std(),
            "volatility": ret.std(),
            "mean_return": ret.mean(),
            "max_drawdown": max_drawdown.max()
        })
    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(data_dir / f"metrics_{strategy_tag}_with_baseline.csv", index=False)

    for col in ["sharpe", "volatility", "mean_return", "max_drawdown"]:
        plt.figure(figsize=(8, 4))
        colors = [color_map.get(strategy, "black") for strategy in metric_df["strategy"]]
        bars = plt.bar(metric_df["strategy"], metric_df[col], color=colors)
        plt.title(f"{col.title()} Comparison: {strategy_tag}")
        plt.ylabel(col.title())
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{col}_bar_{strategy_tag}_with_baseline.png")
        plt.close()
