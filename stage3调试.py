from pathlib import Path
import pandas as pd
from crypto_project_pipeline import build_week_dirs
from crypto_pipeline_stage3 import rolling_backtest_with_baseline_combined

# 创建目录结构
data_dir, fig_dir = build_week_dirs(
    week_folder="5545project",
    results_folder="stage3-4_result",
    data_sub="data",
    figure_sub="figures"
)

# 读取合并后的数据（用于计算 baseline）
df_merged = pd.read_csv(Path("stage1-2_results") / "clean_data" / "stage_2_merged.csv")

# 策略标签
tags = ["all", "market"]

# 逐个执行每个策略的回测
for tag in tags:
    print(f"🚀 开始回测策略: {tag}")
    df_enet = pd.read_csv(data_dir / f"pred_enet_{tag}.csv")
    df_extra = pd.read_csv(data_dir / f"pred_extra_{tag}.csv")

    rolling_backtest_with_baseline_combined(
        pred_enet_df=df_enet,
        pred_extra_df=df_extra,
        strategy_tag=tag,
        top_n=20,
        data_dir=data_dir,
        fig_dir=fig_dir,
        df_merged=df_merged
    )

# 合并所有策略的评估指标（含 baseline）
summary_list = []
for tag in tags:
    metrics_df = pd.read_csv(data_dir / f"metrics_{tag}_with_baseline.csv")
    metrics_df["tag"] = tag
    summary_list.append(metrics_df)

summary = pd.concat(summary_list, ignore_index=True)
summary.to_csv(data_dir / "strategy_vs_baseline_metrics.csv", index=False)
print("\n✅ 所有策略已完成，评估指标如下：\n")
print(summary)
