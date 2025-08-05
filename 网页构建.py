import streamlit as st
from PIL import Image
import pandas as pd
from pathlib import Path
import subprocess
import base64
import streamlit.components.v1 as components
from datetime import datetime

# === 设置页面信息 ===
st.set_page_config(page_title="Crypto Investment Strategy Hub", layout="wide")
st.title("🪙 Cryptocurrency Market Analysis & Strategy Visualization")

# === 路径设置 ===
BASE_DIR = Path(__file__).resolve().parent
fig_dir = BASE_DIR / "figures"
text_dir = BASE_DIR / "text"

# === 辅助函数：居中显示图像 ===
def show_centered_img(img_path, caption="", width_percent=50, height=None):
    if not Path(img_path).exists():
        st.warning(f"no pictures: {img_path}")
        return

    with open(img_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    html_code = f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" style="width: {width_percent}%; border-radius: 8px;" />
            <div style="font-size: 14px; color: gray; margin-top: 8px;">{caption}</div>
        </div>
    """
    if height is None:
        height = int(width_percent * 6)
    components.html(html_code, height=height)

# === 辅助函数：运行外部脚本并提取关键输出 ===
def run_external_script(script_path: str):
    result_lines = []
    top_outputs = []
    bot_outputs = []
    notices = []   # ✅ 用于保存 ###NOTICE###

    process = subprocess.Popen(
        ["python", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    with st.expander("📄 Streaming Log Output (Click to Expand) ", expanded=False):
        for line in process.stdout:
            st.write(line.strip())
            result_lines.append(line.strip())

            # ✅ 提取关键输出
            if line.strip().startswith("###NOTICE###"):
                notices.append(line.strip().replace("###NOTICE###", "").strip())
            elif line.strip().startswith(">>>TOP20_"):
                top_outputs.append(line.strip().replace(">>>TOP20_", ""))
            elif line.strip().startswith(">>>BOT20_"):
                bot_outputs.append(line.strip().replace(">>>BOT20_", ""))

    process.wait()
    if process.returncode == 0:
        st.success("✅ Script Execution Completed! ")
    else:
        st.error(f"❌ Script Execution Failed, Return Code:{process.returncode}")

    return top_outputs, bot_outputs, notices


# === 顶部 Tabs 页面结构 ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎓Educational",
    "💭Market Sentiment",
    "📝Model Strategy",
    "🛠️Hyber-Parameters",
    "🔍Feature Selection",
    "📊Backtest Results",
    "▶️Prediction"
])

# === 页面 1：加密货币基础 ===
with tab1:
    st.header("📚 Crypto Basics & Educational Resources")

    st.markdown("Here are some beginner-friendly learning resources for new users:")

    # 🔗 教学链接
    st.markdown("🔗 [Binance Academy – What Is Cryptocurrency?](https://academy.binance.com/en/articles/what-is-a-cryptocurrency)")
    st.markdown("🔗 [CoinMarketCap：Stay Updated on the Latest Cryptocurrency Price Trends](https://coinmarketcap.com/)")
    st.markdown("🔗 [educational YouTuber：@simplyexplained](https://www.youtube.com/@simplyexplained)")

    # 🎥 教学视频嵌入
    st.subheader("🎥 What is crypto currency🪙?")
    st.video("https://www.youtube.com/watch?v=Zoz9gvhLgpM")


# === 页面 2：市场情绪指数 ===
with tab2:
    st.header("📰 Market Sentiment Index over the last 7-days ")
    update_file = BASE_DIR / "last_updated_wordcloud.txt"

    # 如果文件存在，读取日期
    if update_file.exists():
        last_updated_str = update_file.read_text().strip()
        st.info(f"📅 Last updated: {last_updated_str}")  # ✅ 永久显示
        # 转为日期进行比较
        try:
            last_updated_date = datetime.strptime(last_updated_str, "%Y-%m-%d").date()
            today = datetime.today().date()

            # ✅ 如果不是今天，提示用户需要更新
            if last_updated_date < today:
                st.warning("⚠️ This data may be outdated.")
        except ValueError:
            st.error("❌ Invalid update date format in last_updated_wordcloud.txt.")
    else:
        st.info("ℹ️ Word cloud and Gauge have not been generated yet.")

    if st.button("▶️ Update Word Cloud and Fear & Greed Gauge"):
        script_path = BASE_DIR / "情绪图更新.py"
        if not script_path.exists():
            st.error(f"❌ Script not found: {script_path}")
        else:
            with st.spinner("Generating... This may take up to 2 minutes."):
                exit_code = subprocess.call(["python", str(script_path)])
            if exit_code == 0:
                st.success("✅ Updated successfully!")

    option = st.radio(
        label="Display Options",
        options=["Word Cloud", "Fear & Greed Index"],
        horizontal=True,
        label_visibility="collapsed"
    )
    if option == "Word Cloud":
        st.subheader("☁️ Sentiment Word Cloud")

        col1, col2 = st.columns(2)
        with col1:
            wc_pos_path = fig_dir / "wordcloud_positive.png"
            show_centered_img(wc_pos_path, caption="Positive wordcloud", width_percent=90)
        with col2:
            wc_neg_path = fig_dir / "wordcloud_negative.png"
            show_centered_img(wc_neg_path, caption="Negative wordcloud", width_percent=90)
    elif option == "Fear & Greed Index":
        st.subheader("🧭 Fear & Greed Gauge")
        gauge_path = fig_dir / "fear_greed_gauge.png"
        show_centered_img(gauge_path, caption="Fear & Greed Gauge this week", width_percent=60,height=600)

# === 页面 3：模型策略介绍 ===
emoji_map = {
    "market": "🔵",
    "all": "🟢",
    "enet_EW": "🟡",
    "extra_EW": "🟠",
    "enet_ls": "🔴",
    "extra_ls": "🟣",
    "fusion_ls": "🟤"
}

with tab3:
    st.header("📘 Model Strategy Overview")
    intro_path = text_dir / "strategy_intro.txt"
    if intro_path.exists():
        with open(intro_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if ":" in line:
                    key = line.split(":")[0]
                    emoji = emoji_map.get(key, "")
                    st.markdown(f"{emoji} **{line}**")
                else:
                    st.markdown(line)
    else:
        st.info("no txt")


# === 页面 4：最优参数展示 ===
with tab4:
    st.header("⚙️ Optimal Hyber-Parameters Display")
    param_path = text_dir / "best_params.txt"
    if param_path.exists():
        with open(param_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="yaml")
    else:
        st.info("no best_params.txt ")

# === 页面 5：特征选择 ===
with tab5:
    st.header("💡 Feature Selection")
    st.markdown(
        "Below are the optimal features selected based on data from 2020-01-01 to 2025-07-30."
    )
    # 展示 enet 特征选择 txt
    enet_feat_path = text_dir / "enet_features.txt"
    if enet_feat_path.exists():
        st.subheader("🔴 ElasticNet Selected Features")
        st.markdown(
            "The ElasticNet model automatically selects features it considers important and contributive, while compressing others."
        )
        with open(enet_feat_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    else:
        st.info("no enet_selected_features.txt")

    # 展示 ExtraTrees 特征图像
    st.subheader("🔵 ExtraTrees Features Importance")
    extra_fig_all = fig_dir / "extra_all_feature_importance.png"
    extra_fig_market = fig_dir / "extra_market_feature_importance.png"

    col1, col2 = st.columns(2)
    with col1:
        show_centered_img(extra_fig_all, caption="All Features", width_percent=90, height=500)
    with col2:
        show_centered_img(extra_fig_market, caption="Market Features", width_percent=90, height=500)

# === 页面 6：策略回测结果对比 ===
with tab6:
    st.header("📈 Strategy Backtest Results Comparison")
    tab = st.radio(
        label="Comparison Metric Options",
        options=["Cumulative Return", "Volatility", "Average Return", "Sharpe Ratio", "Maximum Drawdown"],
        horizontal=True,
        label_visibility="collapsed"
    )
    base_name_map = {
        "Cumulative Return": "cum_return_comparison",
        "Volatility": "volatility_bar",
        "Average Return": "mean_return_bar",
        "Sharpe Ratio": "sharpe_bar",
        "Maximum Drawdown": "max_drawdown_bar"
    }
    base_name = base_name_map[tab]
    variants = ["all", "market"]
    for variant in variants:
        img_name = f"{base_name}_{variant}_with_baseline.png"
        img_path = fig_dir / img_name
        if "cum_return" in img_name:
            show_centered_img(img_path, width_percent=70, height=440)
        else:
            show_centered_img(img_path, width_percent=50, height=380)

# === 页面 7：运行函数 ===
with tab7:
    st.header("🧠 Get Next Week's Recommended Portfolio ! ")
    st.markdown(
        "🟦 Click the button to automatically fetch crypto market and news data up to the most recent Wednesday (t), merge it with historical data, and perform ETL.  \n"
        "🟨 Models are trained on the past 52 weeks (t−52 to t−1), and using features observed in week t to rank expected returns for week t+1.  \n"
        "🟥 **Disclaimer:** For reference only. Not financial advice."
    )

    if st.button("▶️ Click to Get Recommended Tokens"):
        st.session_state["log_expanded"] = False

        script_path = BASE_DIR / "客户调用脚本.py"
        if not script_path.exists():
            st.error(f"no {script_path}")
        else:
            with st.spinner("Running, please wait... Might take 5–10 mins if you haven't run this page in a while"):
                top_list, bot_list, notice_list = run_external_script(str(script_path))
            # ✅ 显示提示语（如“今天是周三…”）
            if notice_list:
                st.subheader("📢")
                for notice in notice_list:
                    st.warning(notice)

            if top_list:
                st.subheader(" 🟢 Top 20 Long Strategy Suggestions ")
                for item in top_list:
                    st.success(item)
            if bot_list:
                st.subheader("🔴 Bottom 20 Short Suggestions")
                for item in bot_list:
                    st.error(item)

            # ✅ 实时日志输出（默认折叠，强制控制）
            #log_path = Path("client_output/logs/runtime.log")
            #if log_path.exists():
                #with st.expander("📄 Streaming Log Output (Click to Expand) ", expanded=st.session_state["log_expanded"]):
                    #with open(log_path, "r", encoding="utf-8") as f:
                        #st.text(f.read())

# streamlit run C:\Users\10526\PycharmProjects\网页构建\网页构建.py