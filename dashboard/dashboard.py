import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import time
import base64
import os

# ================== Page Setup ==================
st.set_page_config(page_title="UPS Anomaly Detection Dashboard", layout="wide")

# ================== Custom CSS ==================
st.markdown("""
    <style>
        #MainMenu, header, footer {visibility: hidden;}
        .st-emotion-cache-1v0mbdj { right: 20px !important; bottom: 20px !important; }
        .stToast {
            right: 20px !important;
            bottom: 80px !important;
            width: 280px !important;
            font-size: 13px !important;
            white-space: pre-line !important;
            padding: 8px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ================== Sidebar: Model Selection ==================
with st.sidebar:
    st.header("Select Detection Model")
    model_option = st.radio("Choose Model:", ["Isolation Forest", "KMeans"])
    run_simulation = st.checkbox("Start Real-Time Simulation")

# ================== Load Data ==================
if model_option == "Isolation Forest":
    df = pd.read_csv(
        "C:/Users/malle/ups_ai_backend/dashboard/ups_isolation_forest_results.csv")
    score_col = "iso_score"
    anomaly_col = "iso_anomaly"
    failure_col = "failure_iso"
    threshold = -0.05
    def compare(x): return x < threshold
    title = "Isolation Forest Anomaly Score Over Time"
else:
    df = pd.read_csv(
        "C:/Users/malle/ups_ai_backend/dashboard/ups_kmeans_results.csv")
    score_col = "kmeans_score"
    anomaly_col = "kmeans_anomaly"
    failure_col = "failure_kmeans"
    threshold = 6.0
    def compare(x): return x > threshold
    title = "KMeans Anomaly Score Over Time"

# ================== Failure Mapping ==================
failure_effect_phrases = {
    "Input Frequency Fault": "UPS switches to battery mode.",
    "Input Voltage Instability": "Battery shutdown.",
    "Overload": "UPS shutdown.",
    "Output Voltage Issue": "Hardware damage.",
    "Fan Failure": "Cooling system risk.",
    "Battery Degradation": "Reduced backup capacity.",
    "Unknown": "Unknown failure behavior."
}

# ================== Failure Distribution ==================
st.markdown(
    f"<h5>Frequency of Each Failure ({model_option})</h5>", unsafe_allow_html=True)

if failure_col in df.columns and not df[failure_col].dropna().empty:
    failure_counts = df[failure_col].value_counts().reset_index()
    failure_counts.columns = ['Failure Type', 'Count']
    bar_fig = px.bar(
        failure_counts,
        x='Failure Type',
        y='Count',
        color='Failure Type',
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    bar_fig.update_layout(
        xaxis_title="Failure Type",
        yaxis_title="Count",
        showlegend=False
    )
    st.plotly_chart(bar_fig, use_container_width=True,
                    config={"displayModeBar": False})
else:
    st.info("No failure data available to plot.")

# ================== Anomaly Score Line Plot ==================
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(df.index, df[score_col], color='royalblue',
        linewidth=1.2, label="Anomaly Score")
ax.axhline(y=threshold, color='crimson', linestyle='--',
           linewidth=1.2, label="Threshold")

anomalies = df[compare(df[score_col])]
ax.scatter(anomalies.index, anomalies[score_col],
           color='red', s=25, label="Detected Anomaly", zorder=5)

ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel("Time Index", fontsize=13)
ax.set_ylabel("Anomaly Score", fontsize=13)
ax.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.tick_params(axis='both', labelsize=11)
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
ax.legend(loc='upper right', fontsize=11, frameon=True,
          fancybox=True, facecolor='white')
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ================== Failure Table ==================
failure_options = df[failure_col].dropna().unique().tolist()
st.markdown("<h5>Failure Category Filter</h5>", unsafe_allow_html=True)
selected_failure = st.selectbox("Select a failure type:", [
                                "All"] + failure_options)
filtered_df = df[df[failure_col] ==
                 selected_failure] if selected_failure != "All" else df
st.markdown("Filtered Dataset View")
st.dataframe(filtered_df, use_container_width=True, height=500)

# ================== Real-Time Simulation (optional) ==================
if run_simulation:
    st.markdown("<h5>Real-Time Anomaly Simulation</h5>",
                unsafe_allow_html=True)
    placeholder_chart = st.empty()
    placeholder_alert = st.empty()
    live_scores = []

    # Sound alert function
    alert_path = "C:/Users/malle/ups_ai_backend/dashboard/alert.mp3"

    def play_alert_sound():
        if os.path.exists(alert_path):
            with open(alert_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                b64 = base64.b64encode(audio_bytes).decode()
                st.markdown(f"""
                    <audio autoplay>
                        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                """, unsafe_allow_html=True)

    # [your sound function remains the same]

    for i in range(len(df)):
        row = df.iloc[i]
        score = row[score_col]
        anomaly = row[anomaly_col]
        failure = row[failure_col] if pd.notna(row[failure_col]) else "Unknown"
        effect = failure_effect_phrases.get(failure, "Unknown.")

        live_scores.append(score)

        # 🔽 Reduced figure size and font sizes
        fig_rt, ax_rt = plt.subplots(figsize=(2.8, 1.2))  # smaller graph size
        ax_rt.plot(range(1, len(live_scores) + 1), live_scores,
                   color="blue", label="Anomaly Score", linewidth=0.5)
        ax_rt.axhline(y=threshold, color="red", linestyle="--",
                      label="Threshold", linewidth=0.5)
        ax_rt.set_title("Live UPS Anomaly Stream", fontsize=3)
        ax_rt.set_ylabel("Score", fontsize=3)
        ax_rt.set_xlabel("Time Index", fontsize=3)
        ax_rt.tick_params(axis='both', labelsize=3)
        ax_rt.legend(fontsize=3)
        ax_rt.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

        placeholder_chart.pyplot(fig_rt)
        plt.close(fig_rt)

        if anomaly == -1:
            alert_msg = f"Failure: {failure}\nEffect: {effect}"
            try:
                st.toast(alert_msg.strip(), icon="❌")
            except:
                placeholder_alert.error(alert_msg.strip())
            play_alert_sound()
        else:
            try:
                st.toast("Normal", icon="✅")
            except:
                placeholder_alert.success("Normal")

        time.sleep(2.0)
