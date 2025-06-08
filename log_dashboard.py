import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go

# --- SABITLER & KONFIGÜRASYON ---
COLORS = {
    "primary":    "#005F73",   # Koyu petrol
    "secondary":  "#0A9396",   # Ortam mavisi
    "accent":     "#EE9B00",   # Sıcak amber
    "light":      "#94D2BD",   # Silik yeşil
    "background": "#E9D8A6",   # Krem beji
    "text":       "#001219",   # Çok koyu lacivert
    "success":    "#52B788",   # Yumuşak yeşil
    "warning":    "#F4A261",   # Pastel turuncu
    "danger":     "#E63946",   # Tuğla kırmızısı
    "neutral":    "#ADB5BD"    # Açık gri
}
CSV_PATH     = "alerts.csv"
REFRESH_MIN  = 5
REFRESH_MAX  = 120
REFRESH_DEFAULT = 30
DOWNLOAD_TEMPLATE = "security_logs_{:%Y%m%d_%H%M%S}.csv"

# --- SAYFA AYARLARI & GLOBAL CSS ---
st.set_page_config(
    page_title="🛡️ Security Log Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.markdown(f"""
    <style>
      /* Arka plan ve satır hover */
      .stApp {{ background-color: {COLORS['background']}; }}
      .stDataFrame tbody tr:hover {{ background-color: #f0f2f5; }}
      /* Metrik kart stil */
      .metric-card {{ padding:15px; border-radius:10px; background:white;
                      box-shadow:0px 2px 5px rgba(0,0,0,0.1); }}
      /* Sekme başlıkları büyütme */
      button[data-baseweb="tab"] > div[title] {{
        font-size: 1.1rem !important;
        font-weight: 600;
      }}
      /* Bölüm başlıkları */
      .section-title {{
        font-size: 1.6rem !important;
        font-weight: 700;
        color: {COLORS['text']} !important;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
      }}
      /* Attack Analysis bölümü arka plan */
      .attack-section {{
        background-color: {COLORS['secondary']}20;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
      }}
    </style>
""", unsafe_allow_html=True)

# --- VERI ERIŞIM & ON-ISLEME ---
def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    """CSV’den veriyi oku, timestamp/ek sütunları oluştur."""
    try:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"]      = df["timestamp"].dt.hour
        df["date"]      = df["timestamp"].dt.date
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return pd.DataFrame()

# --- METRIK HESAPLAMA ---
def compute_metrics(df: pd.DataFrame) -> dict:
    """Anahtar metrikleri hesapla."""
    total        = len(df)
    unique_ips   = df["srcip"].nunique() if total else 0
    peak_hour    = df["hour"].value_counts().idxmax() if total else 0
    normal_count = len(df[df.get("attack_cat")=="Normal"]) if "attack_cat" in df.columns else 0
    anomaly_rate = (1 - normal_count/total)*100 if total else 0
    return {
        "total_flows":  total,
        "unique_ips":   unique_ips,
        "peak_hour":    peak_hour,
        "normal_count": normal_count,
        "anomaly_rate": anomaly_rate
    }

# --- AYARLAR PANELI ---
def render_settings() -> int:
    """Yenileme hızını seç ve otomatik yenilemeyi başlat."""
    with st.sidebar:
        st.header("⚙️ Settings")
        rate = st.slider("Refresh Rate (s)", REFRESH_MIN, REFRESH_MAX, REFRESH_DEFAULT)
        st_autorefresh(interval=rate*1000, key="auto_refresh")
    return rate

# --- HEADER ---
def render_header():
    st.markdown(f"""
        <div style='background-color:{COLORS['primary']}; padding:12px; border-radius:8px; margin-bottom:20px'>
          <h1 style='color:white; text-align:center; margin:0'>🛡️ Security Log Dashboard</h1>
          <p style='color:white; text-align:center; margin:0.5rem 0 0'>Real-time network traffic monitoring & attack detection</p>
        </div>
    """, unsafe_allow_html=True)

# --- METRIK KARTLARI ---
def render_metrics(metrics: dict):
    cols = st.columns(5)
    specs = [
        ("Total Flows",     metrics["total_flows"]),
        ("Unique IPs",      metrics["unique_ips"]),
        ("Peak Hour",       f"{metrics['peak_hour']}:00"),
        ("Normal Traffic",  metrics["normal_count"]),
        ("Anomaly Rate %",  f"{metrics['anomaly_rate']:.1f}%")
    ]
    for idx, (label, value) in enumerate(specs):
        with cols[idx]:
            st.markdown(f"""
                <div class='metric-card' style='border-left:5px solid {list(COLORS.values())[idx]};'>
                  <h4 style='margin:0;color:{COLORS['text']}'>{label}</h4>
                  <h2 style='margin:5px 0;color:{list(COLORS.values())[idx]}'>{value}</h2>
                </div>
            """, unsafe_allow_html=True)

# --- GRAFIK VE TABLO FONKSIYONLARI ---
def render_tcp_dist(df: pd.DataFrame):
    st.markdown("<div class='section-title'>🔄 TCP State Distribution</div>", unsafe_allow_html=True)
    data = df["state"].value_counts().reset_index()
    data.columns = ["state","count"]
    fig = px.bar(data, x="state", y="count", color="count",
                 color_continuous_scale=[COLORS["light"], COLORS["primary"]],
                 height=300)
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20,b=20,l=40,r=20),
        font=dict(color=COLORS["text"], size=12),
        xaxis=dict(tickfont=dict(color=COLORS["text"])),
        yaxis=dict(tickfont=dict(color=COLORS["text"]))
    )
    st.plotly_chart(fig, use_container_width=True)

def render_attack_dist(df: pd.DataFrame):
    if "attack_cat" not in df.columns: return
    st.markdown("<div class='attack-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔍 Attack Type Analysis</div>", unsafe_allow_html=True)
    data = df["attack_cat"].value_counts().reset_index()
    data.columns = ["attack","count"]
    fig = px.pie(data, names="attack", values="count", hole=0.4,
                 color_discrete_map={
                     k: (COLORS["success"] if k=="Normal" else COLORS["danger"])
                     for k in data["attack"]
                 },
                 height=300)
    fig.update_layout(margin=dict(t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_time_series(df: pd.DataFrame):
    if "attack_cat" not in df.columns: return
    st.markdown("<div class='section-title'>⏱️ Time Series Analysis</div>", unsafe_allow_html=True)
    ts = df.set_index("timestamp")
    normal  = ts[ts["attack_cat"]=="Normal"].resample("1T").size().reset_index(name="count")
    anomaly = ts[ts["attack_cat"]!="Normal"].resample("1T").size().reset_index(name="count")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normal["timestamp"], y=normal["count"], mode="lines",
                             line=dict(color=COLORS["success"], width=2), name="Normal"))
    fig.add_trace(go.Scatter(x=anomaly["timestamp"], y=anomaly["count"], mode="markers",
                             marker=dict(color=COLORS["danger"], size=8), name="Anomaly"))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=30,b=20,l=40,r=20), height=320,
        font=dict(color=COLORS["text"], size=12),
        xaxis=dict(title_text="Time", title_font=dict(size=14, color=COLORS["text"]),
                   tickfont=dict(size=11, color=COLORS["text"]), gridcolor="#E0E0E0"),
        yaxis=dict(title_text="Flow Count", title_font=dict(size=14, color=COLORS["text"]),
                   tickfont=dict(size=11, color=COLORS["text"]), gridcolor="#E0E0E0"),
        legend=dict(font=dict(size=12, color=COLORS["text"]), orientation="h", y=1.02, x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def render_data_table(df: pd.DataFrame):
    st.markdown("<div class='section-title'>📄 Recent Records (Top 50)</div>", unsafe_allow_html=True)
    sub = df.sort_values("timestamp", ascending=False).head(50).copy()
    def style_attack(v):
        return f"background-color:{COLORS['success']}" if v=="Normal" else f"background-color:{COLORS['danger']}"
    if "attack_cat" in sub.columns:
        st.dataframe(sub.style.applymap(style_attack, subset=["attack_cat"]), height=250)
    else:
        st.dataframe(sub, height=250)

# --- FOOTER ---
def render_footer():
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:{COLORS['neutral']}'>"
        f"Security Log Dashboard | Updated: {datetime.now():%Y-%m-%d %H:%M:%S}"
        f"</div>",
        unsafe_allow_html=True
    )

# --- ANA AKIS ---
def main():
    refresh_rate = render_settings()
    df = load_data()
    if df.empty: return

    metrics = compute_metrics(df)
    render_header()
    render_metrics(metrics)

    tabs = st.tabs(["Overview","Analysis","Data"])
    with tabs[0]:
        render_tcp_dist(df)
        render_attack_dist(df)
    with tabs[1]:
        render_time_series(df)
    with tabs[2]:
        render_data_table(df)
        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=DOWNLOAD_TEMPLATE.format(datetime.now()),
            mime="text/csv"
        )

    render_footer()

if __name__ == "__main__":
    main()
