import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Karhutla Analytics - Situasi Karhutla Nasional 2025",
    layout="wide",
    page_icon="◆",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. KONSTANTA GLOBAL & TOKEN WARNA GLASSMORPHISM
# ==============================================================================
FEATURE_COLS = ['Luas_Karhutla_2025', 'Emisi_CO2_2025', 'Total_Hotspot_2025']

ZONE_COLORS = {
    'Aman': '#4CC38A',
    'Rendah': '#A7D06B',
    'Waspada': '#FFC247',
    'Siaga': '#FF8A2B',
    'Bahaya': '#F04438'
}

CLUSTER_LABEL_MAP = {
    2: ['Aman', 'Bahaya'],
    3: ['Aman', 'Waspada', 'Bahaya'],
    4: ['Aman', 'Waspada', 'Siaga', 'Bahaya'],
    5: ['Aman', 'Rendah', 'Waspada', 'Siaga', 'Bahaya']
}

PROV_COORDS = {
    'Aceh': (4.6951, 96.7494),
    'Bali': (-8.4095, 115.1889),
    'Bangka Belitung': (-2.7411, 106.4406),
    'Banten': (-6.4058, 106.0640),
    'Bengkulu': (-3.5778, 102.3464),
    'DKI Jakarta': (-6.2088, 106.8456),
    'Gorontalo': (0.6999, 122.4467),
    'Jambi': (-1.6101, 102.7724),
    'Jawa Barat': (-6.9175, 107.6191),
    'Jawa Tengah': (-7.1510, 110.1403),
    'Jawa Timur': (-7.5361, 112.2384),
    'Kalimantan Barat': (-0.2788, 111.4753),
    'Kalimantan Selatan': (-3.0926, 115.2838),
    'Kalimantan Tengah': (-1.6815, 113.3824),
    'Kalimantan Timur': (0.5387, 116.4194),
    'Kalimantan Utara': (3.0731, 116.0414),
    'Kepulauan Riau': (1.0800, 104.0300),
    'Lampung': (-4.5586, 105.4068),
    'Maluku': (-3.2385, 130.1453),
    'Maluku Utara': (1.5710, 127.8088),
    'Nusa Tenggara Barat': (-8.6529, 117.3616),
    'Nusa Tenggara Timur': (-8.6574, 121.0794),
    'Papua': (-2.5337, 140.7181),
    'Papua Barat': (-1.3361, 133.1747),
    'Papua Barat Daya': (-0.8760, 131.2558),
    'Papua Pegunungan': (-4.0833, 138.9500),
    'Papua Selatan': (-7.4950, 140.4000),
    'Papua Tengah': (-3.3667, 135.5000),
    'Riau': (0.2933, 101.7068),
    'Sulawesi Barat': (-2.8441, 119.2321),
    'Sulawesi Selatan': (-3.6688, 119.9741),
    'Sulawesi Tengah': (-1.4300, 121.4456),
    'Sulawesi Tenggara': (-4.1449, 122.1746),
    'Sulawesi Utara': (0.6247, 123.9750),
    'Sumatera Barat': (-0.7399, 100.8000),
    'Sumatera Selatan': (-3.3194, 104.9144),
    'Sumatera Utara': (2.1154, 99.5451),
    'Yogyakarta': (-7.7956, 110.3695)
}

CONTINUOUS_COLORSCALE = [
    [0.0, '#1A0F0B'],
    [0.25, '#5A2108'],
    [0.5, '#C2540F'],
    [0.75, '#FF8A2B'],
    [1.0, '#FFD9A8']
]


# ==============================================================================
# 3. FUNGSI DATA & PEMODELAN (LOGIKA TIDAK BERUBAH)
# ==============================================================================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "DATASET_FINAL_SIAP_LATIH.xlsx")

    if not os.path.exists(file_path):
        st.error(f"File dataset tidak ditemukan di: {file_path}")
        st.stop()

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"Gagal membaca file Excel dataset: {e}")
        st.stop()

    required_cols = ['Provinsi', 'Luas_Karhutla_2025', 'Emisi_CO2_2025', 'Total_Hotspot_2025']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Kolom wajib tidak ditemukan pada dataset: {missing_cols}")
        st.stop()

    mask_total = df['Provinsi'].astype(str).str.strip().str.lower() == 'total'
    nasional = df[mask_total].copy()
    df_prov = df[~mask_total].copy()

    if df_prov.empty:
        st.error("Data provinsi kosong setelah pemisahan baris Total.")
        st.stop()

    df_prov['Intensitas_Emisi'] = df_prov.apply(
        lambda r: (r['Emisi_CO2_2025'] / r['Luas_Karhutla_2025']) if r['Luas_Karhutla_2025'] > 0 else 0.0,
        axis=1
    )

    return df_prov, nasional


def run_kmeans(df, k=3):
    df_res = df.copy()
    X_log = np.log1p(df_res[FEATURE_COLS])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    raw_clusters = km.fit_predict(X_scaled)
    df_res['Zona_Risiko'] = raw_clusters

    cluster_means = df_res.groupby('Zona_Risiko')['Emisi_CO2_2025'].mean().sort_values()
    labels = CLUSTER_LABEL_MAP.get(k, [f"Klaster_{i+1}" for i in range(k)])
    mapping_labels = {cid: label for cid, label in zip(cluster_means.index, labels)}

    df_res['Status_Kerawanan'] = df_res['Zona_Risiko'].map(mapping_labels)
    return df_res


def hitung_korelasi(df):
    return df[FEATURE_COLS].corr(method='pearson')


def cek_data_tidak_lengkap(df):
    kondisi = (
        ((df['Luas_Karhutla_2025'] == 0) & ((df['Emisi_CO2_2025'] > 0) | (df['Total_Hotspot_2025'] > 0))) |
        ((df['Total_Hotspot_2025'] == 0) & ((df['Luas_Karhutla_2025'] > 0) | (df['Emisi_CO2_2025'] > 0))) |
        ((df['Emisi_CO2_2025'] == 0) & ((df['Luas_Karhutla_2025'] > 0) | (df['Total_Hotspot_2025'] > 0)))
    )
    return df[kondisi].copy()


# ==============================================================================
# 4. HELPER FORMATTING & STYLE PLOTLY
# ==============================================================================
def format_id(val, decimals=0):
    if pd.isna(val):
        return "-"
    s = f"{val:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def format_compact_id(val, unit=""):
    if pd.isna(val) or val == 0:
        return f"0 {unit}".strip()
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"{format_id(val / 1_000_000, 2)} Juta {unit}".strip()
    elif abs_val >= 1_000:
        return f"{format_id(val / 1_000, 1)} Ribu {unit}".strip()
    else:
        return f"{format_id(val, 0)} {unit}".strip()


def hex_to_rgba(hex_str, alpha=0.18):
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'


def render_sparkline_bars(values, height=20, bar_color='#FF8A2B', fill_color='rgba(255, 138, 43, 0.18)'):
    if not values or len(values) == 0:
        return ""
    max_val = max(values) if max(values) > 0 else 1.0
    n = len(values)
    bar_w = 8
    gap = 3
    total_w = n * bar_w + (n - 1) * gap
    svg_bars = []
    for i, v in enumerate(values):
        h = max(int((v / max_val) * (height - 3)), 2)
        x = i * (bar_w + gap)
        y = height - h
        svg_bars.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" rx="1" fill="{fill_color}" stroke="{bar_color}" stroke-width="1.2"/>')
    return f'<svg width="{total_w}" height="{height}" viewBox="0 0 {total_w} {height}" style="vertical-align: middle;">{"".join(svg_bars)}</svg>'


def style_plotly(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F4EDE8", size=12),
        title_font=dict(family="Manrope, sans-serif", size=15, color="#F4EDE8"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(color="#F4EDE8")
        ),
        hoverlabel=dict(
            bgcolor="#140D0B",
            bordercolor="#FF8A2B",
            font=dict(family="Inter, sans-serif", color="#F4EDE8", size=12)
        ),
        margin=dict(l=35, r=25, t=45, b=35)
    )
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        linecolor="rgba(255, 255, 255, 0.12)",
        tickfont=dict(color="#9A8B84", family="Inter, sans-serif")
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        linecolor="rgba(255, 255, 255, 0.12)",
        tickfont=dict(color="#9A8B84", family="Inter, sans-serif")
    )
    return fig


# ==============================================================================
# 5. INJEKSI CUSTOM CSS & GOOGLE FONTS (GLASSMORPHISM STYLE)
# ==============================================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap" rel="stylesheet">

<style>
/* Sembunyikan Header / Deploy / Footer Bawaan Streamlit */
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
header {visibility: hidden !important;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stDecoration"] {display: none !important;}
[data-testid="stStatusWidget"] {display: none !important;}

/* Latar Belakang & Tata Letak Dasar */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0706 !important;
    color: #F4EDE8 !important;
    font-family: 'Inter', sans-serif !important;
}

/* 4 Halo Gradasi Ambien di Empat Sudut */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    background: 
        radial-gradient(circle at 10% 8%, rgba(255, 122, 26, 0.16) 0%, transparent 40%),
        radial-gradient(circle at 90% 10%, rgba(255, 122, 26, 0.16) 0%, transparent 40%),
        radial-gradient(circle at 10% 90%, rgba(255, 122, 26, 0.16) 0%, transparent 40%),
        radial-gradient(circle at 90% 90%, rgba(255, 122, 26, 0.16) 0%, transparent 40%);
}

[data-testid="stAppViewContainer"] > .main {
    position: relative;
    z-index: 1;
}

/* Sidebar Glass Panel */
[data-testid="stSidebar"] {
    background: rgba(10, 7, 6, 0.85) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Animasi Fade-In Ringan Sekali Saat Memuat Halaman */
@media (prefers-reduced-motion: no-preference) {
    .glass-main-view {
        animation: fadeIn 0.35s ease-out forwards;
    }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Panel Navigasi Atas */
.top-nav-panel {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.09);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 16px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: -15px;
    margin-bottom: 20px;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-icon-box {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    border: 1px solid rgba(255, 138, 43, 0.35);
    background: rgba(255, 138, 43, 0.10);
    display: flex;
    align-items: center;
    justify-content: center;
}
.brand-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #F4EDE8;
    letter-spacing: -0.3px;
}
.data-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: 'Manrope', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    color: #F4EDE8;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 999px;
    padding: 5px 13px;
}
.data-dot {
    width: 6px;
    height: 6px;
    background: #FFB04A;
    border-radius: 50%;
    display: inline-block;
}

/* Hero Section */
.hero-wrapper {
    margin-bottom: 20px;
}
.hero-title-row {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 6px;
}
.hero-title {
    font-family: 'Manrope', sans-serif !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #F4EDE8 !important;
    margin: 0 !important;
    line-height: 1.2;
    letter-spacing: -0.5px;
}
.year-badge {
    font-family: 'Manrope', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    padding: 2px 12px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #FF8A2B;
    background: rgba(255, 138, 43, 0.08);
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    color: #9A8B84;
    line-height: 1.5;
    margin-bottom: 14px;
}
.hero-divider {
    height: 1px;
    background: rgba(255, 255, 255, 0.08);
    margin-bottom: 20px;
}

/* Glass Card Umum */
.glass-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.09);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 16px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    padding: 18px 20px;
    transition: border-color 0.2s ease;
}

/* Grid Kartu KPI */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 20px;
}
@media (max-width: 1024px) {
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 600px) {
    .kpi-grid { grid-template-columns: 1fr; }
}

.kpi-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.09);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 16px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    padding: 18px 20px;
    transition: border-color 0.2s ease;
}
.kpi-card:hover {
    border-color: rgba(255, 138, 43, 0.45) !important;
}

.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.kpi-label {
    color: #9A8B84;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.kpi-icon-box {
    width: 30px;
    height: 30px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.10);
    background: rgba(255, 255, 255, 0.03);
    display: flex;
    align-items: center;
    justify-content: center;
}
.kpi-number {
    font-family: 'Manrope', sans-serif !important;
    font-variant-numeric: tabular-nums;
    font-size: 1.85rem;
    font-weight: 700;
    color: #F4EDE8;
    line-height: 1.15;
    letter-spacing: -0.5px;
    white-space: nowrap;
}
.kpi-unit {
    font-family: 'Inter', sans-serif;
    color: #9A8B84;
    font-size: 0.8rem;
    margin-top: 3px;
}
.kpi-sparkline-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}
.kpi-sparkline-note {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: #9A8B84;
}
.kpi-caption {
    font-family: 'Inter', sans-serif;
    font-size: 0.76rem;
    color: #9A8B84;
    margin-top: 6px;
}

/* Insight Box */
.quiet-insight-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-left: 3px solid #FF8A2B;
    border-radius: 14px;
    padding: 16px 20px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    margin-bottom: 22px;
}
.insight-badge-label {
    font-family: 'Manrope', sans-serif;
    font-size: 0.76rem;
    font-weight: 700;
    color: #FF8A2B;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
    display: block;
    text-transform: uppercase;
}
.insight-body-quiet {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: #9A8B84;
    line-height: 1.65;
}
.insight-body-quiet b {
    color: #F4EDE8;
    font-weight: 600;
}

/* Styling Tabs sebagai Navigasi Kaca */
[data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.035) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 8px !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    margin-bottom: 22px !important;
    display: inline-flex !important;
}

button[data-baseweb="tab"] {
    background-color: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #9A8B84 !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 8px 18px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

button[data-baseweb="tab"]:hover {
    color: #F4EDE8 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #FF8A2B !important;
    border-bottom: 2px solid #FF8A2B !important;
    background-color: rgba(255, 138, 43, 0.10) !important;
}

/* Header Seksi */
.section-header {
    margin-bottom: 16px;
}
.section-index-line {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
}
.section-badge {
    font-family: 'Manrope', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: #FF8A2B;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.10);
    padding: 2px 8px;
    border-radius: 6px;
}
.section-line {
    flex: 1;
    height: 1px;
    background: rgba(255, 255, 255, 0.08);
}
.section-title {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #F4EDE8 !important;
    margin: 0 !important;
}
.section-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.86rem;
    color: #9A8B84;
    margin: 4px 0 0 0;
}

/* Sidebar: Multiselect Tag & Outline Button */
[data-baseweb="tag"] {
    background-color: rgba(255, 138, 43, 0.12) !important;
    border: 1px solid #FF8A2B !important;
    color: #F4EDE8 !important;
    border-radius: 999px !important;
}
[data-baseweb="tag"] span {
    color: #F4EDE8 !important;
}

.stButton>button {
    background: transparent !important;
    color: #F4EDE8 !important;
    border: 1px solid #FF8A2B !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}
.stButton>button:hover {
    background: rgba(255, 138, 43, 0.12) !important;
    border-color: #FF8A2B !important;
    color: #F4EDE8 !important;
    box-shadow: none !important;
    transform: none !important;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 6. PIPELINE DATA AWAL & K-MEANS
# ==============================================================================
df_prov, nasional = load_data()


# ==============================================================================
# 7. SIDEBAR (FILTER & KONTROL PANEL)
# ==============================================================================
st.sidebar.markdown("""
<div style="padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 18px;">
    <h2 style="font-family: 'Manrope', sans-serif; font-size: 1.2rem; font-weight: 700; color: #F4EDE8; margin: 0; letter-spacing: -0.3px;">
        PANEL KONTROL
    </h2>
    <span style="color: #9A8B84; font-size: 0.78rem;">Konfigurasi parameter & filter spasial</span>
</div>
""", unsafe_allow_html=True)

# Slider Jumlah Klaster (k)
k_val = st.sidebar.slider(
    "Jumlah Klaster Risiko (k)",
    min_value=2,
    max_value=5,
    value=3,
    step=1,
    help="Pilih jumlah klaster risiko untuk algoritma K-Means."
)

df_clustered = run_kmeans(df_prov, k=k_val)

# Pilihan Metrik Utama
METRIC_OPTIONS = {
    'Luas Karhutla (ha)': 'Luas_Karhutla_2025',
    'Emisi CO2 (ton)': 'Emisi_CO2_2025',
    'Total Hotspot (titik)': 'Total_Hotspot_2025',
    'Intensitas Emisi (ton/ha)': 'Intensitas_Emisi'
}
selected_metric_label = st.sidebar.radio(
    "Pilihan Metrik Utama",
    options=list(METRIC_OPTIONS.keys()),
    index=1
)
selected_metric_col = METRIC_OPTIONS[selected_metric_label]

# Multiselect Zona Risiko
available_zones = sorted(df_clustered['Status_Kerawanan'].unique().tolist())
selected_zones = st.sidebar.multiselect(
    "Filter Zona Risiko",
    options=available_zones,
    default=available_zones
)

# Multiselect Provinsi
all_provinces = sorted(df_clustered['Provinsi'].unique().tolist())
selected_provinces = st.sidebar.multiselect(
    "Filter Provinsi",
    options=all_provinces,
    default=[],
    placeholder="Semua Provinsi (atau pilih spesifik)"
)

# Tombol Reset Filter
if st.sidebar.button("Reset Filter", use_container_width=True):
    st.rerun()

st.sidebar.markdown("<div style='height: 1px; background: rgba(255,255,255,0.08); margin: 18px 0;'></div>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style="font-size: 0.78rem; color: #9A8B84; line-height: 1.6;">
    <span style="font-family: 'Manrope', sans-serif; color: #FF8A2B; font-weight: 700;">PARAMETER AKTIF:</span><br>
    • Cakupan: 38 Provinsi Indonesia<br>
    • Algoritma: K-Means (log1p + StandardScaler)<br>
    • Klasterisasi: {k_val} Kategori Risiko
</div>
""", unsafe_allow_html=True)

# Terapkan Filter
df_f = df_clustered.copy()
if selected_zones:
    df_f = df_f[df_f['Status_Kerawanan'].isin(selected_zones)]
if selected_provinces:
    df_f = df_f[df_f['Provinsi'].isin(selected_provinces)]


# ==============================================================================
# 8. TOP NAVIGATION BAR (GLASS PANEL)
# ==============================================================================
st.markdown("""
<div class="glass-main-view">
    <div class="top-nav-panel">
        <div class="nav-brand">
            <div class="brand-icon-box">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FF8A2B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="9"/>
                    <circle cx="12" cy="12" r="5"/>
                    <circle cx="12" cy="12" r="1.5" fill="#FF8A2B"/>
                </svg>
            </div>
            <span class="brand-title">Karhutla Analytics</span>
        </div>
        <div class="data-badge">
            <span class="data-dot"></span>
            Data 2025
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 9. HERO SECTION
# ==============================================================================
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-title-row">
        <h1 class="hero-title">Situasi Karhutla Nasional</h1>
        <span class="year-badge">2025</span>
    </div>
    <div class="hero-subtitle">
        Sistem Pemantauan Terpadu, Analisis Jejak Karbon & Klasterisasi Risiko Kebakaran Hutan dan Lahan di 38 Provinsi Indonesia
    </div>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 10. BARIS KPI NASIONAL (4 KARTU GLASS DENGAN SPARKLINE TOP 8)
# ==============================================================================
total_luas = df_f['Luas_Karhutla_2025'].sum()
total_emisi = df_f['Emisi_CO2_2025'].sum()
total_hotspot = df_f['Total_Hotspot_2025'].sum()
bahaya_count = (df_f['Status_Kerawanan'] == 'Bahaya').sum()
total_prov_f = len(df_f)

# Data Top 8 untuk Sparkline Masing-Masing Metrik
spark_vals_luas = df_f['Luas_Karhutla_2025'].sort_values(ascending=False).head(8).tolist()
spark_vals_emisi = df_f['Emisi_CO2_2025'].sort_values(ascending=False).head(8).tolist()
spark_vals_hotspot = df_f['Total_Hotspot_2025'].sort_values(ascending=False).head(8).tolist()
spark_vals_bahaya = df_f[df_f['Status_Kerawanan'] == 'Bahaya']['Emisi_CO2_2025'].sort_values(ascending=False).head(8).tolist()
if not spark_vals_bahaya:
    spark_vals_bahaya = [0] * 8

svg_spark_luas = render_sparkline_bars(spark_vals_luas)
svg_spark_emisi = render_sparkline_bars(spark_vals_emisi)
svg_spark_hotspot = render_sparkline_bars(spark_vals_hotspot)
svg_spark_bahaya = render_sparkline_bars(spark_vals_bahaya)

icon_area = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FF8A2B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>"""
icon_emission = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FF8A2B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>"""
icon_hotspot = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FF8A2B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><line x1="12" y1="3" x2="12" y2="7"/><line x1="12" y1="17" x2="12" y2="21"/><line x1="3" y1="12" x2="7" y2="12"/><line x1="17" y1="12" x2="21" y2="12"/></svg>"""
icon_danger = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FF8A2B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""

pct_bahaya = (bahaya_count / total_prov_f * 100) if total_prov_f > 0 else 0.0

kpi_html = f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-label">Total Luas Terbakar</span>
            <div class="kpi-icon-box">{icon_area}</div>
        </div>
        <div class="kpi-number">{format_id(total_luas, 2)}</div>
        <div class="kpi-unit">Hektar (ha) • {format_compact_id(total_luas, 'ha')}</div>
        <div class="kpi-sparkline-row">
            <div>{svg_spark_luas}</div>
            <span class="kpi-sparkline-note">Top 8 wilayah</span>
        </div>
        <div class="kpi-caption">Akumulasi wilayah terpilih</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-label">Total Emisi CO2</span>
            <div class="kpi-icon-box">{icon_emission}</div>
        </div>
        <div class="kpi-number">{format_id(total_emisi, 0)}</div>
        <div class="kpi-unit">Ton CO2 • {format_compact_id(total_emisi, 'ton')}</div>
        <div class="kpi-sparkline-row">
            <div>{svg_spark_emisi}</div>
            <span class="kpi-sparkline-note">Top 8 wilayah</span>
        </div>
        <div class="kpi-caption">Estimasi pelepasan karbon</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-label">Total Hotspot</span>
            <div class="kpi-icon-box">{icon_hotspot}</div>
        </div>
        <div class="kpi-number">{format_id(total_hotspot, 0)}</div>
        <div class="kpi-unit">Titik Panas • {format_compact_id(total_hotspot, 'titik')}</div>
        <div class="kpi-sparkline-row">
            <div>{svg_spark_hotspot}</div>
            <span class="kpi-sparkline-note">Top 8 wilayah</span>
        </div>
        <div class="kpi-caption">Deteksi sensor satelit</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-label">Provinsi Zona Bahaya</span>
            <div class="kpi-icon-box">{icon_danger}</div>
        </div>
        <div class="kpi-number">{bahaya_count} dari {total_prov_f}</div>
        <div class="kpi-unit">Klasifikasi K-Means • {format_id(pct_bahaya, 1)}%</div>
        <div class="kpi-sparkline-row">
            <div>{svg_spark_bahaya}</div>
            <span class="kpi-sparkline-note">Top 8 zona bahaya</span>
        </div>
        <div class="kpi-caption">Kategori risiko tinggi</div>
    </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)


# ==============================================================================
# 11. KARTU INSIGHT OTOMATIS
# ==============================================================================
if not df_f.empty:
    top_emisi_row = df_f.loc[df_f['Emisi_CO2_2025'].idxmax()]
    top_intensitas_row = df_f.loc[df_f['Intensitas_Emisi'].idxmax()]

    corr_df = hitung_korelasi(df_f)
    r_luas_emisi = corr_df.loc['Luas_Karhutla_2025', 'Emisi_CO2_2025'] if not corr_df.empty else 0.0
    r_hotspot_emisi = corr_df.loc['Total_Hotspot_2025', 'Emisi_CO2_2025'] if not corr_df.empty else 0.0

    narasi_luas = "sangat kuat" if r_luas_emisi >= 0.70 else ("moderat" if r_luas_emisi >= 0.40 else "lemah")
    narasi_hotspot = "hampir tidak berkorelasi" if abs(r_hotspot_emisi) < 0.30 else ("moderat" if abs(r_hotspot_emisi) < 0.70 else "kuat")

    insight_html = f"""
    <div class="quiet-insight-card">
        <span class="insight-badge-label">RINGKASAN ANALISIS DATA AKTIF</span>
        <div class="insight-body-quiet">
            • Emisi CO2 tertinggi tercatat di <b>{top_emisi_row['Provinsi']}</b> sebesar <b>{format_id(top_emisi_row['Emisi_CO2_2025'], 0)} ton CO2</b>.<br>
            • Intensitas emisi tertinggi tercatat di <b>{top_intensitas_row['Provinsi']}</b> sebesar <b>{format_id(top_intensitas_row['Intensitas_Emisi'], 1)} ton/ha</b>.<br>
            • Berdasarkan klasterisasi K-Means, sebanyak <b>{bahaya_count} provinsi</b> ({format_id(pct_bahaya, 1)}%) tergolong dalam <b>Zona Bahaya</b>.<br>
            • Korelasi Pearson antara Luas Karhutla dan Emisi adalah <b>r = {r_luas_emisi:.2f} ({narasi_luas})</b>, sedangkan Hotspot terhadap Emisi adalah <b>r = {r_hotspot_emisi:.2f} ({narasi_hotspot})</b>. Nilai ini menunjukkan bahwa luasan area terbakar berkorelasi erat dengan emisi, sementara jumlah hotspot tidak berkorelasi langsung dengan pelepasan karbon.
        </div>
    </div>
    """
    st.markdown(insight_html, unsafe_allow_html=True)


# ==============================================================================
# 12. LIMA TABS UTAMA (NAMA & URUTAN TETAP SAMA PERSIS)
# ==============================================================================
tab_peta, tab_ranking, tab_zona, tab_korelasi, tab_data = st.tabs([
    "Peta Sebaran",
    "Ranking",
    "Zona Risiko",
    "Korelasi",
    "Data"
])

# ------------------------------------------------------------------------------
# TAB 1: PETA SEBARAN
# ------------------------------------------------------------------------------
with tab_peta:
    st.markdown("""
    <div class="section-header">
        <div class="section-index-line">
            <span class="section-badge">01</span>
            <div class="section-line"></div>
        </div>
        <h2 class="section-title">Peta Sebaran Spasial</h2>
        <p class="section-desc">Distribusi geografis 38 provinsi di Indonesia berdasarkan indikator terpilih dan zona risiko.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_f.empty:
        st.warning("Tidak ada data provinsi yang sesuai dengan filter yang dipilih.")
    else:
        # 6. Koordinat check: cek keberadaan koordinat di PROV_COORDS
        missing_coords = [p for p in df_f['Provinsi'] if p not in PROV_COORDS]
        if missing_coords:
            st.markdown(
                f'<div style="font-family: \'Inter\', sans-serif; font-size: 0.80rem; color: #F04438; margin-bottom: 6px;">'
                f'{len(missing_coords)} provinsi tidak ditampilkan (koordinat tidak tersedia): {", ".join(missing_coords)}'
                f'</div>',
                unsafe_allow_html=True
            )

        # Filter data hanya yang memiliki koordinat valid
        df_map = df_f[df_f['Provinsi'].isin(PROV_COORDS)].copy()
        df_map['lat'] = df_map['Provinsi'].map(lambda p: PROV_COORDS[p][0])
        df_map['lon'] = df_map['Provinsi'].map(lambda p: PROV_COORDS[p][1])

        # 7. Sinkronisasi baris info di atas peta
        n_plotted = len(df_map)
        hint_filter = " Ubah Filter Zona Risiko di sidebar untuk melihat semua provinsi." if n_plotted < 38 else ""
        st.markdown(
            f'<div style="font-family: \'Inter\', sans-serif; font-size: 0.82rem; color: #9A8B84; margin-bottom: 2px;">'
            f'Menampilkan {n_plotted} dari 38 provinsi.{hint_filter}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="font-family: \'Inter\', sans-serif; font-size: 0.78rem; color: #9A8B84; margin-bottom: 12px;">'
            f'Ukuran titik sebanding dengan akar kuadrat dari {selected_metric_label}; titik terkecil berarti nilai 0.'
            f'</div>',
            unsafe_allow_html=True
        )

        # 2. Hitung ukuran titik: size = 10 + sqrt(max(val, 0)) / max_sqrt * 26
        raw_vals = np.maximum(df_map[selected_metric_col].fillna(0).values, 0)
        sqrt_vals = np.sqrt(raw_vals)
        max_sqrt = sqrt_vals.max() if len(sqrt_vals) > 0 and sqrt_vals.max() > 0 else 1.0
        df_map['Ukuran_Titik'] = 10.0 + (sqrt_vals / max_sqrt) * 26.0

        # 5. Label nama provinsi: hanya untuk Top 5 provinsi dengan nilai indikator terpilih > 0
        df_gt_zero = df_map[df_map[selected_metric_col] > 0]
        if not df_gt_zero.empty:
            top5_provs = set(
                df_gt_zero.sort_values(
                    by=[selected_metric_col, 'Provinsi'],
                    ascending=[False, True]
                ).head(5)['Provinsi'].tolist()
            )
        else:
            top5_provs = set()

        df_map['Label_Peta'] = df_map['Provinsi'].apply(lambda p: p if p in top5_provs else '')

        # 3 & 4. Format hovertext seragam untuk semua titik
        def build_hover(row):
            prov = row['Provinsi']
            zona = row['Status_Kerawanan']
            luas = row['Luas_Karhutla_2025']
            emisi = row['Emisi_CO2_2025']
            hotspot = row['Total_Hotspot_2025']
            intensitas = row['Intensitas_Emisi']
            metric_val = row[selected_metric_col]

            if pd.isna(luas) or luas == 0:
                intensitas_str = "0 ton/ha (tidak ada luasan terbakar)"
            else:
                intensitas_str = f"{format_id(intensitas, 2)} ton/ha"

            lines = [
                f"<b>{prov}</b>",
                f"Zona Risiko: {zona}",
                f"Luas Karhutla: {format_id(luas, 2)} ha",
                f"Emisi CO2: {format_id(emisi, 0)} ton CO2",
                f"Total Hotspot: {format_id(hotspot, 0)} titik",
                f"Intensitas Emisi: {intensitas_str}"
            ]
            if pd.isna(metric_val) or metric_val <= 0:
                lines.append("Nilai 0 — kemungkinan tidak ada kejadian atau data belum tercatat")

            return "<br>".join(lines)

        df_map['Hover_Text'] = df_map.apply(build_hover, axis=1)

        # 1. Gambar peta dengan go.Scattermap / go.Scattermapbox per Status_Kerawanan
        scatter_go_cls = go.Scattermap if hasattr(go, 'Scattermap') else go.Scattermapbox
        fig_map = go.Figure()

        # Urutan zona yang rapi
        zone_order_ref = ['Bahaya', 'Siaga', 'Waspada', 'Rendah', 'Aman']
        present_zones = [z for z in zone_order_ref if z in df_f['Status_Kerawanan'].values]
        for z in df_f['Status_Kerawanan'].unique():
            if z not in present_zones:
                present_zones.append(z)

        for zone in present_zones:
            # Hitung jumlah provinsi di df_f untuk zona ini
            count_in_dff = int((df_f['Status_Kerawanan'] == zone).sum())
            legend_title = f"{zone} ({count_in_dff})"
            df_zone = df_map[df_map['Status_Kerawanan'] == zone]
            color = ZONE_COLORS.get(zone, '#FF8A2B')

            fig_map.add_trace(scatter_go_cls(
                lat=df_zone['lat'].tolist(),
                lon=df_zone['lon'].tolist(),
                mode='markers+text',
                marker=dict(
                    size=df_zone['Ukuran_Titik'].tolist(),
                    sizemode='diameter',
                    color=color,
                    opacity=0.85
                ),
                text=df_zone['Label_Peta'].tolist(),
                textposition='top right',
                textfont=dict(
                    size=11,
                    color='#F4EDE8',
                    family='Inter, sans-serif'
                ),
                hovertext=df_zone['Hover_Text'].tolist(),
                hoverinfo='text',
                name=legend_title
            ))

        # 8. Tata letak & styling peta
        if hasattr(go, 'Scattermap'):
            fig_map.update_layout(
                map=dict(
                    style='carto-darkmatter',
                    center=dict(lat=-2.2, lon=118.0),
                    zoom=3.5
                ),
                height=560,
                margin=dict(l=0, r=0, t=10, b=10)
            )
        else:
            fig_map.update_layout(
                mapbox=dict(
                    style='carto-darkmatter',
                    center=dict(lat=-2.2, lon=118.0),
                    zoom=3.5
                ),
                height=560,
                margin=dict(l=0, r=0, t=10, b=10)
            )

        fig_map = style_plotly(fig_map)
        fig_map.update_layout(
            legend=dict(
                bgcolor="rgba(20, 13, 11, 0.65)",
                bordercolor="rgba(255, 255, 255, 0.08)",
                borderwidth=1,
                font=dict(color="#F4EDE8", family="Inter, sans-serif", size=11)
            ),
            margin=dict(l=0, r=0, t=10, b=10)
        )
        st.plotly_chart(fig_map, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: RANKING
# ------------------------------------------------------------------------------
with tab_ranking:
    st.markdown("""
    <div class="section-header">
        <div class="section-index-line">
            <span class="section-badge">02</span>
            <div class="section-line"></div>
        </div>
        <h2 class="section-title">Peringkat Wilayah & Intensitas Emisi</h2>
        <p class="section-desc">Analisis komparatif sepuluh provinsi teratas berdasarkan indikator utama dan densitas emisi per hektar.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_f.empty:
        st.warning("Tidak ada data provinsi yang sesuai dengan filter yang dipilih.")
    else:
        col_rank1, col_rank2 = st.columns([1.1, 0.9], gap="medium")

        with col_rank1:
            df_top_metric = df_f.sort_values(selected_metric_col, ascending=True).tail(10).copy()
            df_top_metric['Label_Val'] = df_top_metric[selected_metric_col].apply(
                lambda v: format_id(v, 2 if selected_metric_col in ['Luas_Karhutla_2025', 'Intensitas_Emisi'] else 0)
            )
            df_top_metric['Fill_Color'] = df_top_metric['Status_Kerawanan'].map(lambda s: hex_to_rgba(ZONE_COLORS.get(s, '#FF8A2B'), 0.18))
            df_top_metric['Line_Color'] = df_top_metric['Status_Kerawanan'].map(lambda s: ZONE_COLORS.get(s, '#FF8A2B'))

            fig_rank = go.Figure(go.Bar(
                x=df_top_metric[selected_metric_col],
                y=df_top_metric['Provinsi'],
                orientation='h',
                marker=dict(
                    color=df_top_metric['Fill_Color'],
                    line=dict(color=df_top_metric['Line_Color'], width=1.5)
                ),
                text=df_top_metric['Label_Val'],
                textposition='outside',
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>" + selected_metric_label + ": %{x:,.2f}<extra></extra>"
            ))
            fig_rank = style_plotly(fig_rank)
            fig_rank.update_layout(title=f"Top 10 Provinsi — {selected_metric_label}")
            st.plotly_chart(fig_rank, use_container_width=True)

        with col_rank2:
            df_top_int = df_f.sort_values('Intensitas_Emisi', ascending=True).tail(10).copy()
            df_top_int['Label_Int'] = df_top_int['Intensitas_Emisi'].apply(lambda v: f"{format_id(v, 1)} ton/ha")
            df_top_int['Fill_Color'] = 'rgba(255, 138, 43, 0.18)'
            df_top_int['Line_Color'] = '#FF8A2B'

            fig_int = go.Figure(go.Bar(
                x=df_top_int['Intensitas_Emisi'],
                y=df_top_int['Provinsi'],
                orientation='h',
                marker=dict(
                    color=df_top_int['Fill_Color'],
                    line=dict(color=df_top_int['Line_Color'], width=1.5)
                ),
                text=df_top_int['Label_Int'],
                textposition='outside',
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Intensitas Emisi: %{x:,.1f} ton/ha<extra></extra>"
            ))
            fig_int = style_plotly(fig_int)
            fig_int.update_layout(title="Top 10 Intensitas Emisi (Ton CO2 / Ha)")
            st.plotly_chart(fig_int, use_container_width=True)

        df_anomali = cek_data_tidak_lengkap(df_f)
        if not df_anomali.empty:
            anomali_items = []
            for _, r in df_anomali.iterrows():
                alasan = []
                if r['Luas_Karhutla_2025'] > 0 and r['Total_Hotspot_2025'] == 0:
                    alasan.append("hotspot 0 titik namun luas tercatat > 0 ha")
                elif r['Luas_Karhutla_2025'] == 0 and r['Total_Hotspot_2025'] > 0:
                    alasan.append("terdeteksi hotspot namun luas tercatat 0 ha")
                elif r['Emisi_CO2_2025'] == 0 and r['Luas_Karhutla_2025'] > 0:
                    alasan.append("emisi 0 namun luas tercatat > 0 ha")
                detail = f"<b>{r['Provinsi']}</b> ({', '.join(alasan)})"
                anomali_items.append(detail)

            anomali_text = " • ".join(anomali_items)
            st.markdown(f"""
            <div style="font-size: 0.82rem; color: #9A8B84; margin-top: 10px; padding: 10px 14px; background: rgba(255, 255, 255, 0.035); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; line-height: 1.5;">
                <span style="color: #FF8A2B; font-weight: 600;">Catatan kelengkapan data:</span> {anomali_text}
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: ZONA RISIKO (K-MEANS)
# ------------------------------------------------------------------------------
with tab_zona:
    st.markdown("""
    <div class="section-header">
        <div class="section-index-line">
            <span class="section-badge">03</span>
            <div class="section-line"></div>
        </div>
        <h2 class="section-title">Pemodelan Zona Risiko (K-Means)</h2>
        <p class="section-desc">Klasterisasi multivariat fitur terstandardisasi dengan pengurutan tingkat kerawanan berbasis emisi CO2.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_f.empty:
        st.warning("Tidak ada data provinsi yang sesuai dengan filter yang dipilih.")
    else:
        col_scatter, col_donut = st.columns([1.25, 0.75], gap="medium")

        df_scatter = df_f.copy()
        df_scatter['Luas_Plot'] = df_scatter['Luas_Karhutla_2025'].apply(lambda x: max(float(x), 1.0))
        df_scatter['Emisi_Plot'] = df_scatter['Emisi_CO2_2025'].apply(lambda x: max(float(x), 1.0))

        max_hotspot_sqrt = np.sqrt(df_scatter['Total_Hotspot_2025'].max()) if df_scatter['Total_Hotspot_2025'].max() > 0 else 1.0
        df_scatter['Bubble_Size'] = df_scatter['Total_Hotspot_2025'].apply(
            lambda x: 6.0 + (np.sqrt(max(float(x), 0.0)) / max_hotspot_sqrt) * 24.0
        )

        df_scatter['Format_Luas'] = df_scatter['Luas_Karhutla_2025'].apply(lambda v: f"{format_id(v, 2)} ha")
        df_scatter['Format_Emisi'] = df_scatter['Emisi_CO2_2025'].apply(lambda v: f"{format_id(v, 0)} ton")
        df_scatter['Format_Hotspot'] = df_scatter['Total_Hotspot_2025'].apply(lambda v: f"{format_id(v, 0)} titik")

        with col_scatter:
            fig_cluster = px.scatter(
                df_scatter,
                x='Luas_Plot',
                y='Emisi_Plot',
                size='Bubble_Size',
                color='Status_Kerawanan',
                color_discrete_map=ZONE_COLORS,
                hover_name='Provinsi',
                hover_data={
                    'Luas_Plot': False,
                    'Emisi_Plot': False,
                    'Bubble_Size': False,
                    'Status_Kerawanan': True,
                    'Format_Luas': True,
                    'Format_Emisi': True,
                    'Format_Hotspot': True
                },
                labels={
                    'Luas_Plot': 'Luas Karhutla (ha) [Skala Log]',
                    'Emisi_Plot': 'Emisi CO2 (ton) [Skala Log]',
                    'Format_Luas': 'Luas Karhutla',
                    'Format_Emisi': 'Emisi CO2',
                    'Format_Hotspot': 'Total Hotspot',
                    'Status_Kerawanan': 'Zona Risiko'
                },
                log_x=True,
                log_y=True,
                title="Sebaran Klaster: Luas Karhutla vs Emisi CO2 (Skala Log)"
            )
            fig_cluster.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color='rgba(255,255,255,0.35)')))
            fig_cluster = style_plotly(fig_cluster)
            st.plotly_chart(fig_cluster, use_container_width=True)

        with col_donut:
            zone_counts = df_f['Status_Kerawanan'].value_counts().reset_index()
            zone_counts.columns = ['Status_Kerawanan', 'Jumlah']
            order_key = {label: idx for idx, label in enumerate(CLUSTER_LABEL_MAP.get(k_val, []))}
            zone_counts['sort_idx'] = zone_counts['Status_Kerawanan'].map(lambda s: order_key.get(s, 99))
            zone_counts = zone_counts.sort_values('sort_idx').drop(columns=['sort_idx'])

            fig_donut = px.pie(
                zone_counts,
                values='Jumlah',
                names='Status_Kerawanan',
                hole=0.65,
                color='Status_Kerawanan',
                color_discrete_map=ZONE_COLORS,
                title="Proporsi Jumlah Provinsi per Zona"
            )
            fig_donut = style_plotly(fig_donut)
            fig_donut.update_traces(
                textinfo='label+value',
                marker=dict(line=dict(color='#0A0706', width=2))
            )
            fig_donut.update_layout(
                showlegend=False,
                annotations=[dict(
                    text=f"<b>{len(df_f)}</b><br><span style='font-size:11px; color:#9A8B84;'>Provinsi</span>",
                    x=0.5, y=0.5,
                    font=dict(size=18, family="Manrope", color="#F4EDE8"),
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("""
        <div style="font-family: 'Manrope', sans-serif; font-size: 1.05rem; font-weight: 700; color: #F4EDE8; margin: 20px 0 8px 0;">
            Karakteristik Rata-Rata per Zona Risiko
        </div>
        """, unsafe_allow_html=True)

        summary_df = df_f.groupby('Status_Kerawanan').agg(
            Jumlah_Provinsi=('Provinsi', 'count'),
            Rerata_Luas_ha=('Luas_Karhutla_2025', 'mean'),
            Rerata_Emisi_ton=('Emisi_CO2_2025', 'mean'),
            Rerata_Hotspot_titik=('Total_Hotspot_2025', 'mean'),
            Rerata_Intensitas_ton_ha=('Intensitas_Emisi', 'mean')
        ).reset_index()

        summary_df['sort_idx'] = summary_df['Status_Kerawanan'].map(lambda s: order_key.get(s, 99))
        summary_df = summary_df.sort_values('sort_idx').drop(columns=['sort_idx'])

        display_summary = summary_df.copy()
        display_summary['Rerata_Luas_ha'] = display_summary['Rerata_Luas_ha'].apply(lambda x: format_id(x, 2))
        display_summary['Rerata_Emisi_ton'] = display_summary['Rerata_Emisi_ton'].apply(lambda x: format_id(x, 0))
        display_summary['Rerata_Hotspot_titik'] = display_summary['Rerata_Hotspot_titik'].apply(lambda x: format_id(x, 0))
        display_summary['Rerata_Intensitas_ton_ha'] = display_summary['Rerata_Intensitas_ton_ha'].apply(lambda x: format_id(x, 2))

        display_summary.columns = [
            'Zona Risiko', 'Jumlah Provinsi', 'Rata-rata Luas (ha)',
            'Rata-rata Emisi (ton CO2)', 'Rata-rata Hotspot (titik)', 'Rata-rata Intensitas (ton/ha)'
        ]
        st.dataframe(display_summary, use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="font-family: 'Manrope', sans-serif; font-size: 1.05rem; font-weight: 700; color: #F4EDE8; margin: 20px 0 10px 0;">
            Daftar Provinsi Berdasarkan Kategori Risiko
        </div>
        """, unsafe_allow_html=True)

        for status in summary_df['Status_Kerawanan'].unique():
            prov_in_zone = sorted(df_f[df_f['Status_Kerawanan'] == status]['Provinsi'].tolist())
            zone_col = ZONE_COLORS.get(status, '#FF8A2B')
            badges = "".join([
                f"""<span style="display:inline-flex; align-items:center; background:rgba(255,255,255,0.04); color:#F4EDE8; border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:3px 9px; font-size:0.8rem; margin:2px 3px;">
                    <span style="width:6px; height:6px; background:{zone_col}; border-radius:50%; margin-right:6px;"></span>{p}
                </span>"""
                for p in prov_in_zone
            ])
            st.markdown(f"""
            <div style="margin-bottom: 10px; background: rgba(255, 255, 255, 0.035); padding: 10px 14px; border-radius: 8px; border-left: 3px solid {zone_col};">
                <span style="font-family: 'Manrope', sans-serif; font-size: 0.8rem; font-weight: 700; color: {zone_col}; margin-right: 10px;">{status.upper()} ({len(prov_in_zone)}):</span>
                {badges if badges else '<span style="color:#9A8B84; font-size:0.8rem;">Tidak ada provinsi</span>'}
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 4: KORELASI & ANALISIS
# ------------------------------------------------------------------------------
with tab_korelasi:
    st.markdown("""
    <div class="section-header">
        <div class="section-index-line">
            <span class="section-badge">04</span>
            <div class="section-line"></div>
        </div>
        <h2 class="section-title">Analisis Korelasi Pearson</h2>
        <p class="section-desc">Pengukuran relasi linear objektif antar variabel kuantitatif karhutla tanpa inferensi kausal spekulatif.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_f.empty or len(df_f) < 2:
        st.warning("Diperlukan minimal 2 provinsi untuk menghitung matriks korelasi.")
    else:
        corr_matrix = hitung_korelasi(df_f)
        r_lh = corr_matrix.loc['Luas_Karhutla_2025', 'Total_Hotspot_2025']
        r_le = corr_matrix.loc['Luas_Karhutla_2025', 'Emisi_CO2_2025']
        r_he = corr_matrix.loc['Total_Hotspot_2025', 'Emisi_CO2_2025']

        col_corr1, col_corr2 = st.columns([1, 1], gap="medium")

        with col_corr1:
            custom_labels = ['Luas Karhutla', 'Emisi CO2', 'Total Hotspot']
            fig_heat = px.imshow(
                corr_matrix.values,
                x=custom_labels,
                y=custom_labels,
                color_continuous_scale=CONTINUOUS_COLORSCALE,
                zmin=-0.2,
                zmax=1.0,
                text_auto='.2f',
                title="Matriks Korelasi Pearson (r)"
            )
            fig_heat = style_plotly(fig_heat)
            fig_heat.update_traces(textfont=dict(size=14, family="Manrope"))
            st.plotly_chart(fig_heat, use_container_width=True)

        with col_corr2:
            fig_scatter_he = px.scatter(
                df_f,
                x='Total_Hotspot_2025',
                y='Emisi_CO2_2025',
                color='Status_Kerawanan',
                color_discrete_map=ZONE_COLORS,
                hover_name='Provinsi',
                hover_data={
                    'Total_Hotspot_2025': ':,',
                    'Emisi_CO2_2025': ':,',
                    'Luas_Karhutla_2025': ':.2f'
                },
                labels={
                    'Total_Hotspot_2025': 'Total Hotspot (titik)',
                    'Emisi_CO2_2025': 'Emisi CO2 (ton)',
                    'Status_Kerawanan': 'Zona Risiko'
                },
                title="Sebaran Titik: Total Hotspot vs Emisi CO2"
            )
            fig_scatter_he.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color='rgba(255,255,255,0.35)')))
            fig_scatter_he = style_plotly(fig_scatter_he)
            st.plotly_chart(fig_scatter_he, use_container_width=True)

        narasi_hotspot_emisi = "hampir tidak berkorelasi" if abs(r_he) < 0.30 else ("berkorelasi lemah hingga sedang" if abs(r_he) < 0.70 else "berkorelasi kuat")
        narasi_luas_emisi = "sangat kuat" if r_le >= 0.70 else ("sedang" if r_le >= 0.40 else "lemah")

        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.035); border: 1px solid rgba(255, 255, 255, 0.08); border-left: 3px solid #FF8A2B; border-radius: 12px; padding: 18px 20px; margin-top: 10px;">
            <div style="font-family: 'Manrope', sans-serif; font-size: 0.78rem; font-weight: 700; color: #FF8A2B; margin-bottom: 8px; letter-spacing: 0.5px; text-transform: uppercase;">
                Catatan Metodologi & Interpretasi Statistik
            </div>
            <div style="font-size: 0.9rem; color: #9A8B84; line-height: 1.65;">
                • <b style="color:#F4EDE8;">Luas Karhutla vs Emisi CO2 (r = {r_le:.2f} — {narasi_luas}):</b> Luasan areal terbakar berkorelasi linear positif secara konsisten dengan volume pelepasan karbon.<br>
                • <b style="color:#F4EDE8;">Total Hotspot vs Emisi CO2 (r = {r_he:.2f} — {narasi_hotspot_emisi}):</b> Angka r mendekati nol menunjukkan bahwa jumlah titik hotspot {narasi_hotspot_emisi} dengan besaran emisi, sehingga kuantitas hotspot tidak dapat diposisikan sebagai penyebab langsung besaran emisi.<br>
                • <b style="color:#F4EDE8;">Konteks Pengukuran:</b> Deteksi hotspot oleh satelit mengukur anomali termal sesaat pada tutupan lahan, sedangkan emisi aktual ditentukan oleh kedalaman bahan bakar organik dan tipe vegetasi yang terbakar.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 5: DATA INTERAKTIF & UNDUH CSV
# ------------------------------------------------------------------------------
with tab_data:
    st.markdown("""
    <div class="section-header">
        <div class="section-index-line">
            <span class="section-badge">05</span>
            <div class="section-line"></div>
        </div>
        <h2 class="section-title">Dataset Lengkap & Ekspor</h2>
        <p class="section-desc">Tabel data komprehensif seluruh provinsi beserta status kerawanan dan fasilitas unduh CSV.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_f.empty:
        st.warning("Tidak ada data provinsi yang sesuai dengan filter yang dipilih.")
    else:
        df_tabel = df_f[[
            'Provinsi', 'Status_Kerawanan', 'Luas_Karhutla_2025',
            'Emisi_CO2_2025', 'Total_Hotspot_2025', 'Intensitas_Emisi'
        ]].copy()

        st.dataframe(
            df_tabel,
            use_container_width=True,
            column_config={
                'Provinsi': st.column_config.TextColumn('Provinsi', width='medium'),
                'Status_Kerawanan': st.column_config.TextColumn('Status Kerawanan', width='small'),
                'Luas_Karhutla_2025': st.column_config.NumberColumn('Luas Karhutla (ha)', format="%.2f"),
                'Emisi_CO2_2025': st.column_config.NumberColumn('Emisi CO2 (ton)', format="%d"),
                'Total_Hotspot_2025': st.column_config.NumberColumn('Total Hotspot (titik)', format="%d"),
                'Intensitas_Emisi': st.column_config.NumberColumn('Intensitas Emisi (ton/ha)', format="%.2f")
            },
            hide_index=True
        )

        col_dl, col_space = st.columns([1, 2])
        with col_dl:
            csv_data = df_f.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Data CSV",
                data=csv_data,
                file_name="analisis_karhutla_2025.csv",
                mime="text/csv",
                use_container_width=True
            )