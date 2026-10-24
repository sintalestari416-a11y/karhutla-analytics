import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Pantauan Karhutla Indonesia",
    layout="wide"
)

# --- 2. KOORDINAT PUSAT PROVINSI ---
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

TIER_COLORS = {
    'Sangat rendah': '#2E8B6A',
    'Rendah': '#4CC38A',
    'Sedang': '#FFC247',
    'Tinggi': '#F04438',
    'Sangat tinggi': '#B3261E'
}

# --- 3. FUNGSI MEMUAT DATA ---
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA_MASTER_2018_2025.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File data tidak ditemukan: {file_path}")
    
    df = pd.read_csv(file_path)
    
    required_cols = ['Provinsi', 'Tahun', 'Luas_Karhutla', 'Emisi_CO2', 'Total_Hotspot']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak lengkap: {missing_cols}")
    
    df['Provinsi'] = df['Provinsi'].astype(str).str.strip().replace({
        'Dki Jakarta': 'DKI Jakarta',
        'Ikn': 'IKN'
    })
    
    df = df[~df['Provinsi'].isin(['Total', 'IKN'])].copy()
    
    for col in ['Tahun', 'Luas_Karhutla', 'Emisi_CO2', 'Total_Hotspot']:
        df[col] = pd.to_numeric(df[col])
    if df[required_cols].isna().any().any():
        raise ValueError("Terdapat nilai NaN pada kolom utama setelah menghapus entitas Total dan IKN.")
    
    papua_baru = ['Papua Barat Daya', 'Papua Pegunungan', 'Papua Selatan', 'Papua Tengah']
    cond_belum_tercatat = (
        df['Provinsi'].isin(papua_baru) &
        df['Tahun'].between(2018, 2022) &
        (df['Luas_Karhutla'] == 0) &
        (df['Emisi_CO2'] == 0) &
        (df['Total_Hotspot'] == 0)
    )
    cond_hotspot_meragukan = (df['Total_Hotspot'] == 0) & (df['Luas_Karhutla'] > 0)
    
    df['Status_Data'] = np.select(
        [cond_belum_tercatat, cond_hotspot_meragukan],
        ['Belum tercatat', 'Hotspot meragukan'],
        default='Normal'
    )
    
    df['Intensitas_Emisi'] = np.where(df['Luas_Karhutla'] > 0, df['Emisi_CO2'] / df['Luas_Karhutla'], np.nan)
    
    return df

# --- 4. FUNGSI MACHINE LEARNING 1: K-MEANS ---
def jalankan_kmeans(df_tahun_terpilih, k=3):
    df_cluster = df_tahun_terpilih.copy()
    
    X_raw = df_cluster[['Luas_Karhutla', 'Emisi_CO2']].values
    X_log = np.log1p(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
    
    urutan = df_cluster.groupby('Cluster')['Emisi_CO2'].mean().sort_values().index.tolist()
    
    label_map_by_k = {
        2: ['Rendah', 'Tinggi'],
        3: ['Rendah', 'Sedang', 'Tinggi'],
        4: ['Rendah', 'Sedang', 'Tinggi', 'Sangat tinggi'],
        5: ['Sangat rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat tinggi']
    }
    labels = label_map_by_k.get(k, [f'Klaster {i+1}' for i in range(k)])
    mapping = {cluster_id: label for cluster_id, label in zip(urutan, labels)}
    
    df_cluster['Tingkat_Karhutla'] = df_cluster['Cluster'].map(mapping)
    return df_cluster

def ringkasan_zona(df_klaster):
    ringkasan = df_klaster.groupby('Tingkat_Karhutla').agg(
        Jumlah_Provinsi=('Provinsi', 'count'),
        Emisi_Min=('Emisi_CO2', 'min'),
        Emisi_Maks=('Emisi_CO2', 'max'),
        Luas_Median=('Luas_Karhutla', 'median')
    ).reset_index()
    ringkasan = ringkasan.sort_values('Emisi_Min', ascending=True).reset_index(drop=True)
    return ringkasan

# --- 5. FUNGSI MACHINE LEARNING 2: REGRESI & PROYEKSI SKENARIO EMISI ---
def prediksi_emisi_2026(df, provinsi_pilihan):
    if 'Status_Data' in df.columns:
        df_prov = df[(df['Provinsi'] == provinsi_pilihan) & (df['Status_Data'] != 'Belum tercatat')].copy()
    else:
        df_prov = df[df['Provinsi'] == provinsi_pilihan].copy()
        
    df_prov = df_prov.sort_values('Tahun').reset_index(drop=True)
    
    if len(df_prov) == 0:
        return 0.0, df_prov, None
        
    X = df_prov[['Tahun']].values
    y = df_prov['Emisi_CO2'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    proyeksi_2026 = float(model.predict([[2026]])[0])
    nilai = max(0.0, proyeksi_2026)
    return nilai, df_prov, model

def evaluasi_proyeksi(df_prov):
    if 'Status_Data' in df_prov.columns:
        df_p = df_prov[df_prov['Status_Data'] != 'Belum tercatat'].copy()
    else:
        df_p = df_prov.copy()
        
    df_p = df_p.sort_values('Tahun').reset_index(drop=True)
    target_years = [2023, 2024, 2025]
    records = []
    
    for t in target_years:
        aktual_rows = df_p[df_p['Tahun'] == t]
        if aktual_rows.empty:
            continue
        aktual = float(aktual_rows['Emisi_CO2'].values[0])
        
        df_train = df_p[df_p['Tahun'] < t]
        if len(df_train) < 5:
            continue
            
        reg = LinearRegression().fit(df_train[['Tahun']].values, df_train['Emisi_CO2'].values)
        pred_linear = max(0.0, float(reg.predict([[t]])[0]))
        pred_median = float(df_train['Emisi_CO2'].median())
        pred_last = float(df_train.sort_values('Tahun').iloc[-1]['Emisi_CO2'])
        
        records.append({
            'Tahun': t,
            'Aktual': aktual,
            'Linear': pred_linear,
            'Median': pred_median,
            'Tahun_Terakhir': pred_last
        })
        
    df_eval = pd.DataFrame(records, columns=['Tahun', 'Aktual', 'Linear', 'Median', 'Tahun_Terakhir'])
    
    if len(df_eval) > 0:
        mae_linear = float(np.mean(np.abs(df_eval['Aktual'] - df_eval['Linear'])))
        mae_median = float(np.mean(np.abs(df_eval['Aktual'] - df_eval['Median'])))
        mae_last = float(np.mean(np.abs(df_eval['Aktual'] - df_eval['Tahun_Terakhir'])))
        maes = {
            'Linear': mae_linear,
            'Median': mae_median,
            'Tahun_Terakhir': mae_last,
            'linear': mae_linear,
            'median': mae_median,
            'tahun_terakhir': mae_last
        }
    else:
        maes = {
            'Linear': np.nan,
            'Median': np.nan,
            'Tahun_Terakhir': np.nan,
            'linear': np.nan,
            'median': np.nan,
            'tahun_terakhir': np.nan
        }
        
    return df_eval, maes

def proyeksi_lengkap(df_prov):
    if 'Status_Data' in df_prov.columns:
        df_p = df_prov[df_prov['Status_Data'] != 'Belum tercatat'].copy()
    else:
        df_p = df_prov.copy()
        
    df_p = df_p.sort_values('Tahun').reset_index(drop=True)
    n_tahun = len(df_p)
    
    if n_tahun == 0:
        return {
            'linear_mentah': None,
            'linear': None,
            'median': None,
            'rata2': None,
            'min': None,
            'maks': None,
            'n_tahun': 0,
            'dipotong': False,
            'metode_utama': None,
            'alasan': 'Data historis belum cukup (minimal 5 tahun)'
        }
        
    median_val = float(df_p['Emisi_CO2'].median())
    rata2_val = float(df_p['Emisi_CO2'].mean())
    min_val = float(df_p['Emisi_CO2'].min())
    maks_val = float(df_p['Emisi_CO2'].max())
    
    if n_tahun >= 2:
        reg = LinearRegression().fit(df_p[['Tahun']].values, df_p['Emisi_CO2'].values)
        linear_mentah = float(reg.predict([[2026]])[0])
        dipotong = bool(linear_mentah < 0)
        linear_val = max(0.0, linear_mentah)
    else:
        linear_mentah = None
        dipotong = False
        linear_val = None
        
    if n_tahun < 5:
        metode_utama = None
        alasan = 'Data historis belum cukup (minimal 5 tahun)'
    else:
        _, maes = evaluasi_proyeksi(df_p)
        mae_linear = maes.get('linear', np.nan)
        mae_median = maes.get('median', np.nan)
        
        linear_kurang_akurat = (
            not np.isnan(mae_linear) and not np.isnan(mae_median) and mae_linear >= mae_median
        )
        if (linear_mentah is not None and linear_mentah < 0) or linear_kurang_akurat:
            metode_utama = 'median'
            alasan = 'Skenario dasar: median historis'
        else:
            metode_utama = 'linear'
            alasan = 'Skenario tren: regresi linear'
            
    return {
        'linear_mentah': linear_mentah,
        'linear': linear_val,
        'median': median_val,
        'rata2': rata2_val,
        'min': min_val,
        'maks': maks_val,
        'n_tahun': n_tahun,
        'dipotong': dipotong,
        'metode_utama': metode_utama,
        'alasan': alasan
    }

# --- 6. GAYA TAMPILAN (CSS) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #F4EDE8;
}

h1, h2, h3, h4, h5, h6, .brand-title, .metric-value {
    font-family: 'Manrope', sans-serif !important;
    font-variant-numeric: tabular-nums;
}

#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

.ambient-glow {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
    background: 
        radial-gradient(circle at top left, rgba(255, 122, 26, 0.14) 0%, transparent 45%),
        radial-gradient(circle at top right, rgba(255, 122, 26, 0.14) 0%, transparent 45%),
        radial-gradient(circle at bottom left, rgba(255, 122, 26, 0.14) 0%, transparent 45%),
        radial-gradient(circle at bottom right, rgba(255, 122, 26, 0.14) 0%, transparent 45%);
}

.glass-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 16px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 20px;
    margin-bottom: 20px;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px;
}

section[data-testid="stSidebar"] {
    background-color: #140D0B !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

main .block-container {
    animation: fadeIn 0.4s ease-out;
}

@media (prefers-reduced-motion: reduce) {
    main .block-container {
        animation: none;
    }
}
</style>
<div class="ambient-glow"></div>
""", unsafe_allow_html=True)

# --- 7. HERO SECTION ---
st.markdown("""
<div class="glass-card" style="padding: 14px 20px; margin-bottom: 24px; display: flex; align-items: center; justify-content: space-between;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="14" cy="14" r="12" stroke="#FF8A2B" stroke-width="1.8" stroke-dasharray="3 3"/>
            <circle cx="14" cy="14" r="7.5" stroke="#F4EDE8" stroke-width="1.8"/>
            <circle cx="14" cy="14" r="3" fill="#FF8A2B"/>
        </svg>
        <span style="font-family: 'Manrope', sans-serif; font-size: 1.15rem; font-weight: 800; color: #F4EDE8; letter-spacing: 0.06em;">KACC</span>
    </div>
    <div style="background: rgba(255, 138, 43, 0.12); color: #FF8A2B; border: 1px solid rgba(255, 138, 43, 0.3); padding: 4px 14px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.02em;">
        Data 2018–2025
    </div>
</div>

<h1 style="font-family: 'Manrope', sans-serif; font-size: 2.2rem; font-weight: 800; color: #F4EDE8; margin-bottom: 8px; letter-spacing: -0.02em;">Pantauan Karhutla Indonesia</h1>
<p style="color: #9A8B84; font-size: 1.05rem; line-height: 1.55; margin-bottom: 20px; max-width: 900px;">
Memanfaatkan data luas lahan terbakar dan emisi CO2 periode 2018–2025 untuk menilai skala kebakaran hutan dan lahan per provinsi serta menyajikan proyeksi skenario masa depan.
</p>

<div class="glass-card" style="border-left: 3px solid rgba(255, 255, 255, 0.2); padding: 14px 18px; margin-bottom: 28px;">
    <div style="color: #9A8B84; font-size: 0.875rem; line-height: 1.5;">
        Tingkat ini berdasarkan luas lahan terbakar dan emisi CO2, bukan pengukuran kualitas udara atau data kasus ISPA. Untuk kondisi udara dan kesehatan, rujuk informasi resmi BMKG, KLHK, dan dinas kesehatan setempat.
    </div>
</div>
""", unsafe_allow_html=True)

# --- 8. SIDEBAR KONTROL ---
st.sidebar.markdown("""
<div style="font-family: 'Manrope', sans-serif; font-size: 1.15rem; font-weight: 700; color: #F4EDE8; margin-bottom: 16px;">
Pusat Kendali Publik
</div>
""", unsafe_allow_html=True)

tahun_pilihan = st.sidebar.slider("Pilih Tahun Historis", 2018, 2025, 2025)
provinsi_list = sorted(list(PROV_COORDS.keys()))
provinsi_pilihan = st.sidebar.selectbox("Pilih Provinsi", provinsi_list)
k_clusters = st.sidebar.slider("Jumlah tingkat (k)", 2, 5, 3)

# Memuat data
try:
    df_master = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

df_valid = df_master[df_master['Status_Data'] != 'Belum tercatat'].copy()

# Data klaster tahun terpilih
df_tahun_ini = df_valid[df_valid['Tahun'] == tahun_pilihan].copy()
if not df_tahun_ini.empty:
    df_klaster = jalankan_kmeans(df_tahun_ini, k=k_clusters)
else:
    df_klaster = pd.DataFrame(columns=['Provinsi', 'Luas_Karhutla', 'Emisi_CO2', 'Total_Hotspot', 'Status_Data', 'Tingkat_Karhutla'])

# --- 9. BLOK 1: PERINGATAN DINI ---
st.markdown("""
<h2 style="font-family: 'Manrope', sans-serif; font-size: 1.45rem; font-weight: 700; color: #F4EDE8; margin-top: 10px; margin-bottom: 14px;">
Peringatan Dini Tingkat Karhutla
</h2>
""", unsafe_allow_html=True)

row_target = df_master[(df_master['Provinsi'] == provinsi_pilihan) & (df_master['Tahun'] == tahun_pilihan)]
is_belum_tercatat = row_target.empty or (row_target['Status_Data'].values[0] == 'Belum tercatat')

if is_belum_tercatat:
    st.markdown(f"""
    <div class="glass-card">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
            <div style="font-family: 'Manrope', sans-serif; font-size: 1.25rem; font-weight: 700; color: #F4EDE8;">
                {provinsi_pilihan} • Tahun {tahun_pilihan}
            </div>
            <span style="background: rgba(154, 139, 132, 0.15); color: #9A8B84; border: 1px solid rgba(154, 139, 132, 0.3); padding: 4px 14px; border-radius: 999px; font-size: 0.85rem; font-weight: 600;">
                Belum tercatat
            </span>
        </div>
        <div style="color: #9A8B84; font-size: 0.95rem;">
            Provinsi ini belum tercatat pada data tahun tersebut.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    row_klaster = df_klaster[df_klaster['Provinsi'] == provinsi_pilihan]
    if not row_klaster.empty:
        tingkat_val = row_klaster['Tingkat_Karhutla'].values[0]
        badge_col = TIER_COLORS.get(tingkat_val, '#FF8A2B')
        luas_val = row_klaster['Luas_Karhutla'].values[0]
        emisi_val = row_klaster['Emisi_CO2'].values[0]
        hotspot_val = row_klaster['Total_Hotspot'].values[0]
        status_d = row_klaster['Status_Data'].values[0]
        
        if status_d == 'Hotspot meragukan':
            hotspot_str = "Data hotspot belum lengkap"
        else:
            hotspot_str = f"{int(hotspot_val):,} titik"
            
        rank_val = int((df_klaster['Emisi_CO2'] > emisi_val).sum() + 1)
        total_p = len(df_klaster)
        rank_str = f"{rank_val} dari {total_p} provinsi"
        
        prev_row = df_valid[(df_valid['Provinsi'] == provinsi_pilihan) & (df_valid['Tahun'] == tahun_pilihan - 1)]
        if prev_row.empty:
            delta_str = "tidak dibandingkan"
        else:
            prev_emisi = float(prev_row['Emisi_CO2'].values[0])
            if prev_emisi == 0:
                delta_str = "tidak dibandingkan"
            else:
                pct_chg = ((emisi_val - prev_emisi) / prev_emisi) * 100
                sign = "+" if pct_chg > 0 else ""
                delta_str = f"{sign}{pct_chg:,.1f}%"
                
        st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
                <div>
                    <div style="font-family: 'Manrope', sans-serif; font-size: 1.35rem; font-weight: 700; color: #F4EDE8;">
                        {provinsi_pilihan} • Tahun {tahun_pilihan}
                    </div>
                    <div style="color: #9A8B84; font-size: 0.85rem; margin-top: 2px;">
                        Status Tingkat Karhutla Nasional
                    </div>
                </div>
                <span style="background: {badge_col}22; color: {badge_col}; border: 1px solid {badge_col}66; padding: 6px 16px; border-radius: 999px; font-family: 'Manrope', sans-serif; font-size: 0.95rem; font-weight: 700;">
                    {tingkat_val}
                </span>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 14px;">
                <div style="background: rgba(255,255,255,0.02); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="color: #9A8B84; font-size: 0.8rem;">Luas terbakar</div>
                    <div style="font-family: 'Manrope', sans-serif; font-size: 1.15rem; font-weight: 700; color: #F4EDE8; margin-top: 4px;">{luas_val:,.2f} ha</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="color: #9A8B84; font-size: 0.8rem;">Emisi CO2</div>
                    <div style="font-family: 'Manrope', sans-serif; font-size: 1.15rem; font-weight: 700; color: #F4EDE8; margin-top: 4px;">{emisi_val:,.0f} ton</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="color: #9A8B84; font-size: 0.8rem;">Total hotspot</div>
                    <div style="font-family: 'Manrope', sans-serif; font-size: 1.15rem; font-weight: 700; color: #F4EDE8; margin-top: 4px;">{hotspot_str}</div>
                </div>
                <div style="background: rgba(255,255,255,0.02); padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="color: #9A8B84; font-size: 0.8rem;">Peringkat emisi</div>
                    <div style="font-family: 'Manrope', sans-serif; font-size: 1.15rem; font-weight: 700; color: #F4EDE8; margin-top: 4px;">{rank_str}</div>
                </div>
            </div>
            
            <div style="color: #9A8B84; font-size: 0.85rem; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 10px;">
                Perubahan dibanding tahun sebelumnya: <span style="color: #F4EDE8; font-weight: 600;">{delta_str}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Tabs
tab_peta, tab_tabel, tab_peringkat = st.tabs(["Peta", "Tabel", "Peringkat"])

with tab_peta:
    df_peta = df_klaster[df_klaster['Provinsi'].isin(PROV_COORDS)].copy()
    df_peta['Lat'] = df_peta['Provinsi'].map(lambda p: PROV_COORDS[p][0])
    df_peta['Lon'] = df_peta['Provinsi'].map(lambda p: PROV_COORDS[p][1])
    
    st.caption(f"Menampilkan {len(df_peta)} dari {len(PROV_COORDS)} provinsi")
    
    max_emisi_peta = df_peta['Emisi_CO2'].max() if len(df_peta) > 0 else 0
    sqrt_max = np.sqrt(max_emisi_peta) if max_emisi_peta > 0 else 1.0
    
    fig_peta = go.Figure()
    
    ordered_tiers = ['Sangat rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat tinggi']
    present_tiers = [t for t in ordered_tiers if t in df_peta['Tingkat_Karhutla'].unique()]
    for t in df_peta['Tingkat_Karhutla'].unique():
        if t not in present_tiers:
            present_tiers.append(t)
            
    for tier in present_tiers:
        sub = df_peta[df_peta['Tingkat_Karhutla'] == tier]
        if sub.empty:
            continue
            
        tier_sizes = [
            10.0 + 26.0 * (np.sqrt(max(0.0, float(val))) / sqrt_max)
            for val in sub['Emisi_CO2']
        ]
        
        hotspot_texts = [
            "Data hotspot belum lengkap" if s == 'Hotspot meragukan' else f"{int(h):,} titik"
            for s, h in zip(sub['Status_Data'], sub['Total_Hotspot'])
        ]
        
        custom_data = np.stack((
            sub['Provinsi'],
            sub['Tingkat_Karhutla'],
            [f"{v:,.2f}" for v in sub['Luas_Karhutla']],
            [f"{v:,.0f}" for v in sub['Emisi_CO2']],
            hotspot_texts
        ), axis=-1)
        
        fig_peta.add_trace(go.Scattermap(
            lat=sub['Lat'],
            lon=sub['Lon'],
            mode='markers',
            marker=dict(
                size=tier_sizes,
                color=TIER_COLORS.get(tier, '#FF8A2B'),
                opacity=0.88
            ),
            name=tier,
            customdata=custom_data,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Tingkat: %{customdata[1]}<br>"
                "Luas Karhutla: %{customdata[2]} ha<br>"
                "Emisi CO2: %{customdata[3]} ton<br>"
                "Total Hotspot: %{customdata[4]}"
                "<extra></extra>"
            )
        ))
        
    fig_peta.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=-2.2, lon=118.0),
            zoom=3.5
        ),
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F4EDE8"),
        legend=dict(
            title=dict(text="Tingkat", font=dict(family="Manrope, sans-serif", size=12)),
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(10, 7, 6, 0.75)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1
        )
    )
    st.plotly_chart(fig_peta, width="stretch")

with tab_tabel:
    df_tabel = df_klaster[['Provinsi', 'Tingkat_Karhutla', 'Luas_Karhutla', 'Emisi_CO2', 'Total_Hotspot']].copy()
    max_emisi_val = float(df_tabel['Emisi_CO2'].max()) if len(df_tabel) > 0 and df_tabel['Emisi_CO2'].max() > 0 else 1.0
    st.dataframe(
        df_tabel.sort_values('Emisi_CO2', ascending=False),
        column_config={
            "Provinsi": st.column_config.TextColumn("Provinsi"),
            "Tingkat_Karhutla": st.column_config.TextColumn("Tingkat"),
            "Luas_Karhutla": st.column_config.NumberColumn("Luas Terbakar (ha)", format="%,.2f ha"),
            "Emisi_CO2": st.column_config.ProgressColumn("Emisi CO2 (ton)", min_value=0, max_value=max_emisi_val, format="%,.0f ton"),
            "Total_Hotspot": st.column_config.NumberColumn("Total Hotspot", format="%,.0f titik")
        },
        hide_index=True,
        width="stretch"
    )

with tab_peringkat:
    df_top10 = df_klaster.sort_values('Emisi_CO2', ascending=False).head(10).copy()
    df_top10 = df_top10.sort_values('Emisi_CO2', ascending=True)
    
    fig_top10 = go.Figure()
    bar_colors = [TIER_COLORS.get(t, '#FF8A2B') for t in df_top10['Tingkat_Karhutla']]
    
    fig_top10.add_trace(go.Bar(
        x=df_top10['Emisi_CO2'],
        y=df_top10['Provinsi'],
        orientation='h',
        marker=dict(color=bar_colors, line=dict(color="rgba(255, 255, 255, 0.15)", width=1)),
        customdata=np.stack((
            df_top10['Tingkat_Karhutla'],
            [f"{v:,.2f}" for v in df_top10['Luas_Karhutla']],
            [f"{v:,.0f}" for v in df_top10['Emisi_CO2']]
        ), axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Tingkat: %{customdata[0]}<br>"
            "Emisi CO2: %{customdata[2]} ton<br>"
            "Luas: %{customdata[1]} ha"
            "<extra></extra>"
        )
    ))
    
    fig_top10.update_layout(
        title=dict(
            text="Top 10 Provinsi Berdasarkan Emisi CO2",
            font=dict(family="Manrope, sans-serif", size=15, color="#F4EDE8")
        ),
        xaxis=dict(
            title="Emisi CO2 (ton)",
            gridcolor="rgba(255, 255, 255, 0.08)",
            color="#9A8B84",
            tickformat=","
        ),
        yaxis=dict(
            title="",
            color="#F4EDE8",
            tickfont=dict(family="Inter, sans-serif", size=12)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F4EDE8"),
        margin=dict(l=10, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_top10, width="stretch")

# Ringkasan Zona
df_ring = ringkasan_zona(df_klaster)
df_ring_display = pd.DataFrame({
    'Tingkat': df_ring['Tingkat_Karhutla'],
    'Jumlah Provinsi': df_ring['Jumlah_Provinsi'],
    'Rentang Emisi (ton)': [f"{row['Emisi_Min']:,.0f} – {row['Emisi_Maks']:,.0f}" for _, row in df_ring.iterrows()],
    'Median Luas (ha)': [f"{row['Luas_Median']:,.2f}" for _, row in df_ring.iterrows()]
})
st.dataframe(df_ring_display, hide_index=True, width="stretch")
st.caption("Tingkat bersifat relatif antar provinsi pada tahun yang dipilih.")

st.markdown("<hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 32px 0;'>", unsafe_allow_html=True)

# --- 10. BLOK 2: PROYEKSI EMISI 2026 ---
st.markdown(f"""
<h2 style="font-family: 'Manrope', sans-serif; font-size: 1.45rem; font-weight: 700; color: #F4EDE8; margin-top: 10px; margin-bottom: 14px;">
Proyeksi Skenario Emisi Karbon 2026: {provinsi_pilihan}
</h2>
""", unsafe_allow_html=True)

df_prov_valid = df_valid[df_valid['Provinsi'] == provinsi_pilihan].sort_values('Tahun').reset_index(drop=True)
info_proyeksi = proyeksi_lengkap(df_prov_valid)

# Kartu 1: Emisi aktual 2025
row_2025 = df_prov_valid[df_prov_valid['Tahun'] == 2025]
if not row_2025.empty:
    emisi_2025_val = float(row_2025['Emisi_CO2'].values[0])
    emisi_2025_str = f"{emisi_2025_val:,.0f} ton"
else:
    emisi_2025_str = "Data belum tersedia"

# Kartu 2: Proyeksi skenario 2026
if info_proyeksi['metode_utama'] is None:
    proyeksi_2026_str = "Data historis belum cukup"
    proyeksi_2026_sub = "Minimal 5 tahun data historis"
else:
    if info_proyeksi['metode_utama'] == 'median':
        val_proj = info_proyeksi['median']
        proyeksi_2026_sub = "Median historis"
    else:
        val_proj = info_proyeksi['linear']
        proyeksi_2026_sub = "Tren linear"
    proyeksi_2026_str = f"{val_proj:,.0f} ton"

# Kartu 3: Rentang historis (min–maks 2018–2025)
if info_proyeksi['min'] is not None and info_proyeksi['maks'] is not None:
    rentang_str = f"{info_proyeksi['min']:,.0f} – {info_proyeksi['maks']:,.0f} ton"
else:
    rentang_str = "Data belum cukup"
rentang_sub = "Min – Maks (2018–2025)"

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(f"""
    <div class="glass-card" style="margin-bottom: 0;">
        <div style="color: #9A8B84; font-size: 0.85rem; font-weight: 500; margin-bottom: 6px;">Emisi Aktual (2025)</div>
        <div style="font-family: 'Manrope', sans-serif; font-size: 1.45rem; font-weight: 700; color: #F4EDE8; font-variant-numeric: tabular-nums;">{emisi_2025_str}</div>
        <div style="color: #9A8B84; font-size: 0.75rem; margin-top: 4px;">Data tercatat KLHK</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
    <div class="glass-card" style="margin-bottom: 0;">
        <div style="color: #9A8B84; font-size: 0.85rem; font-weight: 500; margin-bottom: 6px;">Proyeksi Skenario (2026)</div>
        <div style="font-family: 'Manrope', sans-serif; font-size: 1.45rem; font-weight: 700; color: #FF8A2B; font-variant-numeric: tabular-nums;">{proyeksi_2026_str}</div>
        <div style="color: #9A8B84; font-size: 0.75rem; margin-top: 4px;">{proyeksi_2026_sub}</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""
    <div class="glass-card" style="margin-bottom: 0;">
        <div style="color: #9A8B84; font-size: 0.85rem; font-weight: 500; margin-bottom: 6px;">Rentang Historis</div>
        <div style="font-family: 'Manrope', sans-serif; font-size: 1.45rem; font-weight: 700; color: #F4EDE8; font-variant-numeric: tabular-nums;">{rentang_str}</div>
        <div style="color: #9A8B84; font-size: 0.75rem; margin-top: 4px;">{rentang_sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Grafik Proyeksi
fig_proy = go.Figure()

if info_proyeksi['min'] is not None and info_proyeksi['maks'] is not None:
    fig_proy.add_hrect(
        y0=info_proyeksi['min'],
        y1=info_proyeksi['maks'],
        fillcolor="rgba(255, 138, 43, 0.06)",
        line_width=0,
        annotation_text="Rentang historis (min–maks)",
        annotation_position="top left",
        annotation_font=dict(color="#9A8B84", size=10)
    )

fig_proy.add_trace(go.Scatter(
    x=df_prov_valid['Tahun'],
    y=df_prov_valid['Emisi_CO2'],
    mode='lines+markers',
    name='Historis',
    line=dict(color='#FF8A2B', width=2.5),
    marker=dict(size=8, color='#FF8A2B'),
    hovertemplate="Tahun %{x}<br>Emisi: %{y:,.0f} ton<extra></extra>"
))

if info_proyeksi['metode_utama'] is not None:
    val_proj = info_proyeksi['median'] if info_proyeksi['metode_utama'] == 'median' else info_proyeksi['linear']
    last_year = int(df_prov_valid['Tahun'].max())
    last_emisi = float(df_prov_valid[df_prov_valid['Tahun'] == last_year]['Emisi_CO2'].values[0])
    
    fig_proy.add_trace(go.Scatter(
        x=[last_year, 2026],
        y=[last_emisi, val_proj],
        mode='lines',
        name='Arah Skenario',
        line=dict(color='#FFC247', width=2, dash='dash'),
        hoverinfo='skip'
    ))
    
    fig_proy.add_trace(go.Scatter(
        x=[2026],
        y=[val_proj],
        mode='markers+text',
        name='Skenario 2026',
        marker=dict(size=12, color='#FFC247', symbol='circle', line=dict(color='#0A0706', width=2)),
        text=['2026'],
        textposition='top center',
        hovertemplate="Proyeksi Skenario 2026<br>Emisi: %{y:,.0f} ton<extra></extra>"
    ))

fig_proy.update_layout(
    title=dict(
        text=f"Tren Historis dan Proyeksi Skenario Emisi CO2 - {provinsi_pilihan}",
        font=dict(family="Manrope, sans-serif", size=15, color="#F4EDE8")
    ),
    xaxis=dict(
        title="Tahun",
        tickmode='linear',
        tick0=2018,
        dtick=1,
        gridcolor="rgba(255, 255, 255, 0.08)",
        color="#9A8B84"
    ),
    yaxis=dict(
        title="Emisi CO2 (ton)",
        gridcolor="rgba(255, 255, 255, 0.08)",
        color="#9A8B84",
        tickformat=","
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#F4EDE8"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(10, 7, 6, 0.75)",
        bordercolor="rgba(255, 255, 255, 0.1)",
        borderwidth=1
    ),
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig_proy, width="stretch")

# Expander Evaluasi
with st.expander("Seberapa andal proyeksi ini?"):
    df_eval, maes = evaluasi_proyeksi(df_prov_valid)
    if len(df_eval) > 0:
        st.markdown("<p style='color: #9A8B84; font-size: 0.9rem;'>Hasil evaluasi backtest bergulir (rolling) membandingkan akurasi metode terhadap data aktual tahun 2023–2025:</p>", unsafe_allow_html=True)
        st.dataframe(
            df_eval,
            column_config={
                "Tahun": st.column_config.NumberColumn("Tahun", format="%d"),
                "Aktual": st.column_config.NumberColumn("Aktual (ton)", format="%,.0f ton"),
                "Linear": st.column_config.NumberColumn("Linear (ton)", format="%,.0f ton"),
                "Median": st.column_config.NumberColumn("Median (ton)", format="%,.0f ton"),
                "Tahun_Terakhir": st.column_config.NumberColumn("Tahun Terakhir (ton)", format="%,.0f ton")
            },
            hide_index=True,
            width="stretch"
        )
        
        kesimpulan_lines = []
        if info_proyeksi['linear_mentah'] is not None and info_proyeksi['linear_mentah'] < 0:
            kesimpulan_lines.append("Tren linear menghasilkan nilai negatif sehingga dipotong menjadi 0 dan tidak dipakai.")
        
        if not np.isnan(maes['linear']) and not np.isnan(maes['median']):
            if maes['linear'] < maes['median']:
                kesimpulan_lines.append(f"Metode Regresi Linear memiliki rata-rata kesalahan absolut (MAE {maes['linear']:,.0f} ton) yang lebih rendah dibandingkan Median Historis (MAE {maes['median']:,.0f} ton).")
            else:
                kesimpulan_lines.append(f"Metode Median Historis memiliki rata-rata kesalahan absolut (MAE {maes['median']:,.0f} ton) yang lebih rendah atau setara dibandingkan Regresi Linear (MAE {maes['linear']:,.0f} ton), sehingga lebih andal sebagai skenario dasar.")
        
        for k_line in kesimpulan_lines:
            st.markdown(f"<p style='color: #F4EDE8; font-size: 0.9rem; margin-top: 6px;'>• {k_line}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #9A8B84;'>Data historis belum mencukupi untuk melakukan evaluasi backtest (diperlukan minimal 5 tahun data sebelum tahun target).</p>", unsafe_allow_html=True)
        if info_proyeksi['linear_mentah'] is not None and info_proyeksi['linear_mentah'] < 0:
            st.markdown("<p style='color: #F4EDE8;'>• Tren linear menghasilkan nilai negatif sehingga dipotong menjadi 0 dan tidak dipakai.</p>", unsafe_allow_html=True)

# Fixed Info Box
st.markdown("""
<div class="glass-card" style="margin-top: 15px; border-left: 3px solid #FF8A2B;">
    <div style="color: #F4EDE8; font-size: 0.9rem; line-height: 1.5;">
        Ini proyeksi skenario, bukan prediksi resmi. Data tahun 2026 belum tersedia. Emisi karhutla sangat dipengaruhi tahun ekstrem (pada data ini 2019 dan 2023).
    </div>
</div>
""", unsafe_allow_html=True)