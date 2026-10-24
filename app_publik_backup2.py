import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Peringatan Dini Karhutla & ISPA", layout="wide", page_icon="🔥")

# --- 2. FUNGSI MEMUAT DATA ---
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
    
    # 1. Normalisasi nama entitas
    df['Provinsi'] = df['Provinsi'].astype(str).str.strip().replace({
        'Dki Jakarta': 'DKI Jakarta',
        'Ikn': 'IKN'
    })
    
    # 2. Hapus entitas 'Total' dan 'IKN' dari tabel utama
    df = df[~df['Provinsi'].isin(['Total', 'IKN'])].copy()
    
    # Konversi kolom numerik dan validasi tidak ada NaN pada data utama
    for col in ['Tahun', 'Luas_Karhutla', 'Emisi_CO2', 'Total_Hotspot']:
        df[col] = pd.to_numeric(df[col])
    if df[required_cols].isna().any().any():
        raise ValueError("Terdapat nilai NaN pada kolom utama setelah menghapus entitas Total dan IKN.")
    
    # 3. Label status data:
    # - 'Belum tercatat': 4 provinsi baru Papua pada 2018-2022 (Luas=0, Emisi=0, Hotspot=0, total 20 baris)
    # - 'Hotspot meragukan': Total_Hotspot == 0 dan Luas_Karhutla > 0 (total 9 baris)
    # - 'Normal': baris lainnya
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
    
    # 4. Intensitas_Emisi = Emisi / Luas; bernilai NaN jika Luas == 0
    df['Intensitas_Emisi'] = np.where(df['Luas_Karhutla'] > 0, df['Emisi_CO2'] / df['Luas_Karhutla'], np.nan)
    
    return df

# --- 3. FUNGSI MACHINE LEARNING 1: K-MEANS (KLASTERISASI TINGKAT KARHUTLA) ---
def jalankan_kmeans(df_tahun_terpilih, k=3):
    df_cluster = df_tahun_terpilih.copy()
    
    # Fitur Luas_Karhutla dan Emisi_CO2 di-log1p lalu di-standardize
    X_raw = df_cluster[['Luas_Karhutla', 'Emisi_CO2']].values
    X_log = np.log1p(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Urutkan label klaster berdasarkan rata-rata emisi secara ascending
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
    """
    Menghasilkan ringkasan per Tingkat_Karhutla:
    - Jumlah provinsi
    - Emisi_CO2 minimum dan maksimum
    - Luas_Karhutla median
    """
    ringkasan = df_klaster.groupby('Tingkat_Karhutla').agg(
        Jumlah_Provinsi=('Provinsi', 'count'),
        Emisi_Min=('Emisi_CO2', 'min'),
        Emisi_Maks=('Emisi_CO2', 'max'),
        Luas_Median=('Luas_Karhutla', 'median')
    ).reset_index()
    ringkasan = ringkasan.sort_values('Emisi_Min', ascending=True).reset_index(drop=True)
    return ringkasan

# --- 4. FUNGSI MACHINE LEARNING 2: REGRESI & PROYEKSI SKENARIO EMISI ---
def prediksi_emisi_2026(df, provinsi_pilihan):
    # Ambil data historis khusus provinsi yang dipilih dari subset data valid
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
    """
    Rolling backtest untuk tahun target 2023, 2024, 2025.
    Memerlukan minimal 5 tahun data historis sebelum tahun target.
    Membandingkan: Linear Regression (dipotong pada 0), Median historis, dan Nilai tahun sebelumnya.
    """
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
            
        # Metode 1: Regresi linear (dipotong pada 0 jika minus)
        reg = LinearRegression().fit(df_train[['Tahun']].values, df_train['Emisi_CO2'].values)
        pred_linear = max(0.0, float(reg.predict([[t]])[0]))
        
        # Metode 2: Median historis
        pred_median = float(df_train['Emisi_CO2'].median())
        
        # Metode 3: Nilai tahun sebelumnya
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
    """
    Menghitung proyeksi lengkap dan menentukan metode utama berdasarkan evaluasi data.
    """
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

# --- 5. ANTARMUKA PENGGUNA (UI) STREAMLIT ---

# Memuat data master dengan penanganan kesalahan
try:
    df_master = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# Subset data valid (mengecualikan 'Belum tercatat')
df_valid = df_master[df_master['Status_Data'] != 'Belum tercatat'].copy()

# Judul Aplikasi
st.title("🔥 KACC: Public Health Warning & Proyeksi Skenario Karhutla")
st.markdown("Sistem peringatan dini risiko ISPA berbasis data Karhutla historis (2018-2025) dan proyeksi skenario emisi masa depan menggunakan *Machine Learning*.")
st.markdown("---")

# Sidebar untuk Filter
st.sidebar.header("Pusat Kendali Publik")
tahun_pilihan = st.sidebar.slider("Pilih Tahun Historis", 2018, 2025, 2025)
provinsi_list = sorted(df_valid['Provinsi'].unique())
provinsi_pilihan = st.sidebar.selectbox("Pilih Provinsi untuk Proyeksi Skenario", provinsi_list)

# Menjalankan K-Means untuk tahun terpilih (menggunakan df_valid)
df_tahun_ini = df_valid[df_valid['Tahun'] == tahun_pilihan].copy()
if not df_tahun_ini.empty:
    df_klaster = jalankan_kmeans(df_tahun_ini)
else:
    df_klaster = pd.DataFrame(columns=['Provinsi', 'Luas_Karhutla', 'Emisi_CO2', 'Tingkat_Karhutla'])

# Menjalankan Proyeksi Skenario (Regresi Linear) untuk provinsi terpilih (menggunakan df_valid)
proyeksi_emisi, df_prov_historis, model_regresi = prediksi_emisi_2026(df_valid, provinsi_pilihan)

# --- TAMPILAN UTAMA ---

# Bagian 1: Status Tingkat Karhutla (K-Means)
st.subheader(f"Status Tingkat Karhutla Nasional ({tahun_pilihan})")
status_provinsi = df_klaster[df_klaster['Provinsi'] == provinsi_pilihan]['Tingkat_Karhutla'].values
if len(status_provinsi) > 0:
    st.info(f"**Tingkat Karhutla {provinsi_pilihan} ({tahun_pilihan}):** {status_provinsi[0]}")
else:
    st.warning("Data tidak tersedia untuk provinsi dan tahun ini.")

# Tampilkan Peta atau Tabel Sederhana
st.dataframe(df_klaster[['Provinsi', 'Luas_Karhutla', 'Emisi_CO2', 'Tingkat_Karhutla']].sort_values('Emisi_CO2', ascending=False), use_container_width=True)

st.markdown("---")

# Bagian 2: Proyeksi Skenario Emisi 2026 (Regresi Linear)
st.subheader(f"Proyeksi Skenario Emisi Karbon 2026: {provinsi_pilihan}")

# Layout kolom untuk menampilkan angka proyeksi
col1, col2 = st.columns(2)
with col1:
    emisi_terakhir = df_prov_historis[df_prov_historis['Tahun'] == 2025]['Emisi_CO2'].values[0] if len(df_prov_historis[df_prov_historis['Tahun'] == 2025]) > 0 else 0
    st.metric(label="Emisi Aktual (2025)", value=f"{emisi_terakhir:,.0f} Ton")

with col2:
    # Hitung selisih tren
    delta = proyeksi_emisi - emisi_terakhir
    st.metric(label="Proyeksi Skenario Emisi (2026)", value=f"{proyeksi_emisi:,.0f} Ton", delta=f"{delta:,.0f} Ton dari 2025", delta_color="inverse")

# Membuat Grafik Tren & Garis Proyeksi Skenario menggunakan Plotly
# Gabungkan data historis dan data proyeksi skenario untuk grafik
df_grafik = df_prov_historis[['Tahun', 'Emisi_CO2']].copy()
df_proyeksi = pd.DataFrame({'Tahun': [2026], 'Emisi_CO2': [proyeksi_emisi]})
df_grafik_lengkap = pd.concat([df_grafik, df_proyeksi], ignore_index=True)

# Plotting grafik garis
fig = px.line(df_grafik_lengkap, x='Tahun', y='Emisi_CO2', markers=True, 
              title=f"Tren Emisi Historis dan Proyeksi Skenario 2026 - {provinsi_pilihan}")

# Menandai titik proyeksi skenario tahun 2026 dengan warna beda
fig.add_scatter(x=[2026], y=[proyeksi_emisi], mode='markers', marker=dict(color='red', size=10), name='Proyeksi Skenario 2026')

st.plotly_chart(fig, use_container_width=True)
