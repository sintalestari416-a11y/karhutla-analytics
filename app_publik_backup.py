import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Peringatan Dini Karhutla & ISPA", layout="wide", page_icon="🔥")

# --- 2. FUNGSI MEMUAT DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('DATA_MASTER_2018_2025.csv')
    df.fillna(0, inplace=True) 
    
    # 🔴 TAMBAHKAN BARIS INI UNTUK MEMBUANG BARIS "TOTAL" 
    df = df[df['Provinsi'] != 'Total']
    
    # Hitung intensitas emisi. Jika luas 0, intensitas 0 agar tidak error (dibagi nol)
    df['Intensitas_Emisi'] = np.where(df['Luas_Karhutla'] > 0, df['Emisi_CO2'] / df['Luas_Karhutla'], 0)
    return df

# --- 3. FUNGSI MACHINE LEARNING 1: K-MEANS (KLASTERISASI KESEHATAN) ---
def jalankan_kmeans(df_tahun_terpilih, k=3):
    # Kita kelompokkan berdasarkan Luas dan Emisi
    X = df_tahun_terpilih[['Luas_Karhutla', 'Emisi_CO2']]
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_tahun_terpilih['Cluster'] = kmeans.fit_predict(X)
    
    # Otomatis memberi nama zona berdasarkan rata-rata emisi tiap klaster
    urutan = df_tahun_terpilih.groupby('Cluster')['Emisi_CO2'].mean().sort_values().index
    # PERUBAHAN LEVEL UP: Nama zona berorientasi pada masyarakat umum
    mapping = {
        urutan[0]: 'Aman (Udara Bersih)', 
        urutan[1]: 'Siaga Masker (Kualitas Udara Menurun)', 
        urutan[2]: 'Bahaya ISPA (Darurat Asap)'
    }
    df_tahun_terpilih['Peringatan_Kesehatan'] = df_tahun_terpilih['Cluster'].map(mapping)
    return df_tahun_terpilih

# --- 4. FUNGSI MACHINE LEARNING 2: REGRESI LINEAR (PREDIKSI MASA DEPAN) ---
def prediksi_emisi_2026(df, provinsi_pilihan):
    # Ambil data historis 2018-2025 khusus untuk provinsi yang dipilih
    df_prov = df[df['Provinsi'] == provinsi_pilihan].copy()
    
    # X adalah Tahun, y adalah Emisi yang diproduksi
    X = df_prov[['Tahun']].values
    y = df_prov['Emisi_CO2'].values
    
    # Latih model Machine Learning untuk melihat tren garis lurus (Regresi)
    model = LinearRegression()
    model.fit(X, y)
    
    # Ramal/Prediksi untuk tahun 2026
    prediksi_2026 = model.predict([[2026]])[0]
    
    # Emisi tidak mungkin minus, jadi jika model memprediksi minus, kita jadikan 0
    return max(0, prediksi_2026), df_prov, model

# (Batas Akhir Tahap 2)
# st.write("✅ Sistem Mesin Pembelajaran dan Data Historis (2018-2025) berhasil dimuat!")

# Hapus tulisan centang hijau sebelumnya agar UI-nya bersih
# st.write("✅ Sistem Mesin Pembelajaran dan Data Historis (2018-2025) berhasil dimuat!")

# --- 5. ANTARMUKA PENGGUNA (UI) STREAMLIT ---

# Memuat data
df_master = load_data()

# Judul Aplikasi
st.title("🔥 KACC: Public Health Warning & Prediksi Karhutla")
st.markdown("Sistem peringatan dini risiko ISPA berbasis data Karhutla historis (2018-2025) dan prediksi emisi masa depan menggunakan *Machine Learning*.")
st.markdown("---")

# Sidebar untuk Filter
st.sidebar.header("Pusat Kendali Publik")
tahun_pilihan = st.sidebar.slider("Pilih Tahun Historis", 2018, 2025, 2025)
provinsi_list = sorted(df_master['Provinsi'].unique())
provinsi_pilihan = st.sidebar.selectbox("Pilih Provinsi untuk Prediksi", provinsi_list)

# Menjalankan K-Means untuk tahun terpilih
df_tahun_ini = df_master[df_master['Tahun'] == tahun_pilihan].copy()
df_klaster = jalankan_kmeans(df_tahun_ini)

# Menjalankan Prediksi (Regresi Linear) untuk provinsi terpilih
prediksi_emisi, df_prov_historis, model_regresi = prediksi_emisi_2026(df_master, provinsi_pilihan)

# --- TAMPILAN UTAMA ---

# Bagian 1: Status Peringatan Dini (K-Means)
st.subheader(f"Peringatan Dini Kesehatan Nasional ({tahun_pilihan})")
status_provinsi = df_klaster[df_klaster['Provinsi'] == provinsi_pilihan]['Peringatan_Kesehatan'].values
if len(status_provinsi) > 0:
    st.info(f"**Status {provinsi_pilihan} ({tahun_pilihan}):** {status_provinsi[0]}")
else:
    st.warning("Data tidak tersedia untuk provinsi dan tahun ini.")

# Tampilkan Peta atau Tabel Sederhana
st.dataframe(df_klaster[['Provinsi', 'Luas_Karhutla', 'Emisi_CO2', 'Peringatan_Kesehatan']].sort_values('Emisi_CO2', ascending=False), use_container_width=True)

st.markdown("---")

# Bagian 2: Prediksi Emisi 2026 (Regresi Linear)
st.subheader(f"Proyeksi Emisi Karbon 2026: {provinsi_pilihan}")

# Layout kolom untuk menampilkan angka prediksi
col1, col2 = st.columns(2)
with col1:
    emisi_terakhir = df_prov_historis[df_prov_historis['Tahun'] == 2025]['Emisi_CO2'].values[0] if len(df_prov_historis[df_prov_historis['Tahun'] == 2025]) > 0 else 0
    st.metric(label="Emisi Aktual (2025)", value=f"{emisi_terakhir:,.0f} Ton")

with col2:
    # Hitung selisih tren
    delta = prediksi_emisi - emisi_terakhir
    st.metric(label="Prediksi Emisi (2026)", value=f"{prediksi_emisi:,.0f} Ton", delta=f"{delta:,.0f} Ton dari 2025", delta_color="inverse")

# Membuat Grafik Tren & Garis Prediksi menggunakan Plotly
# Gabungkan data historis dan data prediksi untuk grafik
df_grafik = df_prov_historis[['Tahun', 'Emisi_CO2']].copy()
df_prediksi = pd.DataFrame({'Tahun': [2026], 'Emisi_CO2': [prediksi_emisi]})
df_grafik_lengkap = pd.concat([df_grafik, df_prediksi], ignore_index=True)

# Plotting grafik garis
fig = px.line(df_grafik_lengkap, x='Tahun', y='Emisi_CO2', markers=True, 
              title=f"Tren Emisi Historis dan Prediksi 2026 - {provinsi_pilihan}")

# Menandai titik prediksi tahun 2026 dengan warna beda
fig.add_scatter(x=[2026], y=[prediksi_emisi], mode='markers', marker=dict(color='red', size=10), name='Prediksi 2026')

st.plotly_chart(fig, use_container_width=True)
