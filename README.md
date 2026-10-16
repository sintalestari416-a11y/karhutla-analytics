Karhutla Analytics 2025

Dashboard interaktif untuk menganalisis kebakaran hutan dan lahan (karhutla) di 38 provinsi Indonesia tahun 2025. Aplikasi ini mengelompokkan provinsi ke dalam zona risiko dengan K-Means Clustering, lalu menampilkannya lewat peta, ranking, dan analisis korelasi dalam tema gelap yang minimalis.

Dibuat dengan Python dan Streamlit.

<!-- Tambahkan screenshot dashboard di sini setelah diunggah ke repo: ![Tampilan dashboard](docs/screenshot.png) -->
Fitur
KPI nasional: total luas terbakar, total emisi CO₂, total hotspot, dan jumlah provinsi zona Bahaya.
Insight otomatis: ringkasan dinamis yang berubah mengikuti filter yang dipilih.
Peta sebaran: titik per provinsi di peta Indonesia. Ukuran titik mengikuti metrik terpilih, warna mengikuti zona risiko.
Ranking: Top 10 provinsi berdasarkan metrik terpilih, termasuk intensitas emisi (ton per hektar).
Zona risiko (K-Means): sebaran provinsi per zona, ringkasan statistik tiap zona, dan daftar provinsinya.
Korelasi: matriks korelasi Pearson antara luas terbakar, emisi CO₂, dan hotspot.
Data: tabel hasil klasterisasi yang dapat diunduh dalam format CSV.
Filter: jumlah klaster (k = 2 sampai 5), metrik utama, zona risiko, dan provinsi.
Data

Sumber berkas: DATASET_FINAL_SIAP_LATIH.xlsx (data tahun 2025).

Kolom	Keterangan	Satuan
Provinsi	Nama provinsi	-
Luas_Karhutla_2025	Luas lahan terbakar	hektar (ha)
Emisi_CO2_2025	Estimasi emisi CO₂	ton
Total_Hotspot_2025	Jumlah titik panas	titik

Dataset berisi 38 provinsi dan satu baris Total. Baris Total dipisahkan dari data provinsi dan tidak ikut dalam klasterisasi, grafik, maupun tabel.

Sumber Data

- Nama sumber: SiPongi+ (Sistem Pemantauan Karhutla), Kementerian Kehutanan Republik Indonesia
- URL: https://sipongi.gakkum.kehutanan.go.id/
- Tahun data: 2025
- Catatan: Hotspot adalah indikasi titik panas hasil deteksi satelit dan tidak selalu berarti terjadi kebakaran hutan dan lahan. Klasterisasi zona risiko dalam dashboard ini bersifat relatif antarprovinsi dan dihitung dari data yang tersedia.

Metodologi
Pembersihan: baris Total dipisahkan dari data provinsi.
Fitur turunan: Intensitas_Emisi = Emisi_CO2 / Luas_Karhutla (ton per hektar). Bernilai 0 bila luas terbakar 0.
Transformasi: data sangat timpang (nilai antar provinsi berbeda ribuan kali lipat), sehingga tiga fitur diubah dengan log1p lalu distandarisasi dengan StandardScaler.
Klasterisasi: KMeans(random_state=42, n_init=10) dengan k yang dapat diatur (default 3).
Pelabelan zona: klaster diurutkan berdasarkan rata-rata emisi CO₂ dari terendah ke tertinggi.
k	Label zona (rendah ke tinggi)
2	Aman, Bahaya
3	Aman, Waspada, Bahaya
4	Aman, Waspada, Siaga, Bahaya
5	Aman, Rendah, Waspada, Siaga, Bahaya
Temuan utama (k = 3)
Total nasional: 359.619,42 ha terbakar, 56.204.535 ton emisi CO₂, dan 1.068.622 titik panas (jumlah dari 38 provinsi).
Lima provinsi dengan emisi tertinggi: Sumatera Utara, Riau, Nusa Tenggara Timur, Kalimantan Barat, dan Maluku.
Intensitas emisi tertinggi: Riau (sekitar 388 ton/ha), Papua Tengah, dan Kalimantan Barat.
Hasil klaster: 26 provinsi Bahaya, 10 Waspada, dan 2 Aman.
Korelasi Pearson dengan emisi CO₂: luas terbakar r = 0,74 (kuat), jumlah hotspot r = 0,18 (lemah). Hotspot saja tidak cukup untuk memperkirakan besarnya emisi. Korelasi ini menunjukkan keterkaitan statistik, bukan hubungan sebab-akibat.
Struktur proyek
.
├── app.py                         # Aplikasi Streamlit
├── DATASET_FINAL_SIAP_LATIH.xlsx  # Dataset
├── requirements.txt               # Daftar dependensi
├── .streamlit/
│   └── config.toml                # Tema gelap
└── .gitignore
Menjalankan secara lokal

Prasyarat: Python 3.10 atau lebih baru.

bash
# 1. Clone repositori
git clone https://github.com/sintalestari416-a11y/karhutla-analytics.git
cd karhutla-analytics

# 2. (Opsional) buat virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Instal dependensi
pip install -r requirements.txt

# 4. Jalankan dashboard
python -m streamlit run app.py

Dashboard terbuka di http://localhost:8501. Jalankan perintah dari folder yang berisi app.py agar berkas tema di .streamlit/config.toml terbaca.

Peta memakai tile CARTO sehingga membutuhkan koneksi internet.

Teknologi
Streamlit: kerangka aplikasi web
pandas dan openpyxl: pengolahan data dan pembacaan Excel
scikit-learn: K-Means dan StandardScaler
Plotly: grafik dan peta interaktif
Keterbatasan
Data hanya mencakup tahun 2025, sehingga tidak ada analisis tren antar tahun.
Zona risiko bersifat relatif antar provinsi dalam dataset ini, bukan standar risiko resmi. Zona "Aman" pada k = 3 hanya berisi provinsi dengan luas terbakar 0.
Beberapa provinsi memiliki nilai 0 pada satu kolom. Contohnya Bangka Belitung, yang luas terbakarnya 1.670,83 ha tetapi hotspot-nya 0. Ini kemungkinan menandakan data tidak lengkap, dan dashboard menampilkan catatannya.
Titik pada peta diletakkan di koordinat pusat provinsi (perkiraan), bukan batas wilayah administratif.
Pada k = 4 atau lebih, klaster kecil dapat terbentuk karena nilai ekstrem atau data kosong, bukan karena perbedaan risiko yang nyata.
Penulis

@sintalestari416-a11y
