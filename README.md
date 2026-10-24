# Pantauan Karhutla Indonesia (KACC)

Aplikasi web analitik untuk memantau situasi kebakaran hutan dan lahan (karhutla) serta estimasi emisi karbon di 38 provinsi Indonesia berdasarkan data historis 2018–2025. Aplikasi ini mengelompokkan provinsi ke dalam tingkat karhutla relatif menggunakan *K-Means Clustering*, menyajikan proyeksi skenario emisi 2026, analisis tren nasional, dan peta panas (*heatmap*) provinsi.

Dibangun dengan Python, Streamlit, Plotly, dan Scikit-Learn.

---

## Cara Menjalankan

### 1. Prasyarat
- Python 3.10 atau versi yang lebih baru.
- Berkas data `DATA_MASTER_2018_2025.csv` **harus berada di folder yang sama** dengan `app_publik.py`.

### 2. Instalasi Dependensi
Pasang dependensi yang tercantum pada `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi
Jalankan aplikasi dari terminal:
```bash
python -m streamlit run app_publik.py
```
Aplikasi dapat diakses melalui peramban web di `http://localhost:8501`.

---

## Fitur Utama

1. **Pusat Kendali Publik (Sidebar)**
   - Pemilihan tahun historis (2018–2025).
   - Pemilihan provinsi untuk analisis mendalam.
   - Pengaturan jumlah klaster tingkat karhutla (*k* = 2 hingga 5).

2. **Peringatan Dini Tingkat Karhutla (Blok 1)**
   - Status kartu indikator: Tingkat Karhutla (Rendah, Sedang, Tinggi), Luas Terbakar, Emisi CO2, Total Hotspot, Peringkat Emisi Nasional, dan perubahan dibanding tahun sebelumnya.
   - **Peta Spasial**: Peta interaktif Carto Darkmatter menampilkan sebaran provinsi dengan ukuran lingkaran proporsional terhadap akar kuadrat emisi.
   - **Tabel Data**: Daftar lengkap metrik karhutla per provinsi dilengkapi bilah progres emisi.
   - **Peringkat**: Diagram batang horizontal Top 10 provinsi penyumbang emisi terbesar.
   - **Ringkasan Zona**: Statistik jumlah provinsi, rentang emisi, dan median luas terbakar per tingkat.

3. **Proyeksi Skenario Emisi 2026 (Blok 2)**
   - Perbandingan emisi aktual 2025, proyeksi skenario 2026, dan rentang historis (min–maks).
   - Grafik tren historis beserta arah skenario 2026 dan rentang fluktuasi historis.
   - Evaluasi kelayakan (*rolling backtest* 2023–2025) untuk memilih antara tren linear dan median historis berdasarkan nilai *Mean Absolute Error* (MAE).

4. **Tren Nasional 2018–2025 (Blok 3)**
   - Tiga grafik agregat nasional (*shared x-axis*): Luas Terbakar, Emisi CO2, dan Total Hotspot.
   - Penandaan tahun ekstrem (2019 dan 2023).

5. **Peta Panas Provinsi × Tahun (Blok 4)**
   - Matriks visual interaktif Provinsi × Tahun untuk metrik Luas, Emisi, Hotspot, dan Intensitas Emisi.
   - Menggunakan skala logaritmik `log10(1+x)` dengan nilai aktual pada *hover*.
   - Provinsi diurutkan secara menurun berdasarkan nilai kumulatif.
   - Penandaan eksplisit sel `Belum tercatat`.

6. **Catatan Metodologi & Ekspor Data (Blok 5)**
   - Dokumentasi lengkap metodologi dan batasan data.
   - Tombol unduh data hasil analisis tahun terpilih berformat CSV (UTF-8-SIG).

---

## Batasan Kunci & Metodologi Data

- **Sumber Data**: Berkas `DATA_MASTER_2018_2025.csv`.
- **Pembersihan Entitas**: Baris `Total` dan wilayah `IKN` dikeluarkan dari tabel analisis utama. Baris `Total` memiliki nilai hotspot 0 sehingga tidak konsisten untuk analisis agregat, sedangkan `IKN` tidak memiliki catatan luas terbakar dan bukan merupakan provinsi definitif.
- **Provinsi Pemekaran Papua**: Papua Barat Daya, Papua Pegunungan, Papua Selatan, dan Papua Tengah belum memiliki catatan pada tahun 2018–2022 (seluruh kolom bernilai 0). Baris ini diberi label `Belum tercatat` dan dikecualikan dari pemodelan pada periode tersebut.
- **Tingkat Karhutla (K-Means)**: Variabel Luas Terbakar dan Emisi CO2 ditransformasikan dengan `log1p` lalu `StandardScaler`. Penamaan tingkat diurutkan berdasarkan rata-rata emisi klaster. Tingkat ini bersifat relatif antarprovinsi pada tahun terpilih dan bukan merupakan pengukuran kualitas udara atau kasus kesehatan ISPA.
- **Integritas Hotspot**: Hotspot adalah data indikasi satelit awal. Terdapat 9 baris data yang memiliki nilai hotspot 0 meski luas karhutla > 0 (Bangka Belitung 2023–2025, DI Yogyakarta 2023–2024, dan 4 provinsi pemekaran Papua tahun 2023) yang dikategorikan sebagai `Hotspot meragukan`.
- **Proyeksi Skenario**: Nilai tahun 2026 merupakan proyeksi skenario berbasis data historis dengan evaluasi *backtest*, bukan ramalan atau prediksi resmi instansi pemerintah.
- **Korelasi Luas dan Emisi**: Metodologi kalkulasi emisi pada sumber data tidak dipaparkan secara eksplisit dan diasumsikan merupakan turunan dari luas terbakar dikalikan faktor emisi biomassa, sehingga hubungan keduanya sebagian besar bersifat terdefinisi.
