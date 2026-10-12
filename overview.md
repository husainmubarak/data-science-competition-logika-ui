# Overview & Panduan Kompetisi: Data Science Competition LOGIKA UI 2026

Dokumen ini berisi panduan ringkas, aturan teknis, penjelasan metrik evaluasi, serta strategi alur kerja untuk tim dalam mengikuti **Data Science Competition (DSC) LOGIKA UI 2026**.

---

## 📌 1. Informasi Umum & Jadwal

* **Nama Kompetisi:** Data Science Competition (DSC) LOGIKA UI 2026
* **Platform:** Kaggle Competition (Daring & Asinkron)
* **Link Kaggle:** [kaggle.com/t/b8118ad900f84468a7e0735c418ca5b7](https://www.kaggle.com/t/b8118ad900f84468a7e0735c418ca5b7)
* **Periode Babak Penyisihan:** 11 Oktober – 30 Oktober 2026
* **Batas Pengumpulan (Deadline):** 30 Oktober 2026 Pukul 23.59 WIB
* **Pinalti Keterlambatan:** Pengurangan **1 poin per menit** dari nilai akhir jika mengumpulkan setelah 23.59 WIB.
* **Batas Submisi Kaggle:** Maksimal **3 kali submisi per hari**.

---

## 🎯 2. Misi & Problem Statement

* **Domain:** *Consumer Lending* (Pinjaman Konsumen) di Indonesia.
* **Tugas:** Memprediksi probabilitas terjadinya **serious repayment difficulty** (gagal bayar/keterlambatan $\ge 90$ hari atau *Days Past Due* / DPD) dalam kurun waktu 12 bulan setelah pinjaman diberikan (*origination*).
* **Target Variable (`default_probability`):**
  * `1` : Pinjaman mencapai $\ge 90$ DPD dalam 12 bulan.
  * `0` : Pinjaman lancar (tidak mencapai 90 DPD dalam 12 bulan).

---

## 📊 3. Ringkasan Dataset (Multi-Tabel)

Dataset terdiri dari beberapa sumber data yang saling terhubung (*relational*):
1. **Loan Applications:** Informasi pengajuan pinjaman utama.
2. **Customer Profiles:** Demografi, pekerjaan, pendapatan, tanggungan, dan hunian.
3. **Previous Applications:** Riwayat pengajuan pinjaman sebelumnya.
4. **Repayment History:** Riwayat angsuran dan pembayaran historis.
5. **Account Behavior:** Saldo bulanan, arus masuk (*inflow*), arus keluar (*outflow*), dan aktivitas rekening.
6. **Credit Bureau Observations:** Skor kredit, sisa pinjaman, utilitas kredit, dan riwayat tunggakan.
7. **E-commerce Transactions:** Riwayat belanja e-commerce, kategori produk, dan metode pembayaran.

> ⚠️ **Aturan Emas (Data Leakage & Cutoff Time):**
> Hanya gunakan data/transaksi yang terjadi **SEBELUM atau SAMA DENGAN `decision_datetime`** pada pengajuan yang bersangkutan.

---

## ⚙️ 4. Ketentuan Teknis & Batasan Pengembangan Model

| Aturan | Status | Keterangan |
| :--- | :--- | :--- |
| **Bahasa Pemrograman** | **Wajib** | Menggunakan **Python**. |
| **Data Eksternal** | **Dilarang** | Hanya boleh menggunakan data resmi dari panitia. |
| **AutoML & AutoFE** | **Dilarang** | Dilarang menggunakan Auto-ML/Auto-Feature Engineering. |
| **Manual Feature Engineering** | **Diperbolehkan** | Preprocessing, EDA, dan rekayasa fitur secara manual sangat dianjurkan. |
| **Reproduksibilitas** | **Wajib** | Mengatur `random_state`/`seed` agar kode dapat dijalankan ulang dengan hasil identik. |

---

## 🏆 5. Indikator Penilaian & Deliverables

### A. Bobot Penilaian Akhir
* **Public Score Kaggle (45%):** Evaluasi performa awal pada subset data *test*.
* **Private Score Kaggle (45%):** Evaluasi performa akhir yang menentukan finalis.
* **Notebook & Kelancaran Kode (10%):** Reproduksibilitas dan keterbacaan kode.
*(Sebanyak **8 tim teratas** akan lolos ke Babak Final)*.

### B. Deliverables yang Dikumpulkan (via Google Form Panitia)
1. **File Notebook (`.ipynb`)**
   * Nama file: `NomorPeserta_NamaTim_Notebook.ipynb`
2. **Link Drive Bobot Model**
   * Berisi *weights* / *model checkpoint* yang dilatih.
3. **File Submisi CSV (`.csv`)**
   * Nama file: `NomorPeserta_NamaTim_Submisi.csv`
   * Format kolom: `application_id,default_probability`

---

## 📈 6. Metrik Evaluasi: KS Statistic

Evaluasi menggunakan **Kolmogorov–Smirnov (KS) Statistic**, yang dihitung dengan rumus:

$$KS = \max_{s} \left| F_1(s) - F_0(s) \right|$$

* **Fungsi:** Mengukur selisih maksimum antara kurva kumulatif kelompok gagal bayar ($y=1$) dan kelompok lancar ($y=0$).
* **Fokus Utama:** Pemeringkatan risiko (*risk ranking*). Semakin tinggi nilai KS, semakin baik model memisahkan nasabah berisiko tinggi dan rendah.
* **Implementasi Python (Lokal):**
  ```python
  from scipy.stats import ks_2samp

  def calculate_ks_score(y_true, y_pred):
      preds_bad = y_pred[y_true == 1]
      preds_good = y_pred[y_true == 0]
      if len(preds_bad) == 0 or len(preds_good) == 0:
          return 0.0
      return float(ks_2samp(preds_bad, preds_good).statistic)
  ```

---

## 🚀 7. Step-by-Step Execution Strategy

1. **Setup & Data Filtering:**
   * Lakukan *merge* antar-tabel berdasarkan ID.
   * Filter transaksi berdasarkan `decision_datetime` untuk cegah *data leakage*.
2. **Feature Engineering (Fokus Utama):**
   * Hitung agregasi (`mean`, `max`, `sum`, `std`) transaksi, saldo, dan keterlambatan.
   * Buat rasio rasional (misal: *Debt-to-Income ratio*, frekuensi DPD).
3. **Cross-Validation Setup:**
   * Gunakan `StratifiedKFold` (5-Fold) untuk validasi lokal.
   * Hitung skor KS di tiap *fold*.
4. **Pemodelan:**
   * Gunakan algoritma *Gradient Boosting* (LightGBM, XGBoost, CatBoost).
5. **Final Check & Submission:**
   * Simpan file model (`.pkl`/`.joblib`).
   * Pastikan output `.csv` tidak ada nilai `NaN` atau di luar rentang $0-1$.
```
eof

File `overview.md` telah berhasil dibuat. Kamu bisa menyimpan dan membagikan file ini ke rekan tim sebagai rujukan utama selama kompetisi. Ada langkah spesifik yang ingin kita diskusikan lebih lanjut, seperti pembuatan fitur dari tabel-tabel tersebut?