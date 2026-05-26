# Eksperimen_SML_Kholilah-Nurafifah

Repository untuk eksperimen dan preprocessing dataset **Heart Disease** dalam rangka submission kelas **Membangun Sistem Machine Learning (MSML)** — Dicoding.

---

## 📁 Struktur Repository

```
Eksperimen_SML_Kholilah-Nurafifah
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions workflow (Advance)
├── heart.csv                           # Dataset mentah (raw)
├── preprocessing/
│   ├── Eksperimen_Kholilah-Nurafifah.ipynb   # Notebook eksperimen
│   ├── automate_Kholilah-Nurafifah.py        # Script otomatisasi preprocessing
│   └── heart_preprocessing/
│       ├── heart_train.csv            # Dataset train hasil preprocessing
│       └── heart_test.csv             # Dataset test hasil preprocessing
└── README.md
```

---

## 📊 Dataset

- **Nama:** Heart Disease UCI
- **Sumber:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Tipe Masalah:** Klasifikasi Biner (0 = Tidak Sakit, 1 = Sakit Jantung)
- **Jumlah Fitur:** 13 fitur input + 1 target

---

## 🔧 Tahapan Preprocessing

| No | Tahapan | Keterangan |
|----|---------|-----------|
| 1 | Data Loading | Memuat dataset dari CSV |
| 2 | Hapus Duplikat | Menghapus baris yang duplikat |
| 3 | Handle Missing Values | Imputasi median untuk numerik |
| 4 | Handle Outlier | IQR Clipping |
| 5 | One-Hot Encoding | Encoding kolom kategorikal |
| 6 | Train-Test Split | 80:20 dengan stratified split |
| 7 | Feature Scaling | StandardScaler pada fitur numerik |

---

## 🚀 Cara Menjalankan

### Notebook Eksperimen
```bash
cd preprocessing
jupyter notebook Eksperimen_Kholilah-Nurafifah.ipynb
```

### Script Otomatisasi
```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Jalankan preprocessing
cd preprocessing
python automate_Kholilah-Nurafifah.py
```

### GitHub Actions
Workflow otomatis terpicu ketika:
- Push ke branch `main` yang mengubah `heart.csv` atau `automate_Kholilah-Nurafifah.py`
- Manual trigger via tab Actions di GitHub

---

## 👤 Author

**Kholilah Nurafifah**  
Submission Kelas Membangun Sistem Machine Learning — Dicoding
