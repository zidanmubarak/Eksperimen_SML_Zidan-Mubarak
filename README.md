# Eksperimen SML - Zidan Mubarak

Repository ini berisi eksperimen dan preprocessing otomatis untuk **Wine Quality Dataset** sebagai bagian dari submission kelas Machine Learning di Dicoding.

![Preprocessing Status](https://github.com/zidanmubarak/Eksperimen_SML_Zidan-Mubarak/actions/workflows/preprocessing.yml/badge.svg)

## 📁 Struktur Repository

```
Eksperimen_SML_Zidan-Mubarak/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions workflow
├── WineQT_raw/
│   └── WineQT.csv                     # Dataset original
├── preprocessing/
│   ├── Eksperimen_Zidan-Mubarak.ipynb # Notebook eksperimen
│   ├── automate_Zidan-Mubarak.py      # Script preprocessing otomatis
│   └── WineQT_preprocessing.csv       # Dataset hasil preprocessing
├── README.md
└── requirements.txt                   # Dependencies Python
```

## 📊 Dataset

**Wine Quality Dataset** berisi informasi tentang karakteristik kimia wine dan rating kualitasnya.

- **Jumlah sampel**: 1,143 baris
- **Jumlah fitur**: 13 kolom
- **Target variable**: `quality` (rating 3-8)

### Fitur Dataset:
- `fixed acidity`: Keasaman tetap
- `volatile acidity`: Keasaman volatil
- `citric acid`: Asam sitrat
- `residual sugar`: Gula sisa
- `chlorides`: Klorida
- `free sulfur dioxide`: Sulfur dioksida bebas
- `total sulfur dioxide`: Total sulfur dioksida
- `density`: Densitas
- `pH`: Tingkat keasaman
- `sulphates`: Sulfat
- `alcohol`: Kadar alkohol
- `quality`: Kualitas wine (target)
- `Id`: Identifier

## 🔬 Eksperimen (Basic - 2 pts)

Notebook `Eksperimen_Zidan-Mubarak.ipynb` berisi:

1. **Data Loading**: Memuat dataset dari CSV
2. **Exploratory Data Analysis (EDA)**:
   - Statistik deskriptif
   - Analisis missing values
   - Deteksi data duplikat
   - Analisis distribusi
   - Correlation matrix
   - Deteksi outliers
3. **Data Preprocessing**:
   - Menghapus data duplikat
   - Menangani outliers (IQR capping)
   - Feature engineering (drop Id column)
   - Feature scaling (StandardScaler)
   - Train-test split (80:20)
   - Save processed data

## 🤖 Automation Script (Skilled - 3 pts)

File `automate_Zidan-Mubarak.py` adalah script Python yang mengotomatisasi seluruh proses preprocessing.

### Cara Penggunaan:

```bash
# Masuk ke folder preprocessing
cd preprocessing

# Jalankan script
python automate_Zidan-Mubarak.py

# Atau dengan custom path
python automate_Zidan-Mubarak.py ../WineQT_raw/WineQT.csv output.csv
```

### Fungsi-fungsi:
- `load_data()`: Memuat dataset
- `remove_duplicates()`: Menghapus duplikat
- `handle_outliers()`: Menangani outliers
- `feature_engineering()`: Feature engineering
- `scale_features()`: Feature scaling
- `split_data()`: Train-test split
- `save_processed_data()`: Menyimpan hasil
- `preprocess_pipeline()`: Pipeline lengkap

## ⚙️ GitHub Actions (Advance - 4 pts)

Workflow otomatis yang berjalan setiap kali ada perubahan pada:
- Dataset (`WineQT_raw/**`)
- Script preprocessing (`preprocessing/automate_Zidan-Mubarak.py`)
- Workflow file (`.github/workflows/preprocessing.yml`)

### Trigger:
- Push ke branch `main` atau `master`
- Pull request
- Manual trigger (workflow_dispatch)

### Proses:
1. ✅ Checkout repository
2. ✅ Setup Python 3.10
3. ✅ Install dependencies (pandas, numpy, scikit-learn, dll)
4. ✅ Run preprocessing script
5. ✅ Verify output file
6. ✅ Upload artifact
7. ✅ Commit hasil preprocessing (optional)

### Melihat Hasil:
1. Buka tab **Actions** di GitHub repository
2. Pilih workflow run terbaru
3. Download artifact `preprocessed-wine-data`

## 📦 Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/zidanmubarak/Eksperimen_SML_Zidan-Mubarak.git
cd Eksperimen_SML_Zidan-Mubarak
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run Notebook (Basic)
```bash
jupyter notebook preprocessing/Eksperimen_Zidan-Mubarak.ipynb
```

### 4. Run Automation Script (Skilled)
```bash
cd preprocessing
python automate_Zidan-Mubarak.py
```

### 5. Trigger GitHub Actions (Advance)
```bash
git add .
git commit -m "Update dataset or script"
git push origin main
```

## 📈 Hasil Preprocessing

Setelah preprocessing, dataset akan memiliki karakteristik:
- ✅ Tidak ada data duplikat
- ✅ Outliers sudah ditangani (capping method)
- ✅ Kolom Id sudah dihapus
- ✅ Semua fitur sudah di-scale (StandardScaler)
- ✅ Data sudah di-split menjadi train (80%) dan test (20%)

## 📄 License

Dataset ini digunakan untuk keperluan edukasi dalam submission kelas Machine Learning di Dicoding.
