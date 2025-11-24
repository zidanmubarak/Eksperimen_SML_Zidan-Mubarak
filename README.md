# 🌍 Climate Change Impact on Agriculture - Data Preprocessing

## 📋 Deskripsi

Repository ini berisi eksperimen dan automasi preprocessing untuk dataset **Climate Change Impact on Agriculture 2024**. Dataset ini menganalisis dampak perubahan iklim terhadap hasil pertanian di berbagai negara dan region.

![Preprocessing Status](https://github.com/zidanmubarak/Eksperimen_SML_Zidan-Mubarak/actions/workflows/preprocessing.yml/badge.svg)

## 📂 Struktur Folder

```
Eksperimen_SML_Zidan-Mubarak/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions workflow
├── preprocessing/
│   ├── Eksperimen_Zidan-Mubarak.ipynb # Notebook eksperimen
│   ├── automate_Zidan-Mubarak.py      # Script automasi
│   └── climate_change_preprocessing.csv # Hasil preprocessing
├── climate_change_raw.csv             # Dataset asli
├── requirements.txt                   # Dependencies Python
└── README.md                          # Dokumentasi ini
```

---

## 📊 Dataset Information

### Original Dataset
- **Rows**: 10,002
- **Columns**: 15
- **Size**: ~1 MB

### Columns
1. `Year` - Tahun pengamatan (1990-2024)
2. `Country` - Negara
3. `Region` - Region dalam negara
4. `Crop_Type` - Jenis tanaman
5. `Average_Temperature_C` - Suhu rata-rata (°C)
6. `Total_Precipitation_mm` - Curah hujan total (mm)
7. `CO2_Emissions_MT` - Emisi CO2 (MT)
8. `Crop_Yield_MT_per_HA` - **Target**: Hasil panen (MT/HA)
9. `Extreme_Weather_Events` - Jumlah kejadian cuaca ekstrem
10. `Irrigation_Access_%` - Akses irigasi (%)
11. `Pesticide_Use_KG_per_HA` - Penggunaan pestisida (KG/HA)
12. `Fertilizer_Use_KG_per_HA` - Penggunaan pupuk (KG/HA)
13. `Soil_Health_Index` - Indeks kesehatan tanah
14. `Adaptation_Strategies` - Strategi adaptasi
15. `Economic_Impact_Million_USD` - Dampak ekonomi (juta USD)

---

## 🔧 Setup dan Instalasi

### Prerequisites
- Python 3.12.7
- Git

### 1. Clone Repository
```bash
git clone https://github.com/zidanmubarak/Eksperimen_SML_Zidan-Mubarak.git
cd Eksperimen_SML_Zidan-Mubarak
```

### 2. Install Dependencies
```bash
pip install pandas==2.2.3 numpy==2.1.3 scikit-learn==1.5.2 matplotlib==3.9.2 seaborn==0.13.2
```

---

## 🚀 Cara Menggunakan

### Option 1: Manual Experimentation (Basic)
```bash
# Open Jupyter Notebook
jupyter notebook preprocessing/Eksperimen_Zidan-Mubarak.ipynb
```

### Option 2: Automated Preprocessing (Skilled)
```bash
cd preprocessing
python automate_Zidan-Mubarak.py
```

**Output**: `climate_change_preprocessing.csv`

### Option 3: GitHub Actions (Advanced)
1. Push ke GitHub repository
2. Workflow akan otomatis berjalan
3. Download artifact dari Actions tab

---

## 📈 Preprocessing Steps

### 1. Data Loading
- Load dataset dari CSV
- Validasi struktur data

### 2. Data Exploration
- Analisis statistik deskriptif
- Identifikasi missing values
- Analisis distribusi data

### 3. Missing Value Handling
- Numerical: Median imputation
- Categorical: Mode imputation

### 4. Outlier Treatment
- IQR method
- Capping outliers (bukan removal)

### 5. Feature Engineering
Membuat 6 fitur baru:
- `Temperature_Category` - Kategori suhu
- `Precipitation_Category` - Kategori curah hujan
- `Yield_Category` - Kategori hasil panen
- `Climate_Stress` - Indikator stress iklim
- `Resource_Efficiency` - Efisiensi penggunaan resource
- `Environmental_Impact` - Skor dampak lingkungan

### 6. Categorical Encoding
- Label Encoding untuk 7 kolom kategorikal
- Country, Region, Crop_Type, Adaptation_Strategies, dll

### 7. Feature Selection
- 12 numerical features
- 6 encoded categorical features
- 1 target variable

### 8. Feature Scaling
- StandardScaler (mean=0, std=1)
- Semua features kecuali target

### 9. Data Saving
- Save ke CSV format
- Ready untuk model training

---

## 🤖 GitHub Actions Workflow

### Triggers
- Push ke `main`/`master` branch
- Pull request
- Manual dispatch

### Steps
1. ✅ Checkout repository
2. ✅ Setup Python 3.12
3. ✅ Install dependencies
4. ✅ Run preprocessing script
5. ✅ Verify output
6. ✅ Upload artifact (30 days retention)
7. ✅ Commit hasil preprocessing

### Workflow File
`.github/workflows/preprocessing.yml`

---

## 📊 Processed Dataset

### Output
- **File**: `climate_change_preprocessing.csv`
- **Rows**: 10,002
- **Columns**: 19 (12 numerical + 6 encoded + 1 target)
- **Features**: Scaled (mean≈0, std≈1)
- **Target**: `Crop_Yield_MT_per_HA` (original scale)

### Features List
**Numerical Features (12):**
1. Year
2. Average_Temperature_C
3. Total_Precipitation_mm
4. CO2_Emissions_MT
5. Extreme_Weather_Events
6. Irrigation_Access_%
7. Pesticide_Use_KG_per_HA
8. Fertilizer_Use_KG_per_HA
9. Soil_Health_Index
10. Climate_Stress
11. Resource_Efficiency
12. Environmental_Impact

**Encoded Features (6):**
13. Country_Encoded
14. Region_Encoded
15. Crop_Type_Encoded
16. Adaptation_Strategies_Encoded
17. Temperature_Category_Encoded
18. Precipitation_Category_Encoded

**Target (1):**
19. Crop_Yield_MT_per_HA

---

## 🎓 Use Cases

Dataset ini cocok untuk:
- **Regression**: Prediksi Crop Yield
- **Classification**: Kategori Yield (Low/Medium/High/Very High)
- **Time Series**: Analisis trend perubahan iklim
- **Clustering**: Pengelompokan region berdasarkan karakteristik iklim

---

## 📸 Screenshots untuk Submission

### 1. Eksperimen Manual
- Screenshot notebook dengan semua cells executed
- Tampilkan EDA, preprocessing steps, visualisasi

### 2. Script Automasi
- Screenshot terminal menjalankan `automate_Zidan-Mubarak.py`
- Tampilkan output preprocessing summary

### 3. GitHub Actions
- Screenshot workflow running
- Screenshot artifact uploaded
- Screenshot preprocessed file di repository

---

## 🔗 Links

- **GitHub Repository**: `https://github.com/zidanmubarak/Eksperimen_SML_Zidan-Mubarak`
- **GitHub Actions**: `https://github.com/zidanmubarak/Eksperimen_SML_Zidan-Mubarak/actions`
