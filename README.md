# Analisis dan Prediksi Angka Harapan Hidup Negara

## 📌 Deskripsi Project

Project ini merupakan sistem analisis dan prediksi **angka harapan hidup (Life Expectancy)** negara-negara di dunia yang dikembangkan untuk mendukung **SDG 3: Good Health and Well-being**.

Project mengintegrasikan data dari beberapa sumber, yaitu **World Health Organization (WHO)**, **World Development Indicators (WDI) dari World Bank**, dan **Our World in Data (OWID)**. Data tersebut melalui proses **ETL (Extract, Transform, Load)** menggunakan Apache Airflow, kemudian disimpan dalam MySQL untuk selanjutnya digunakan dalam proses analisis data dan Machine Learning.

Hasil analisis dan model prediksi kemudian diimplementasikan ke dalam aplikasi interaktif menggunakan **Streamlit**.

---

## 🎯 Tujuan Project

Project ini bertujuan untuk:

1. Menganalisis perkembangan angka harapan hidup negara dari tahun 2000–2021.
2. Menganalisis hubungan antara adult mortality dengan angka harapan hidup.
3. Menganalisis hubungan berbagai indikator kesehatan dan penyakit dengan angka harapan hidup.
4. Membangun model Machine Learning untuk memprediksi angka harapan hidup berdasarkan indikator kesehatan.
5. Menyediakan dashboard interaktif untuk membantu pengguna memahami data dan hasil prediksi.

---

## ❓ Pertanyaan Bisnis

Project ini menjawab beberapa pertanyaan bisnis berikut:

1. Bagaimana perkembangan rata-rata angka harapan hidup negara dari tahun 2000 hingga 2021?
2. Bagaimana hubungan antara adult mortality dengan angka harapan hidup?
3. Bagaimana hubungan indikator kesehatan dan penyakit terhadap angka harapan hidup?
4. Seberapa baik angka harapan hidup dapat diprediksi berdasarkan indikator kesehatan menggunakan Machine Learning?

---

## 📊 Sumber Data

Project menggunakan beberapa sumber data:

### 1. World Health Organization (WHO)

Data WHO digunakan sebagai sumber utama untuk variabel:

* Life Expectancy
* Tahun
* Jenis kelamin

Dalam analisis, data yang digunakan difokuskan pada kategori **Both sexes** dengan periode tahun 2000–2021.

### 2. World Development Indicators (WDI)

Data WDI digunakan untuk memperoleh beberapa indikator kesehatan:

* Adult Mortality
* Neonatal Mortality
* Health Expenditure

### 3. Our World in Data (OWID)

Data OWID digunakan sebagai data tambahan untuk indikator penyakit:

* Cardiovascular Death Rate
* Cancer Death Rate
* Diabetes Death Rate

### 4. Countries

Data negara digunakan sebagai tabel referensi yang berisi:

* Country Code
* Country Name
* Region

---

## 🔗 Integrasi Data

Data dari berbagai sumber diintegrasikan menggunakan:

**`country_code` + `year`**

`country_code` digunakan sebagai identifier negara sehingga data WHO, WDI, dan OWID dapat dihubungkan berdasarkan negara dan tahun yang sama.

Secara konseptual hubungan database adalah:

```text
                    countries
                 country_code (PK)
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
      who_table      wdi_table     owid_table
   country_code    country_code   country_code
       + year         + year         + year
```

---

## 🔄 ETL Pipeline

Proses pengolahan data dilakukan menggunakan **Apache Airflow**.

Alur utama project:

```text
WHO
 │
WDI ───────► Extract ──► Transform ──► Load
 │                                      │
OWID                                    ▼
                                  MySQL Database
                                       │
                                       ▼
                                  Export CSV
                                       │
                                       ▼
                                    Google
                                   Colab
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
              Data Wrangling         EDA          Machine Learning
                                                         │
                                                         ▼
                                                     model.pkl
                                                         │
                                                         ▼
                                                     Streamlit
```

### Tahapan ETL

**Extract**

Mengambil data dari sumber WHO, WDI, dan OWID.

**Transform**

Melakukan proses:

* Standardisasi nama kolom
* Pembersihan data
* Penanganan nilai tidak valid
* Pengelompokan data berdasarkan negara dan tahun
* Integrasi beberapa sumber data
* Penanganan missing value

**Load**

Memasukkan data hasil transformasi ke dalam database MySQL.

---

## 🗄️ Database

Database menggunakan **MySQL** dan dikelola/diperiksa melalui phpMyAdmin.

Tabel utama yang digunakan:

```text
countries
who_table
wdi_table
owid_table
```

`countries` berfungsi sebagai tabel referensi negara, sedangkan tiga tabel lainnya menyimpan data dari masing-masing sumber.

---

## 🧹 Data Wrangling

Setelah data dari database diekspor menjadi CSV, proses Data Wrangling dilakukan menggunakan Python di Google Colab.

Beberapa proses yang dilakukan:

* Standardisasi nama kolom
* Menghapus duplikasi
* Menangani nilai yang tidak valid
* Menangani missing values
* Filtering data WHO untuk **Both sexes**
* Penggabungan dataset berdasarkan `country_code` dan `year`
* Validasi data
* Pembentukan final dataset

Final dataset terdiri dari:

```text
country_code
country_name
region
year
life_expectancy
adult_mortality
neonatal_mortality
health_expenditure
cardiovascular_death_rate
cancer_death_rate
diabetes_death_rate
```

---

## 📈 Exploratory Data Analysis (EDA)

EDA dilakukan untuk memahami pola dan hubungan antarvariabel.

Analisis yang dilakukan meliputi:

### Analisis Tren

Melihat perkembangan rata-rata angka harapan hidup dari tahun 2000 hingga 2021.

### Adult Mortality vs Life Expectancy

Menganalisis hubungan antara adult mortality dengan angka harapan hidup.

### Analisis Korelasi

Menganalisis hubungan antara:

* Adult Mortality
* Neonatal Mortality
* Health Expenditure
* Cardiovascular Death Rate
* Cancer Death Rate
* Diabetes Death Rate
* Life Expectancy

Visualisasi menggunakan grafik dan correlation heatmap.

---

## 🤖 Machine Learning

Model Machine Learning digunakan untuk memprediksi **Life Expectancy**.

### Target

```text
life_expectancy
```

### Features

Model menggunakan enam indikator kesehatan:

```text
adult_mortality
neonatal_mortality
health_expenditure
cardiovascular_death_rate
cancer_death_rate
diabetes_death_rate
```

### Algoritma

Algoritma yang digunakan adalah:

**Random Forest Regression**

Random Forest dipilih karena mampu menangani hubungan non-linear antarvariabel dan dapat memberikan informasi mengenai tingkat kepentingan setiap fitur terhadap prediksi.

### Pembagian Data

Digunakan pendekatan berdasarkan waktu:

```text
Training Data : tahun <= 2019
Testing Data  : tahun >= 2020
```

Pendekatan ini digunakan agar data pengujian berasal dari periode waktu yang lebih baru dibandingkan data training.

### Missing Value

Missing value pada fitur ditangani menggunakan **median imputation**.

---

## 📏 Evaluasi Model

Model dievaluasi menggunakan tiga metrik:

### MAE

**Mean Absolute Error (MAE)** digunakan untuk mengukur rata-rata kesalahan absolut antara nilai aktual dan prediksi.

### RMSE

**Root Mean Squared Error (RMSE)** digunakan untuk mengukur besarnya kesalahan prediksi dengan memberikan penalti lebih besar terhadap error yang besar.

### R²

**R-squared (R²)** digunakan untuk melihat seberapa baik model menjelaskan variasi pada target.

Selain metrik tersebut, project juga menampilkan:

* Actual vs Predicted
* Error Distribution
* Feature Importance

Model yang telah dilatih disimpan dalam file:

```text
model.pkl
```

---

## 🌐 Streamlit Application

Model dan dataset kemudian diimplementasikan ke dalam aplikasi interaktif menggunakan Streamlit.

Aplikasi terdiri dari beberapa halaman:

### 🏠 Dashboard

Menampilkan:

* Ringkasan project
* Periode data
* Jumlah negara
* Rata-rata life expectancy
* Tren life expectancy
* Insight utama

### 📊 Data

Menampilkan dataset dan menyediakan filter berdasarkan:

* Tahun
* Region
* Negara

### 📈 EDA

Menampilkan hasil Exploratory Data Analysis berupa:

* Tren life expectancy
* Adult mortality vs life expectancy
* Hubungan berbagai indikator kesehatan

### ❤️ Analisis Kesehatan

Digunakan untuk melihat hubungan antara indikator kesehatan tertentu dengan life expectancy.

### 🤖 Prediksi

Digunakan untuk melakukan prediksi life expectancy berdasarkan enam indikator kesehatan.

Pengguna dapat memasukkan nilai indikator secara manual atau memilih data negara dan tahun tertentu.

### 🧠 Model

Menampilkan informasi mengenai:

* MAE
* RMSE
* R²
* Actual vs Predicted
* Error Distribution
* Feature Importance
* Keterbatasan model

---

## 🛠️ Teknologi yang Digunakan

| Teknologi      | Fungsi                             |
| -------------- | ---------------------------------- |
| Python         | Bahasa pemrograman utama           |
| Pandas         | Data manipulation dan wrangling    |
| NumPy          | Komputasi numerik                  |
| Matplotlib     | Visualisasi data                   |
| Scikit-learn   | Machine Learning                   |
| Joblib         | Penyimpanan model                  |
| Apache Airflow | Orkestrasi ETL                     |
| Docker         | Menjalankan environment Airflow    |
| MySQL          | Database                           |
| phpMyAdmin     | Pengelolaan database               |
| Google Colab   | Data analysis dan Machine Learning |
| Streamlit      | Web application                    |
| GitHub         | Version control dan repository     |

---

## 📁 Struktur Repository

```text
life-expectancy-sdg3/
│
├── app.py
├── requirements.txt
├── model.pkl
│
├── countries.csv
├── who_table.csv
├── wdi_table.csv
├── owid_table.csv
│
├── airflow/
│   └── etl_life_expectancy.py
│
├── notebook/
│   └── analisis_life_expectancy.ipynb
│
└── README.md
```

---

## ▶️ Cara Menjalankan Aplikasi

### 1. Clone repository

```bash
git clone <URL-REPOSITORY-GITHUB>
```

### 2. Masuk ke folder project

```bash
cd life-expectancy-sdg3
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan Streamlit

```bash
streamlit run app.py
```

Setelah dijalankan, aplikasi dapat dibuka melalui browser.

---

## 📦 Requirements

Library utama yang digunakan:

```text
streamlit
pandas
numpy
matplotlib
scikit-learn
joblib
```

Versi lengkap dapat dilihat pada file:

```text
requirements.txt
```

---

## ⚠️ Keterbatasan Project

Beberapa keterbatasan project:

1. Model hanya menggunakan beberapa indikator kesehatan yang tersedia dalam dataset.
2. Prediksi tidak dapat dianggap sebagai hubungan sebab-akibat.
3. Hasil model bergantung pada kualitas dan kelengkapan data.
4. Data yang digunakan memiliki periode sampai tahun 2021.
5. Model Machine Learning merupakan alat analisis dan prediksi, bukan alat untuk menentukan kebijakan kesehatan secara langsung.

---

## 🎯 Kaitan dengan SDG 3

Project ini mendukung **Sustainable Development Goal (SDG) 3: Good Health and Well-being** dengan memanfaatkan data kesehatan untuk:

* memahami kondisi kesehatan populasi,
* menganalisis faktor yang berkaitan dengan angka harapan hidup,
* mengidentifikasi pola indikator kesehatan,
* dan melakukan prediksi angka harapan hidup berdasarkan data kesehatan.

Dengan adanya analisis dan visualisasi interaktif, data dapat digunakan sebagai salah satu pendukung dalam memahami kondisi kesehatan masyarakat secara lebih terstruktur.

---

## 👩‍💻 Author

**Wa Ode Yurismawati**
NIM: **E1E124080**

Program Studi: **Teknik Informatika**

---

## 📌 Project Status

**Completed**

Project mencakup:

* [x] Data collection
* [x] Data integration
* [x] ETL Pipeline
* [x] MySQL Database
* [x] Data Wrangling
* [x] Exploratory Data Analysis
* [x] Machine Learning
* [x] Model Evaluation
* [x] Streamlit Application
* [x] GitHub Repository
