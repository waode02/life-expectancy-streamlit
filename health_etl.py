# ============================================================
# HEALTH ETL PIPELINE
# ============================================================
#
# Sumber API:
# 1. WHO                      (Life Expectancy)
# 2. World Bank / WDI         (Adult mortality, Neonatal mortality, Health expenditure)
# 3. Our World in Data / OWID (Death rate: cardiovascular, cancer, diabetes)
#
# Periode : 2000 - 2021
# Target  : Life Expectancy
# Database: MySQL - db_health (dilihat lewat phpMyAdmin)
#
# Tabel:
# 1. countries
# 2. who_table
# 3. wdi_table
# 4. owid_table
# 5. predictions   (hanya dibuat jika belum ada, isinya TIDAK diubah)
#
# Task Airflow (terpisah):
#   extract   : extract WHO + extract WDI (+ metadata negara)
#   transform : transform WHO + transform WDI + extract/transform OWID
#   load      : create database/tabel -> load countries -> load WHO
#               -> load WDI -> load OWID -> validasi
#
# Data antar-task dititipkan lewat file pickle di folder include/
#
# requirements.txt (Astro / Airflow): pymysql, cryptography
# ============================================================


# ============================================================
# IMPORT
# ============================================================

import io
import os
import pickle
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pymysql
import requests

# Airflow 3 (airflow.sdk + provider standard); jika memakai Airflow 2 otomatis pakai import lama
try:
    from airflow.sdk import DAG
except ImportError:  # Airflow 2.x
    from airflow import DAG

try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.python import PythonOperator


# ============================================================
# KONFIGURASI UMUM
# ============================================================

START_YEAR = 2000
END_YEAR = 2021


# ============================================================
# KONFIGURASI FILE PERANTARA ANTAR TASK
# ============================================================

INCLUDE_DIR = os.getenv("HEALTH_INCLUDE_DIR", "/usr/local/airflow/include")

RAW_FILE = os.path.join(INCLUDE_DIR, "health_raw.pkl")

CLEAN_FILE = os.path.join(INCLUDE_DIR, "health_clean.pkl")


# ============================================================
# KONFIGURASI MYSQL (XAMPP di komputer host, diakses dari Docker)
# Bisa diganti lewat environment variable tanpa mengubah kode.
# ============================================================

MYSQL_HOST = os.getenv("HEALTH_DB_HOST", "host.docker.internal")
MYSQL_PORT = int(os.getenv("HEALTH_DB_PORT", "3306"))
MYSQL_USER = os.getenv("HEALTH_DB_USER", "root")
MYSQL_PASSWORD = os.getenv("HEALTH_DB_PASSWORD", "")
MYSQL_DATABASE = os.getenv("HEALTH_DB_NAME", "db_health")


# ============================================================
# KONFIGURASI WHO
# ============================================================

# Alamat dicoba berurutan. Query pertama hanya mengambil baris tingkat NEGARA.
WHO_ENDPOINTS = [
    "https://ghoapi.azureedge.net/api/WHOSIS_000001",
    "https://ghoapi.who.int/api/WHOSIS_000001",
]

WHO_QUERIES = [
    "?$filter=SpatialDimType%20eq%20'COUNTRY'&$format=json",
    "?$format=json",
]


# ============================================================
# KONFIGURASI WORLD BANK / WDI
# ============================================================

WDI_BASE_URL = "https://api.worldbank.org/v2/country/all/indicator"

WDI_COUNTRY_URL = "https://api.worldbank.org/v2/country"

WDI_INDICATORS = {
    "adult_mortality_female": "SP.DYN.AMRT.FE",
    "adult_mortality_male": "SP.DYN.AMRT.MA",
    "neonatal_mortality": "SH.DYN.NMRT",
    "health_expenditure": "SH.XPD.CHEX.PC.CD"
}


# ============================================================
# KONFIGURASI OWID
# ============================================================

OWID_GRAPHER_BASE = "https://ourworldindata.org/grapher"

OWID_INDICATORS = {
    "cardiovascular_death_rate": "death-rate-from-cardiovascular-disease-ghe",
    "cancer_death_rate": "death-rate-from-cancer",
    "diabetes_death_rate": "death-rate-from-diabetes-ghe"
}

OWID_DATA_API_BASE = "https://api.ourworldindata.org/v1/indicators"

# OWID meminta User-Agent khusus untuk pengunduhan data
OWID_HEADERS = {"User-Agent": "Our World In Data data fetch/1.0"}


# ============================================================
# HELPER: REQUEST JSON
# ============================================================

def get_json(url, params=None, timeout=120, retries=3, headers=None):
    """
    Mengambil data JSON dari API dengan retry.
    Error 4xx (kecuali 429) tidak diulang karena percuma.
    """

    last_error = None

    for attempt in range(1, retries + 1):

        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers=headers
            )

            response.raise_for_status()

            return response.json()

        except Exception as error:

            last_error = error

            print(f"Percobaan {attempt}/{retries} gagal: {error}")

            status = getattr(
                getattr(error, "response", None),
                "status_code",
                None
            )

            if status is not None and 400 <= status < 500 and status != 429:
                break

            if attempt < retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Gagal mengambil data dari {url}"
    ) from last_error


# ============================================================
# HELPER: FILE PERANTARA ANTAR TASK
# ============================================================

def save_pickle(obj, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as file:

        pickle.dump(obj, file)

    print(f"Data disimpan ke {path}")


def load_pickle(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File {path} tidak ditemukan. "
            "Jalankan task sebelumnya terlebih dahulu."
        )

    with open(path, "rb") as file:

        return pickle.load(file)


# ============================================================
# HELPER: DATA UNTUK MYSQL
# ============================================================

def to_db_rows(df, columns):
    """
    DataFrame -> list of tuple yang aman untuk pymysql:
    - NaN / NA menjadi None (pymysql menolak NaN)
    - tipe numpy menjadi tipe Python biasa
    """

    subset = df[columns]

    data = subset.astype(object).where(subset.notna(), None)

    return [
        tuple(row)
        for row in data.itertuples(index=False, name=None)
    ]


def execute_in_chunks(connection, sql, rows, chunk_size=1000):
    """
    Insert bertahap agar tidak melebihi max_allowed_packet MySQL.
    Jika ada yang gagal, seluruh perubahan pada tahap itu dibatalkan.
    """

    cursor = connection.cursor()

    try:

        for start in range(0, len(rows), chunk_size):

            cursor.executemany(
                sql,
                rows[start:start + chunk_size]
            )

        connection.commit()

    except Exception:

        connection.rollback()

        raise

    finally:

        cursor.close()


# ============================================================
# KONEKSI MYSQL
# ============================================================

def get_mysql_connection(with_database=True):

    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE if with_database else None,
        charset="utf8mb4",
        connect_timeout=30
    )


def create_database_if_not_exists():
    """Membuat database db_health bila belum ada (agar muncul di phpMyAdmin)."""

    connection = get_mysql_connection(with_database=False)

    try:

        cursor = connection.cursor()

        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
            "CHARACTER SET utf8mb4"
        )

        connection.commit()

        cursor.close()

    finally:

        connection.close()


# ============================================================
# CREATE TABLE
# ============================================================

def create_tables(connection):

    cursor = connection.cursor()

    # ========================================================
    # 1. COUNTRIES
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS countries (

            country_code VARCHAR(3) PRIMARY KEY,

            country_name VARCHAR(150),

            region VARCHAR(100)

        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ========================================================
    # 2. WHO TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS who_table (

            id INT AUTO_INCREMENT PRIMARY KEY,

            country_code VARCHAR(3) NOT NULL,

            `year` INT NOT NULL,

            sex VARCHAR(20) NOT NULL,

            life_expectancy DECIMAL(10,2),

            `low` DECIMAL(10,2),

            `high` DECIMAL(10,2),

            CONSTRAINT fk_who_country
                FOREIGN KEY (country_code)
                REFERENCES countries(country_code),

            UNIQUE KEY unique_who (country_code, `year`, sex)

        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ========================================================
    # 3. WDI TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wdi_table (

            id INT AUTO_INCREMENT PRIMARY KEY,

            country_code VARCHAR(3) NOT NULL,

            `year` INT NOT NULL,

            adult_mortality DECIMAL(10,2),

            neonatal_mortality DECIMAL(10,2),

            health_expenditure DECIMAL(15,2),

            CONSTRAINT fk_wdi_country
                FOREIGN KEY (country_code)
                REFERENCES countries(country_code),

            UNIQUE KEY unique_wdi (country_code, `year`)

        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ========================================================
    # 4. OWID TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS owid_table (

            id INT AUTO_INCREMENT PRIMARY KEY,

            country_code VARCHAR(3) NOT NULL,

            `year` INT NOT NULL,

            cardiovascular_death_rate DECIMAL(15,4),

            cancer_death_rate DECIMAL(15,4),

            diabetes_death_rate DECIMAL(15,4),

            CONSTRAINT fk_owid_country
                FOREIGN KEY (country_code)
                REFERENCES countries(country_code),

            UNIQUE KEY unique_owid (country_code, `year`)

        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ========================================================
    # 5. PREDICTIONS (hanya dibuat jika belum ada)
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            prediction_id INT AUTO_INCREMENT PRIMARY KEY,

            country_code VARCHAR(3) NOT NULL,

            `year` INT NOT NULL,

            actual_value DECIMAL(10,2),

            predicted_value DECIMAL(10,2),

            CONSTRAINT fk_prediction_country
                FOREIGN KEY (country_code)
                REFERENCES countries(country_code),

            UNIQUE KEY unique_prediction (country_code, `year`)

        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    connection.commit()

    cursor.close()

    print("Semua tabel berhasil dibuat / sudah tersedia.")


# ============================================================
# EXTRACT WHO
# ============================================================

def extract_who():

    print("\n" + "=" * 60)
    print("EXTRACT WHO")
    print("=" * 60)

    last_error = None

    for endpoint in WHO_ENDPOINTS:

        for query in WHO_QUERIES:

            try:

                records = []

                next_url = endpoint + query

                while next_url:

                    data = get_json(next_url)

                    if "value" not in data:
                        raise ValueError(
                            "Respons WHO tidak memiliki key 'value'."
                        )

                    records.extend(data["value"])

                    next_url = data.get("@odata.nextLink")

                if records:

                    df = pd.DataFrame(records)

                    print(f"WHO raw data: {len(df)} baris")

                    return df

                print(f"Respons WHO kosong: {endpoint}{query}")

            except Exception as error:

                last_error = error

                print(f"WHO gagal dari {endpoint}{query}: {error}")

    raise RuntimeError(
        "WHO API tidak dapat diakses / tidak mengembalikan data."
    ) from last_error


def filter_who_countries(df):
    """
    Hanya baris tingkat NEGARA (kode ISO3).
    Kode region WHO seperti AFR, AMR, EUR, EMR, WPR juga 3 huruf,
    sehingga harus dibuang lewat SpatialDimType.
    """

    df = df.copy()

    if "SpatialDimType" in df.columns:
        df = df[df["SpatialDimType"] == "COUNTRY"]

    df["SpatialDim"] = (
        df["SpatialDim"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return df[df["SpatialDim"].str.match(r"^[A-Z]{3}$", na=False)]


# ============================================================
# TRANSFORM WHO
# ============================================================

def transform_who(df):

    print("\n" + "=" * 60)
    print("TRANSFORM WHO")
    print("=" * 60)

    required_columns = [
        "SpatialDim",
        "TimeDim",
        "Dim1",
        "NumericValue",
        "Low",
        "High"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom WHO tidak ditemukan: {missing_columns}"
        )

    # --------------------------------------------------------
    # Hanya negara
    # --------------------------------------------------------

    df = filter_who_countries(df)

    # --------------------------------------------------------
    # Rename
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "SpatialDim": "country_code",
            "TimeDim": "year",
            "Dim1": "sex",
            "NumericValue": "life_expectancy",
            "Low": "low",
            "High": "high"
        }
    )

    # --------------------------------------------------------
    # Mapping SEX
    # --------------------------------------------------------

    sex_mapping = {
        "SEX_BTSX": "Both sexes",
        "SEX_FMLE": "Female",
        "SEX_MLE": "Male"
    }

    df["sex"] = df["sex"].map(sex_mapping)

    # --------------------------------------------------------
    # Convert numeric
    # --------------------------------------------------------

    for column in ["year", "life_expectancy", "low", "high"]:

        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    # --------------------------------------------------------
    # Tahun 2000-2021
    # --------------------------------------------------------

    df = df[df["year"].between(START_YEAR, END_YEAR)]

    # --------------------------------------------------------
    # Hapus data invalid (kosong / tidak wajar)
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "country_code",
            "year",
            "sex",
            "life_expectancy"
        ]
    )

    df = df[df["life_expectancy"] > 0]

    df["year"] = df["year"].astype(int)

    # --------------------------------------------------------
    # Hapus duplicate
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["country_code", "year", "sex"]
    )

    # --------------------------------------------------------
    # Kolom akhir
    # --------------------------------------------------------

    df = df[
        [
            "country_code",
            "year",
            "sex",
            "life_expectancy",
            "low",
            "high"
        ]
    ]

    print(f"WHO setelah transform: {len(df)} baris")

    return df


# ============================================================
# EXTRACT WDI
# ============================================================

def extract_wdi_indicator(indicator_code):

    print(f"\nMengambil WDI: {indicator_code}")

    all_data = []

    page = 1

    while True:

        url = f"{WDI_BASE_URL}/{indicator_code}"

        params = {
            "format": "json",
            "per_page": 1000,
            "page": page,
            "date": f"{START_YEAR}:{END_YEAR}"
        }

        data = get_json(url, params=params, timeout=120)

        # ----------------------------------------------------
        # Format WDI: [metadata, data]
        # ----------------------------------------------------

        if (
            not isinstance(data, list)
            or len(data) < 2
            or data[1] is None
        ):
            raise ValueError(
                "Format respons WDI tidak valid "
                f"untuk {indicator_code}: {str(data)[:200]}"
            )

        metadata = data[0]

        rows = data[1]

        if not rows:
            break

        all_data.extend(rows)

        total_pages = int(metadata.get("pages", page))

        if page >= total_pages:
            break

        page += 1

    df = pd.DataFrame(all_data)

    print(f"WDI {indicator_code}: {len(df)} baris")

    return df


def extract_wdi_countries():
    """
    Metadata negara World Bank: nama negara, region, dan penanda agregat.
    (Respons indikator WDI TIDAK memuat region, jadi diambil dari sini.)
    Jika gagal, ETL tetap jalan memakai nama negara dari respons indikator.
    """

    print("\nMengambil metadata negara World Bank...")

    meta = {}

    try:

        page = 1

        while True:

            data = get_json(
                WDI_COUNTRY_URL,
                params={"format": "json", "per_page": 400, "page": page}
            )

            if not isinstance(data, list) or len(data) < 2 or not data[1]:
                break

            for item in data[1]:

                code = str(item.get("id") or "").upper().strip()

                region = (item.get("region") or {}).get("value")

                region = region.strip() if region else None

                meta[code] = {
                    "name": item.get("name"),
                    "region": region,
                    "aggregate": region == "Aggregates"
                }

            if page >= int(data[0].get("pages", 1)):
                break

            page += 1

    except Exception as error:

        print(f"Peringatan: metadata negara World Bank gagal diambil: {error}")

        return {}

    print(f"Metadata negara: {len(meta)} entri")

    return meta


# ============================================================
# TRANSFORM WDI
# ============================================================

def prepare_wdi_frame(df, column_name):
    """Ambil country_code, year, value dari respons WDI dan bersihkan."""

    out = df[["countryiso3code", "date", "value"]].copy()

    out = out.rename(
        columns={
            "countryiso3code": "country_code",
            "date": "year",
            "value": column_name
        }
    )

    out["year"] = pd.to_numeric(out["year"], errors="coerce")

    out[column_name] = pd.to_numeric(out[column_name], errors="coerce")

    # Nilai <= 0 dianggap tidak tersedia
    out.loc[out[column_name] <= 0, column_name] = np.nan

    out["country_code"] = (
        out["country_code"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    out = out[out["country_code"].str.match(r"^[A-Z]{3}$", na=False)]

    out = out.dropna(subset=["year"])

    out["year"] = out["year"].astype(int)

    return out.drop_duplicates(subset=["country_code", "year"])


def transform_wdi(adult_female, adult_male, neonatal, health):

    print("\n" + "=" * 60)
    print("TRANSFORM WDI")
    print("=" * 60)

    # ========================================================
    # ADULT MORTALITY FEMALE + MALE
    # ========================================================

    female = prepare_wdi_frame(adult_female, "adult_mortality_female")

    male = prepare_wdi_frame(adult_male, "adult_mortality_male")

    adult = female.merge(
        male,
        on=["country_code", "year"],
        how="outer"
    )

    # Adult mortality = (Female + Male) / 2
    # (jika hanya satu yang tersedia, pakai yang ada)
    adult["adult_mortality"] = adult[
        ["adult_mortality_female", "adult_mortality_male"]
    ].mean(axis=1)

    adult = adult[["country_code", "year", "adult_mortality"]]

    # ========================================================
    # NEONATAL MORTALITY & HEALTH EXPENDITURE
    # ========================================================

    neonatal = prepare_wdi_frame(neonatal, "neonatal_mortality")

    health = prepare_wdi_frame(health, "health_expenditure")

    # ========================================================
    # MERGE SEMUA WDI
    # ========================================================

    df = adult.merge(
        neonatal,
        on=["country_code", "year"],
        how="outer"
    )

    df = df.merge(
        health,
        on=["country_code", "year"],
        how="outer"
    )

    # ========================================================
    # TAHUN
    # ========================================================

    df = df[df["year"].between(START_YEAR, END_YEAR)]

    # ========================================================
    # NUMERIC
    # ========================================================

    numeric_columns = [
        "adult_mortality",
        "neonatal_mortality",
        "health_expenditure"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Baris tanpa satu pun indikator tidak berguna
    df = df.dropna(subset=numeric_columns, how="all")

    # ========================================================
    # DUPLICATE & KOLOM AKHIR
    # ========================================================

    df = df.drop_duplicates(subset=["country_code", "year"])

    df = df[
        [
            "country_code",
            "year",
            "adult_mortality",
            "neonatal_mortality",
            "health_expenditure"
        ]
    ]

    print(f"WDI setelah transform: {len(df)} baris")

    return df


# ============================================================
# EXTRACT OWID
# ============================================================

def clean_owid_frame(df):
    """df dengan kolom country_code, year, value -> bersih."""

    df = df.copy()

    df["country_code"] = (
        df["country_code"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Filter tahun
    df = df[df["year"].between(START_YEAR, END_YEAR)]

    # Hanya ISO3 (OWID_WRL, kode kosong, dll. otomatis terbuang)
    df = df[df["country_code"].str.match(r"^[A-Z]{3}$", na=False)]

    # Hapus missing
    df = df.dropna(subset=["country_code", "year", "value"])

    df["year"] = df["year"].astype(int)

    return df.drop_duplicates(subset=["country_code", "year"])


def extract_owid_via_api(slug):
    """Cara utama: metadata chart -> variable ID -> data.json."""

    # ========================================================
    # STEP 1: metadata chart
    # ========================================================

    metadata = get_json(
        f"{OWID_GRAPHER_BASE}/{slug}.metadata.json",
        timeout=120,
        headers=OWID_HEADERS
    )

    # ========================================================
    # STEP 2: cari OWID variable ID
    # ========================================================

    columns = metadata.get("columns", {})

    variable_id = None

    for column_name, column_info in columns.items():

        if column_info.get("owidVariableId") is not None:

            variable_id = int(column_info["owidVariableId"])

            break

    if variable_id is None:
        raise ValueError(
            f"OWID variable ID tidak ditemukan untuk slug {slug}"
        )

    print(f"OWID variable ID: {variable_id}")

    # ========================================================
    # STEP 3: data indikator
    # ========================================================

    data = get_json(
        f"{OWID_DATA_API_BASE}/{variable_id}.data.json",
        timeout=120,
        headers=OWID_HEADERS
    )

    for key in ["years", "entities", "values"]:

        if key not in data:
            raise ValueError(
                f"Key '{key}' tidak ditemukan pada data OWID {variable_id}"
            )

    years = data["years"]
    entities = data["entities"]
    values = data["values"]

    if not (len(years) == len(entities) == len(values)):
        raise ValueError(
            "Panjang years, entities, dan values OWID tidak sama."
        )

    # ========================================================
    # STEP 4-5: metadata indikator -> mapping entity
    # ========================================================

    indicator_metadata = get_json(
        f"{OWID_DATA_API_BASE}/{variable_id}.metadata.json",
        timeout=120,
        headers=OWID_HEADERS
    )

    entity_values = (
        indicator_metadata
        .get("dimensions", {})
        .get("entities", {})
        .get("values", [])
    )

    entity_code_map = {
        entity.get("id"): entity.get("code")
        for entity in entity_values
    }

    # ========================================================
    # STEP 6: DataFrame
    # ========================================================

    df = pd.DataFrame({
        "entity_id": entities,
        "year": years,
        "value": values
    })

    df["country_code"] = df["entity_id"].map(entity_code_map)

    # entity tanpa kode -> NaN -> tidak lolos filter ISO3
    df["country_code"] = df["country_code"].astype(object).where(
        df["country_code"].notna(), ""
    )

    return df[["country_code", "year", "value"]]


def extract_owid_via_csv(slug):
    """
    Cadangan: unduhan CSV resmi OWID
    (kolom: Entity, Code, Year, <nama indikator>).
    """

    response = requests.get(
        f"{OWID_GRAPHER_BASE}/{slug}.csv",
        params={"v": 1, "csvType": "full", "useColumnShortNames": "false"},
        headers=OWID_HEADERS,
        timeout=120
    )

    response.raise_for_status()

    raw = pd.read_csv(io.StringIO(response.text))

    if raw.shape[1] < 4:
        raise ValueError(f"Format CSV OWID tidak sesuai untuk {slug}")

    df = pd.DataFrame({
        "country_code": raw.iloc[:, 1],
        "year": raw.iloc[:, 2],
        "value": raw.iloc[:, 3]
    })

    df["country_code"] = df["country_code"].fillna("")

    return df


def extract_owid_indicator(slug):

    print("\n" + "-" * 60)
    print(f"EXTRACT OWID: {slug}")

    try:

        df = extract_owid_via_api(slug)

    except Exception as error:

        print(f"OWID API gagal untuk {slug}: {error}")
        print("Mencoba unduhan CSV resmi OWID...")

        df = extract_owid_via_csv(slug)

    df = clean_owid_frame(df)

    print(f"OWID {slug}: {len(df)} baris")

    return df


# ============================================================
# TRANSFORM OWID
# ============================================================

def transform_owid():

    print("\n" + "=" * 60)
    print("TRANSFORM OWID")
    print("=" * 60)

    transformed_data = []

    # ========================================================
    # Ambil satu per satu indikator.
    # OWID hanya data pelengkap: jika satu indikator gagal,
    # ETL tetap berjalan dan kolomnya dibiarkan kosong (NULL).
    # ========================================================

    for column_name, slug in OWID_INDICATORS.items():

        try:

            df = extract_owid_indicator(slug)

        except Exception as error:

            print(f"PERINGATAN: indikator OWID '{slug}' dilewati: {error}")

            continue

        df = df[["country_code", "year", "value"]].copy()

        df = df.rename(columns={"value": column_name})

        transformed_data.append(df)

    final_columns = [
        "country_code",
        "year",
        "cardiovascular_death_rate",
        "cancer_death_rate",
        "diabetes_death_rate"
    ]

    if not transformed_data:

        print("PERINGATAN: tidak ada data OWID yang berhasil diambil.")

        return pd.DataFrame(columns=final_columns)

    # ========================================================
    # Merge indikator
    # ========================================================

    df = transformed_data[0]

    for next_df in transformed_data[1:]:

        df = df.merge(
            next_df,
            on=["country_code", "year"],
            how="outer"
        )

    # Kolom indikator yang dilewati diisi kosong
    for column in final_columns:

        if column not in df.columns:
            df[column] = np.nan

    # ========================================================
    # Filter tahun, duplicate, kolom akhir
    # ========================================================

    df = df[df["year"].between(START_YEAR, END_YEAR)]

    df = df.drop_duplicates(subset=["country_code", "year"])

    df = df[final_columns]

    print(f"OWID final: {len(df)} baris")

    return df


# ============================================================
# LOAD COUNTRIES
# ============================================================

def build_countries(who_raw, wdi_raw_list, country_meta):
    """
    Negara valid = negara WHO (tingkat COUNTRY) yang juga ada di World Bank,
    bukan agregat (WLD, EUU, dst.). Nama & region dari metadata World Bank.
    """

    # --- WHO: kode negara + region WHO (cadangan)
    who_countries = filter_who_countries(who_raw)

    who_codes = set(who_countries["SpatialDim"])

    who_regions = {}

    if "ParentLocation" in who_countries.columns:

        pairs = who_countries[["SpatialDim", "ParentLocation"]].dropna()

        who_regions = dict(zip(pairs["SpatialDim"], pairs["ParentLocation"]))

    # --- WDI: kode negara + nama dari respons indikator (cadangan)
    wdi_names = {}

    for raw in wdi_raw_list:

        if "countryiso3code" not in raw.columns:
            continue

        for code, country in zip(
            raw["countryiso3code"],
            raw["country"] if "country" in raw.columns else [None] * len(raw)
        ):

            code = str(code or "").upper().strip()

            if len(code) != 3:
                continue

            name = country.get("value") if isinstance(country, dict) else None

            wdi_names.setdefault(code, name)

    valid_codes = {
        code
        for code in (who_codes & set(wdi_names))
        if not country_meta.get(code, {}).get("aggregate", False)
    }

    rows = []

    for code in sorted(valid_codes):

        meta = country_meta.get(code, {})

        rows.append(
            {
                "country_code": code,
                "country_name": meta.get("name") or wdi_names.get(code),
                "region": meta.get("region") or who_regions.get(code)
            }
        )

    return pd.DataFrame(
        rows,
        columns=["country_code", "country_name", "region"]
    )


def load_countries(connection, who_raw, wdi_raw_list, country_meta):

    print("\n" + "=" * 60)
    print("LOAD COUNTRIES")
    print("=" * 60)

    countries = build_countries(who_raw, wdi_raw_list, country_meta)

    if countries.empty:
        raise ValueError(
            "Tidak ada negara yang cocok antara WHO dan World Bank."
        )

    sql = """
        INSERT INTO countries (country_code, country_name, region)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            country_name = VALUES(country_name),
            region = VALUES(region)
    """

    execute_in_chunks(
        connection,
        sql,
        to_db_rows(countries, ["country_code", "country_name", "region"])
    )

    print(f"Countries loaded: {len(countries)} negara")

    return set(countries["country_code"])


def keep_valid_countries(df, valid_codes, label):
    """Buang baris yang negaranya tidak ada di tabel countries (agar FK aman)."""

    before = len(df)

    df = df[df["country_code"].isin(valid_codes)]

    if before != len(df):
        print(
            f"{label}: {before - len(df)} baris dibuang "
            "(bukan negara pada tabel countries)"
        )

    return df


# ============================================================
# LOAD WHO
# ============================================================

def load_who(connection, df, valid_codes):

    print("\n" + "=" * 60)
    print("LOAD WHO")
    print("=" * 60)

    df = keep_valid_countries(df, valid_codes, "WHO")

    sql = """
        INSERT INTO who_table
            (country_code, `year`, sex, life_expectancy, `low`, `high`)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            life_expectancy = VALUES(life_expectancy),
            `low` = VALUES(`low`),
            `high` = VALUES(`high`)
    """

    execute_in_chunks(
        connection,
        sql,
        to_db_rows(
            df,
            ["country_code", "year", "sex", "life_expectancy", "low", "high"]
        )
    )

    print(f"WHO loaded: {len(df)} rows")


# ============================================================
# LOAD WDI
# ============================================================

def load_wdi(connection, df, valid_codes):

    print("\n" + "=" * 60)
    print("LOAD WDI")
    print("=" * 60)

    df = keep_valid_countries(df, valid_codes, "WDI")

    sql = """
        INSERT INTO wdi_table
            (country_code, `year`, adult_mortality,
             neonatal_mortality, health_expenditure)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            adult_mortality = VALUES(adult_mortality),
            neonatal_mortality = VALUES(neonatal_mortality),
            health_expenditure = VALUES(health_expenditure)
    """

    execute_in_chunks(
        connection,
        sql,
        to_db_rows(
            df,
            ["country_code", "year", "adult_mortality",
             "neonatal_mortality", "health_expenditure"]
        )
    )

    print(f"WDI loaded: {len(df)} rows")


# ============================================================
# LOAD OWID
# ============================================================

def load_owid(connection, df, valid_codes):

    print("\n" + "=" * 60)
    print("LOAD OWID")
    print("=" * 60)

    df = keep_valid_countries(df, valid_codes, "OWID")

    sql = """
        INSERT INTO owid_table
            (country_code, `year`, cardiovascular_death_rate,
             cancer_death_rate, diabetes_death_rate)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            cardiovascular_death_rate = VALUES(cardiovascular_death_rate),
            cancer_death_rate = VALUES(cancer_death_rate),
            diabetes_death_rate = VALUES(diabetes_death_rate)
    """

    execute_in_chunks(
        connection,
        sql,
        to_db_rows(
            df,
            ["country_code", "year", "cardiovascular_death_rate",
             "cancer_death_rate", "diabetes_death_rate"]
        )
    )

    print(f"OWID loaded: {len(df)} rows")


# ============================================================
# VALIDATION
# ============================================================

def validate_database(connection):

    print("\n" + "=" * 60)
    print("VALIDATION DATABASE")
    print("=" * 60)

    cursor = connection.cursor()

    # ========================================================
    # ROW COUNT
    # ========================================================

    tables = [
        "countries",
        "who_table",
        "wdi_table",
        "owid_table",
        "predictions"
    ]

    counts = {}

    for table in tables:

        cursor.execute(f"SELECT COUNT(*) FROM {table}")

        counts[table] = cursor.fetchone()[0]

        print(f"{table}: {counts[table]} rows")

    # ========================================================
    # VALIDASI FK
    # ========================================================

    invalid = {}

    for table in ["who_table", "wdi_table", "owid_table"]:

        cursor.execute(f"""
            SELECT COUNT(*)
            FROM {table} t
            LEFT JOIN countries c
                ON t.country_code = c.country_code
            WHERE c.country_code IS NULL
        """)

        invalid[table] = cursor.fetchone()[0]

        print(f"{table} invalid FK: {invalid[table]}")

    cursor.close()

    if any(value > 0 for value in invalid.values()):
        raise ValueError(
            "VALIDATION GAGAL: ditemukan foreign key "
            "yang tidak memiliki pasangan pada countries."
        )

    # Data inti (negara, WHO, WDI) tidak boleh kosong
    for table in ["countries", "who_table", "wdi_table"]:

        if counts[table] == 0:
            raise ValueError(f"VALIDATION GAGAL: tabel {table} kosong.")

    if counts["owid_table"] == 0:
        print("PERINGATAN: owid_table kosong (data OWID tidak berhasil diambil).")

    print("VALIDATION BERHASIL.")


# ============================================================
# TASK 1: EXTRACT
# (extract WHO + extract WDI + metadata negara)
# ============================================================

def run_extract():

    print("\n")
    print("=" * 70)
    print("START TASK EXTRACT")
    print("=" * 70)

    # ========================================================
    # EXTRACT WHO
    # ========================================================

    who_raw = extract_who()

    # ========================================================
    # EXTRACT WDI
    # ========================================================

    adult_female_raw = extract_wdi_indicator("SP.DYN.AMRT.FE")

    adult_male_raw = extract_wdi_indicator("SP.DYN.AMRT.MA")

    neonatal_raw = extract_wdi_indicator("SH.DYN.NMRT")

    health_raw = extract_wdi_indicator("SH.XPD.CHEX.PC.CD")

    country_meta = extract_wdi_countries()

    # ========================================================
    # SIMPAN HASIL EXTRACT UNTUK TASK TRANSFORM & LOAD
    # ========================================================

    save_pickle(
        {
            "who_raw": who_raw,
            "adult_female_raw": adult_female_raw,
            "adult_male_raw": adult_male_raw,
            "neonatal_raw": neonatal_raw,
            "health_raw": health_raw,
            "country_meta": country_meta
        },
        RAW_FILE
    )

    print("\n" + "=" * 70)
    print("TASK EXTRACT SELESAI")
    print("=" * 70)


# ============================================================
# TASK 2: TRANSFORM
# (transform WHO + transform WDI + extract/transform OWID)
# ============================================================

def run_transform():

    print("\n")
    print("=" * 70)
    print("START TASK TRANSFORM")
    print("=" * 70)

    raw = load_pickle(RAW_FILE)

    # ========================================================
    # TRANSFORM WHO
    # ========================================================

    who_df = transform_who(raw["who_raw"])

    # ========================================================
    # TRANSFORM WDI
    # ========================================================

    wdi_df = transform_wdi(
        raw["adult_female_raw"],
        raw["adult_male_raw"],
        raw["neonatal_raw"],
        raw["health_raw"]
    )

    # ========================================================
    # EXTRACT + TRANSFORM OWID
    # ========================================================

    owid_df = transform_owid()

    # ========================================================
    # SIMPAN HASIL TRANSFORM UNTUK TASK LOAD
    # ========================================================

    save_pickle(
        {
            "who_df": who_df,
            "wdi_df": wdi_df,
            "owid_df": owid_df
        },
        CLEAN_FILE
    )

    print("\n" + "=" * 70)
    print("TASK TRANSFORM SELESAI")
    print("=" * 70)


# ============================================================
# TASK 3: LOAD
# (create database/tabel -> load countries/WHO/WDI/OWID -> validasi)
# ============================================================

def run_load():

    print("\n")
    print("=" * 70)
    print("START TASK LOAD")
    print("=" * 70)

    raw = load_pickle(RAW_FILE)

    clean = load_pickle(CLEAN_FILE)

    # ========================================================
    # CREATE DATABASE
    # ========================================================

    create_database_if_not_exists()

    connection = get_mysql_connection()

    try:

        # ====================================================
        # CREATE TABLE
        # ====================================================

        create_tables(connection)

        # ====================================================
        # LOAD COUNTRIES
        # ====================================================

        valid_codes = load_countries(
            connection,
            raw["who_raw"],
            [
                raw["adult_female_raw"],
                raw["adult_male_raw"],
                raw["neonatal_raw"],
                raw["health_raw"]
            ],
            raw["country_meta"]
        )

        # ====================================================
        # LOAD WHO, WDI, OWID
        # ====================================================

        load_who(connection, clean["who_df"], valid_codes)

        load_wdi(connection, clean["wdi_df"], valid_codes)

        load_owid(connection, clean["owid_df"], valid_codes)

        # ====================================================
        # VALIDATION
        # ====================================================

        validate_database(connection)

        print("\n")
        print("=" * 70)
        print("HEALTH ETL BERHASIL - buka phpMyAdmin > database "
              f"'{MYSQL_DATABASE}'")
        print("=" * 70)

    finally:

        connection.close()


# ============================================================
# AIRFLOW DAG
# ============================================================

with DAG(

    dag_id="health_etl",

    start_date=datetime(2025, 1, 1),

    schedule=None,

    catchup=False,

    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=1)
    },

    tags=["health", "etl", "who", "wdi", "owid"]

) as dag:

    extract = PythonOperator(

        task_id="extract",

        python_callable=run_extract

    )

    transform = PythonOperator(

        task_id="transform",

        python_callable=run_transform

    )

    load = PythonOperator(

        task_id="load",

        python_callable=run_load

    )

    extract >> transform >> load