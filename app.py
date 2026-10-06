"""
Analisis dan Prediksi Angka Harapan Hidup Negara — SDG 3 (Good Health and Well-being)

Versi ringan:
- Grafik memakai Altair (sudah ikut terpasang bersama Streamlit) -> tidak perlu matplotlib.
- Model Random Forest dibaca dari model_rf.npz (array numpy), jadi scikit-learn/scipy TIDAK perlu dipasang.
  File itu hasil konversi dari model.pkl; prediksinya identik (selisih 0.0).
- Data dan hasil prediksi di-cache, model TIDAK dilatih ulang.
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------------
st.set_page_config(page_title="Angka Harapan Hidup | SDG 3", page_icon="❤️", layout="wide")

BASE = Path(__file__).parent
TARGET = "life_expectancy"
FEATURES = [
    "adult_mortality",
    "neonatal_mortality",
    "health_expenditure",
    "cardiovascular_death_rate",
    "cancer_death_rate",
    "diabetes_death_rate",
]
LABEL = {
    "life_expectancy": "Life Expectancy (tahun)",
    "adult_mortality": "Adult Mortality",
    "neonatal_mortality": "Neonatal Mortality",
    "health_expenditure": "Health Expenditure",
    "cardiovascular_death_rate": "Cardiovascular Death Rate",
    "cancer_death_rate": "Cancer Death Rate",
    "diabetes_death_rate": "Diabetes Death Rate",
}
KETERANGAN = {
    "adult_mortality": "Peluang kematian usia dewasa (per 1.000 penduduk dewasa)",
    "neonatal_mortality": "Kematian bayi baru lahir (per 1.000 kelahiran hidup)",
    "health_expenditure": "Pengeluaran kesehatan per kapita (US$)",
    "cardiovascular_death_rate": "Angka kematian akibat penyakit kardiovaskular",
    "cancer_death_rate": "Angka kematian akibat kanker",
    "diabetes_death_rate": "Angka kematian akibat diabetes",
}
TAHUN_UJI_MULAI = 2020  # sama dengan notebook: latih <= 2019, uji >= 2020
BIRU, ORANYE, HIJAU, MERAH = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"


# ------------------------------------------------------------------
# LOAD DATA (sama dengan tahap Wrangling di notebook)
# ------------------------------------------------------------------
@st.cache_data
def load_raw():
    """Baca 4 CSV apa adanya (nama kolom dibuat huruf kecil)."""
    def baca(nama):
        df = pd.read_csv(
            BASE / f"{nama}.csv",
            encoding="utf-8-sig",
            na_values=["", "NULL", "null", "NaN", "nan", "N/A"],
        )
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    return {n: baca(n) for n in ["countries", "who_table", "wdi_table", "owid_table"]}


def _kode(df):
    df = df.copy()
    df["country_code"] = df["country_code"].astype(str).str.strip().str.upper()
    return df


@st.cache_data
def build_final():
    """Cleaning + JOIN: WHO (target) -> WDI -> OWID -> countries, key = country_code + year."""
    raw = load_raw()

    countries = _kode(raw["countries"]).drop_duplicates("country_code")
    countries = countries[["country_code", "country_name", "region"]]

    who = _kode(raw["who_table"])
    who = who[who["sex"].astype(str).str.lower().str.contains("both")].copy()
    who.loc[(who[TARGET] <= 0) | (who[TARGET] > 120), TARGET] = np.nan
    who = who.groupby(["country_code", "year"], as_index=False)[TARGET].mean()

    def siapkan(nama, kolom):
        d = _kode(raw[nama])
        for c in kolom:
            d.loc[d[c] < 0, c] = np.nan
        return d.groupby(["country_code", "year"], as_index=False)[kolom].mean()

    wdi = siapkan("wdi_table", ["adult_mortality", "neonatal_mortality", "health_expenditure"])
    owid = siapkan("owid_table", ["cardiovascular_death_rate", "cancer_death_rate", "diabetes_death_rate"])

    df = (
        who.merge(wdi, on=["country_code", "year"], how="left")
        .merge(owid, on=["country_code", "year"], how="left")
        .merge(countries, on="country_code", how="left")
    )
    df["country_name"] = df["country_name"].fillna(df["country_code"])
    df["region"] = df["region"].fillna("Unknown")
    df = df.dropna(subset=[TARGET])
    df = df[["country_code", "country_name", "region", "year", TARGET] + FEATURES]
    return df.sort_values(["country_code", "year"]).reset_index(drop=True)


@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    """Baca model_rf.npz: pohon-pohon Random Forest dalam bentuk array numpy + metadata."""
    z = np.load(BASE / "model_rf.npz", allow_pickle=False)
    paket = json.loads(str(z["meta"]))
    paket.update({k: z[k] for k in ["left", "right", "feature", "threshold", "value", "roots", "importance"]})
    return paket


def get_paket():
    try:
        return load_model(), None
    except Exception as e:  # noqa: BLE001
        return None, e


def predict_rf(paket, X):
    """Prediksi Random Forest (rata-rata semua pohon) tanpa scikit-learn.

    X = DataFrame berisi kolom fitur (boleh ada NaN -> diisi median data latih, sama seperti SimpleImputer).
    """
    fitur = paket["fitur"]
    A = X[fitur].astype(float).fillna(pd.Series(paket["median_fitur"])).to_numpy(dtype=np.float32)
    L, R, F, T = paket["left"], paket["right"], paket["feature"], paket["threshold"]
    n = A.shape[0]
    idx = np.repeat(paket["roots"][:, None], n, axis=1)   # posisi tiap pohon (jumlah pohon x jumlah baris)
    kolom = np.arange(n)[None, :]
    while True:
        daun = L[idx] == -1
        if daun.all():
            break
        kiri = A[kolom, F[idx]] <= T[idx]
        idx = np.where(daun, idx, np.where(kiri, L[idx], R[idx]))
    return paket["value"][idx].mean(axis=0)


@st.cache_data(show_spinner="Menghitung prediksi data uji...")
def hasil_uji():
    """Prediksi model pada data uji (tahun >= 2020). Di-cache supaya hanya dihitung sekali."""
    paket = load_model()
    uji = build_final().query("year >= @TAHUN_UJI_MULAI").copy()
    if uji.empty:
        return uji
    uji["prediksi"] = predict_rf(paket, uji)
    uji["error"] = uji["prediksi"] - uji[TARGET]
    uji["abs_error"] = uji["error"].abs()
    return uji


# ------------------------------------------------------------------
# FUNGSI BANTU
# ------------------------------------------------------------------
def f2(x, nd=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:,.{nd}f}"


def corr_desc(r):
    if pd.isna(r):
        return "tidak dapat dihitung"
    a = abs(r)
    kuat = ("sangat kuat" if a >= 0.8 else "kuat" if a >= 0.6 else "sedang" if a >= 0.4
            else "lemah" if a >= 0.2 else "sangat lemah")
    return f"{kuat} dan {'positif' if r > 0 else 'negatif'}"


def spearman(a, b):
    """Korelasi Spearman = Pearson pada peringkat (tanpa scipy)."""
    d = pd.concat([a, b], axis=1).dropna()
    return d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())


def insight(teks):
    st.info("**Insight:** " + teks)


def tampil(chart):
    """Tampilkan grafik Altair selebar kolom (aman untuk versi Streamlit lama maupun baru)."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # Streamlit < 1.51 belum punya argumen width pada altair_chart
        st.altair_chart(chart, use_container_width=True)


def sumbu_tahun(judul="Tahun"):
    return alt.X("year:Q", title=judul, scale=alt.Scale(zero=False), axis=alt.Axis(format="d", tickMinStep=1))


def line_band(yearly, tinggi=330):
    d = yearly.reset_index()
    d["lo"] = d["mean"] - d["std"].fillna(0)
    d["hi"] = d["mean"] + d["std"].fillna(0)
    base = alt.Chart(d).encode(x=sumbu_tahun())
    y_skala = alt.Scale(zero=False)
    band = base.mark_area(opacity=0.15, color=BIRU).encode(
        y=alt.Y("lo:Q", scale=y_skala, title="Life expectancy (tahun)"), y2="hi:Q")
    garis = base.mark_line(point=True, color=BIRU).encode(
        y=alt.Y("mean:Q", scale=y_skala), tooltip=["year:Q", alt.Tooltip("mean:Q", format=".2f")])
    return (band + garis).properties(height=tinggi)


def hbar(d, kategori, nilai, warna, tinggi=300, judul=""):
    return alt.Chart(d).mark_bar(color=warna).encode(
        x=alt.X(f"{nilai}:Q", title=LABEL.get(nilai, nilai)),
        y=alt.Y(f"{kategori}:N", sort="-x", title=None),
        tooltip=[kategori, alt.Tooltip(f"{nilai}:Q", format=".2f")],
    ).properties(height=tinggi, title=judul)


def scatter_reg(d, x, y, color=None, tinggi=380, judul=""):
    """Scatter + garis regresi linear (dihitung Altair di browser)."""
    kolom = [x, y] + ([color] if color else [])
    d = d[kolom].dropna(subset=[x, y])
    base = alt.Chart(d)
    pts = base.mark_circle(size=28, opacity=0.35).encode(
        x=alt.X(f"{x}:Q", title=LABEL.get(x, x), scale=alt.Scale(zero=False)),
        y=alt.Y(f"{y}:Q", title=LABEL.get(y, y), scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip(f"{x}:Q", format=".2f"), alt.Tooltip(f"{y}:Q", format=".2f")],
    )
    if color:
        pts = pts.encode(color=alt.Color(f"{color}:N", legend=alt.Legend(orient="bottom", title=None, columns=2)))
    garis = base.transform_regression(x, y).mark_line(color=MERAH, size=2.5).encode(x=f"{x}:Q", y=f"{y}:Q")
    return (pts + garis).properties(height=tinggi, title=judul)


def heatmap(corr):
    nama = {c: LABEL[c].replace(" (tahun)", "") for c in corr.columns}
    d = corr.rename(index=nama, columns=nama).stack().reset_index()
    d.columns = ["x", "y", "r"]
    urut = list(nama.values())
    base = alt.Chart(d).encode(
        x=alt.X("x:O", sort=urut, title=None, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("y:O", sort=urut, title=None),
    )
    rect = base.mark_rect().encode(
        color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True), title="r"))
    teks = base.mark_text(fontSize=11).encode(
        text=alt.Text("r:Q", format=".2f"),
        color=alt.condition("abs(datum.r) > 0.6", alt.value("white"), alt.value("black")))
    return (rect + teks).properties(height=420)


# ------------------------------------------------------------------
# MUAT DATA
# ------------------------------------------------------------------
try:
    RAW = load_raw()
    DF = build_final()
except FileNotFoundError as e:
    st.error(f"File data tidak ditemukan: {e}. Pastikan 4 file CSV berada satu folder dengan app.py.")
    st.stop()

TAHUN_MIN, TAHUN_MAX = int(DF["year"].min()), int(DF["year"].max())

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.title("🌍 SDG 3")
st.sidebar.caption("Good Health and Well-being")
MENU = ["🏠 Dashboard", "📊 Data", "🌎 Profil Negara", "📈 EDA", "❤️ Analisis Kesehatan", "🤖 Prediksi", "🧠 Model"]
halaman = st.sidebar.radio("Menu", MENU, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(
    "Sumber data: WHO, World Bank WDI, Our World in Data.\n\n"
    f"Periode {TAHUN_MIN}–{TAHUN_MAX} · {DF['country_code'].nunique()} negara"
)


# ------------------------------------------------------------------
# 1. DASHBOARD
# ------------------------------------------------------------------
def page_dashboard():
    st.title("Analisis dan Prediksi Angka Harapan Hidup Negara")
    st.subheader("SDG 3 — Good Health and Well-being")

    yearly = DF.groupby("year")[TARGET].agg(["count", "mean", "std"])
    awal, akhir = yearly.index.min(), yearly.index.max()
    m_awal, m_akhir = yearly.loc[awal, "mean"], yearly.loc[akhir, "mean"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌎 Jumlah Negara", f"{DF['country_code'].nunique()}")
    c2.metric("📅 Periode Data", f"{TAHUN_MIN}–{TAHUN_MAX}")
    c3.metric("❤️ Rata-rata Life Expectancy", f"{DF[TARGET].mean():.2f} th",
              help="Rata-rata seluruh negara dan seluruh tahun")
    c4.metric("📊 Jumlah Data", f"{len(DF):,}", help="Jumlah baris negara-tahun")

    st.markdown(f"##### Rata-rata Life Expectancy {TAHUN_MIN}–{TAHUN_MAX}")
    tampil(line_band(yearly))
    st.caption("Garis = rata-rata; pita = ± 1 simpangan baku antar negara.")

    d_ref = DF[DF["year"] == akhir].dropna(subset=[TARGET])
    top, bot = d_ref.loc[d_ref[TARGET].idxmax()], d_ref.loc[d_ref[TARGET].idxmin()]
    st.markdown("##### Insight utama")
    st.markdown(
        f"- Rata-rata angka harapan hidup naik dari **{f2(m_awal)} tahun ({awal})** menjadi "
        f"**{f2(m_akhir)} tahun ({akhir})**, yaitu **{f2(m_akhir - m_awal)} tahun**.\n"
        f"- Pada {akhir}, tertinggi: **{top['country_name']} ({f2(top[TARGET])})**, "
        f"terendah: **{bot['country_name']} ({f2(bot[TARGET])})**, selisih "
        f"**{f2(top[TARGET] - bot[TARGET])} tahun**.\n"
        f"- Rata-rata terendah terjadi pada {int(yearly['mean'].idxmin())} ({f2(yearly['mean'].min())})."
    )
    st.caption("Gunakan menu di sebelah kiri untuk melihat data, EDA, dan melakukan prediksi.")


# ------------------------------------------------------------------
# 2. DATA
# ------------------------------------------------------------------
def page_data():
    st.title("📊 Data")
    st.write("Data yang dipakai berasal dari 4 file CSV. Gunakan filter agar mudah ditelusuri.")

    pilihan = {
        "Dataset WHO (who_table)": "who_table",
        "Dataset WDI (wdi_table)": "wdi_table",
        "Dataset OWID (owid_table)": "owid_table",
        "Countries (countries)": "countries",
        "Data gabungan siap analisis (df_final)": "final",
    }
    nama = st.selectbox("Pilih dataset", list(pilihan))
    kunci = pilihan[nama]

    info = RAW["countries"][["country_code", "country_name", "region"]].drop_duplicates("country_code")
    if kunci == "final":
        df = DF.copy()
    elif kunci == "countries":
        df = RAW["countries"].copy()
    else:
        df = RAW[kunci].merge(info, on="country_code", how="left")
        kolom = list(df.columns)
        for k in ["region", "country_name"]:
            kolom.insert(kolom.index("country_code") + 1, kolom.pop(kolom.index(k)))
        df = df[kolom]

    c1, c2, c3 = st.columns(3)
    if "year" in df.columns:
        th = c1.slider("Pilih tahun", int(df["year"].min()), int(df["year"].max()),
                       (int(df["year"].min()), int(df["year"].max())))
        df = df[df["year"].between(*th)]
    if "region" in df.columns:
        pilih_reg = c2.multiselect("Pilih region", sorted(df["region"].dropna().unique()),
                                   placeholder="Semua region")
        if pilih_reg:
            df = df[df["region"].isin(pilih_reg)]
    if "country_name" in df.columns:
        pilih_neg = c3.multiselect("Pilih negara", sorted(df["country_name"].dropna().unique()),
                                   placeholder="Semua negara")
        if pilih_neg:
            df = df[df["country_name"].isin(pilih_neg)]

    st.caption(f"{len(df):,} baris × {df.shape[1]} kolom (tabel menampilkan data sesuai filter)")
    st.dataframe(df, width="stretch", hide_index=True, height=420)

    with st.expander("Statistik deskriptif & missing value"):
        num = df.select_dtypes(include=np.number).drop(columns=["id", "year"], errors="ignore")
        if not num.empty:
            st.dataframe(num.describe().T.round(2), width="stretch")
        miss = df.isna().sum()
        miss = miss[miss > 0]
        if len(miss):
            st.write("Missing value:")
            st.dataframe(pd.DataFrame({"jumlah": miss, "persen": (miss / len(df) * 100).round(2)}))
        else:
            st.write("Tidak ada missing value pada data yang ditampilkan.")

    if kunci == "who_table":
        st.caption("Catatan: who_table berisi 3 kategori `sex`; analisis memakai kategori **Both sexes**.")
    st.download_button("⬇️ Unduh data yang ditampilkan (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{kunci}_filter.csv", mime="text/csv")


# ------------------------------------------------------------------
# PROFIL NEGARA  (cari angka harapan hidup per negara)
# ------------------------------------------------------------------
def page_negara():
    st.title("🌎 Profil Negara")
    st.write("Pilih negara untuk melihat angka harapan hidupnya, dan bandingkan dengan negara lain.")

    daftar = sorted(DF["country_name"].unique())
    c1, c2, c3 = st.columns([2, 3, 2])
    negara = c1.selectbox("Negara", daftar, index=daftar.index("Indonesia") if "Indonesia" in daftar else 0)
    banding = c2.multiselect("Bandingkan dengan (opsional)", [n for n in daftar if n != negara],
                             placeholder="Pilih satu atau lebih negara")
    sub = DF[DF["country_name"] == negara].sort_values("year")
    tahun_ada = sorted(sub["year"].unique())
    tahun = c3.selectbox("Tahun", tahun_ada, index=len(tahun_ada) - 1, key="negara_tahun",
                         help="Dipakai untuk angka ringkasan di bawah; grafik menampilkan semua tahun.")

    baris = sub[sub["year"] == tahun].iloc[0]
    d_th = DF[DF["year"] == tahun]
    rata_dunia = d_th[TARGET].mean()
    peringkat = int(d_th[TARGET].rank(ascending=False, method="min").loc[baris.name])
    awal = sub.iloc[0]

    st.subheader(f"{negara} · {baris['region']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Life expectancy {tahun}", f"{baris[TARGET]:.2f} th",
              delta=f"{baris[TARGET] - rata_dunia:+.2f} th dari rata-rata dunia")
    m2.metric(f"Perubahan sejak {int(awal['year'])}", f"{baris[TARGET] - awal[TARGET]:+.2f} th",
              help=f"{awal[TARGET]:.2f} tahun pada {int(awal['year'])}")
    m3.metric("Peringkat", f"{peringkat} dari {len(d_th)}", help=f"Dari {len(d_th)} negara pada tahun {tahun}")
    m4.metric("Rata-rata dunia", f"{rata_dunia:.2f} th")

    # laki-laki vs perempuan (dari who_table asli)
    who = _kode(RAW["who_table"])
    jk = who[(who["country_code"] == baris["country_code"]) & (who["year"] == tahun)].set_index("sex")[TARGET]
    if {"Female", "Male"} <= set(jk.index):
        j1, j2, j3 = st.columns(3)
        j1.metric("Perempuan", f"{jk['Female']:.2f} th")
        j2.metric("Laki-laki", f"{jk['Male']:.2f} th")
        j3.metric("Selisih (P − L)", f"{jk['Female'] - jk['Male']:+.2f} th")

    # grafik tren
    pilih = [negara] + banding
    tren = DF[DF["country_name"].isin(pilih)][["country_name", "year", TARGET]]
    dunia = DF.groupby("year", as_index=False)[TARGET].mean().assign(country_name="Rata-rata dunia")
    gab = pd.concat([tren, dunia], ignore_index=True)
    garis = alt.Chart(gab).mark_line(point=True).encode(
        x=sumbu_tahun(),
        y=alt.Y(f"{TARGET}:Q", scale=alt.Scale(zero=False), title="Life expectancy (tahun)"),
        color=alt.Color("country_name:N", title=None, legend=alt.Legend(orient="bottom")),
        strokeDash=alt.condition(alt.datum.country_name == "Rata-rata dunia", alt.value([5, 4]), alt.value([0])),
        tooltip=["country_name", "year", alt.Tooltip(f"{TARGET}:Q", format=".2f")],
    ).properties(height=380, title="Tren angka harapan hidup (garis putus-putus = rata-rata dunia)")
    tampil(garis)

    naik = sub.dropna(subset=[TARGET])
    insight(f"Angka harapan hidup **{negara}** pada {tahun} adalah **{f2(baris[TARGET])} tahun**, "
            f"peringkat **{peringkat} dari {len(d_th)}** negara, "
            f"{'di atas' if baris[TARGET] >= rata_dunia else 'di bawah'} rata-rata dunia ({f2(rata_dunia)}). "
            f"Sejak {int(naik['year'].iloc[0])} angkanya berubah {f2(naik[TARGET].iloc[-1] - naik[TARGET].iloc[0])} tahun.")

    # indikator kesehatan vs rata-rata dunia
    st.markdown(f"##### Indikator kesehatan {negara} ({tahun}) dibanding rata-rata dunia")
    tabel = pd.DataFrame({
        "Indikator": [LABEL[f] for f in FEATURES],
        negara: [baris[f] for f in FEATURES],
        "Rata-rata dunia": [d_th[f].mean() for f in FEATURES],
    })
    tabel["Selisih"] = tabel[negara] - tabel["Rata-rata dunia"]
    st.dataframe(tabel.round(2), hide_index=True, width="stretch")
    st.caption("Nilai kosong (NaN) berarti data indikator tersebut tidak tersedia untuk negara dan tahun itu.")

    with st.expander("Tabel angka per tahun"):
        lebar = DF[DF["country_name"].isin(pilih)].pivot(index="year", columns="country_name", values=TARGET)
        st.dataframe(lebar.round(2), width="stretch")


# ------------------------------------------------------------------
# 3. EDA  (hanya satu bagian yang dihitung per klik -> lebih cepat)
# ------------------------------------------------------------------
def page_eda():
    st.title("📈 EDA — Exploratory Data Analysis")
    bagian = st.radio("Pilih pertanyaan",
                      ["Q1 · Perkembangan", "Q2 · Adult Mortality", "Q3 · Indikator Lain"],
                      horizontal=True, label_visibility="collapsed")

    if bagian.startswith("Q1"):
        st.markdown("**Bagaimana perkembangan rata-rata angka harapan hidup negara dari tahun 2000 sampai 2021?**")
        yearly = DF.groupby("year")[TARGET].agg(["count", "mean", "median", "min", "max", "std"]).round(2)
        tampil(line_band(yearly[["mean", "std"]], 360))
        insight(f"Rata-rata berubah dari **{f2(yearly['mean'].iloc[0])}** ({yearly.index[0]}) menjadi "
                f"**{f2(yearly['mean'].iloc[-1])}** ({yearly.index[-1]}); puncaknya {f2(yearly['mean'].max())} "
                f"pada {int(yearly['mean'].idxmax())}.")

        st.markdown("**Perbandingan antar region**")
        tren = DF.groupby(["year", "region"], as_index=False)[TARGET].mean()
        tampil(alt.Chart(tren).mark_line(point=True).encode(
            x=sumbu_tahun(), y=alt.Y(f"{TARGET}:Q", scale=alt.Scale(zero=False), title="Life expectancy (tahun)"),
            color=alt.Color("region:N", legend=alt.Legend(orient="bottom", title=None, columns=2)),
            tooltip=["region", "year", alt.Tooltip(f"{TARGET}:Q", format=".2f")],
        ).properties(height=400))

        st.markdown("**10 negara tertinggi dan terendah**")
        tahun = st.select_slider("Pilih tahun", sorted(DF["year"].unique()), value=TAHUN_MAX)
        d = DF[DF["year"] == tahun].dropna(subset=[TARGET])
        a, b = st.columns(2)
        with a:
            tampil(hbar(d.nlargest(10, TARGET), "country_name", TARGET, HIJAU, judul=f"10 Tertinggi ({tahun})"))
        with b:
            tampil(hbar(d.nsmallest(10, TARGET), "country_name", TARGET, MERAH, judul=f"10 Terendah ({tahun})"))
        with st.expander("Tabel ringkasan per tahun"):
            st.dataframe(yearly, width="stretch")

    elif bagian.startswith("Q2"):
        st.markdown("**Bagaimana hubungan adult mortality dengan life expectancy?**")
        d = DF[["adult_mortality", TARGET]].dropna()
        tampil(scatter_reg(d, "adult_mortality", TARGET, tinggi=420, judul="Adult Mortality vs Life Expectancy"))
        pr = d["adult_mortality"].corr(d[TARGET])
        sr = spearman(d["adult_mortality"], d[TARGET])
        slope = np.polyfit(d["adult_mortality"], d[TARGET], 1)[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Pearson r", f2(pr, 3))
        c2.metric("Spearman ρ", f2(sr, 3))
        c3.metric("Jumlah data (n)", f"{len(d):,}")
        insight(f"Hubungannya **{corr_desc(pr)}**. Setiap kenaikan 1 satuan adult mortality berasosiasi dengan "
                f"perubahan life expectancy sebesar {f2(slope, 3)} tahun. Ini asosiasi, bukan bukti sebab-akibat.")

    else:
        st.markdown("**Bagaimana hubungan adult mortality, neonatal mortality, health expenditure, "
                    "cardiovascular, cancer, dan diabetes death rate dengan life expectancy?**")
        corr = DF[[TARGET] + FEATURES].corr()
        tampil(heatmap(corr))

        ct = corr[TARGET].drop(TARGET).sort_values()
        d_ct = pd.DataFrame({"fitur": [LABEL[c] for c in ct.index], "r": ct.values})
        tampil(alt.Chart(d_ct).mark_bar().encode(
            x=alt.X("r:Q", title="Korelasi Pearson (r)"), y=alt.Y("fitur:N", sort=None, title=None),
            color=alt.condition("datum.r < 0", alt.value(MERAH), alt.value(HIJAU)),
            tooltip=["fitur", alt.Tooltip("r:Q", format=".3f")],
        ).properties(height=260, title="Korelasi terhadap Life Expectancy"))
        insight("; ".join(f"{LABEL[c]}: r = {f2(v, 2)} ({corr_desc(v)})" for c, v in ct.items()) +
                f". Hubungan linear terkuat: **{LABEL[ct.abs().idxmax()]}**.")

        st.markdown("**Scatter plot per indikator**")
        for i in range(0, len(FEATURES), 3):
            cols = st.columns(3)
            for col, f in zip(cols, FEATURES[i:i + 3]):
                with col:
                    r = DF[f].corr(DF[TARGET])
                    tampil(scatter_reg(DF, f, TARGET, tinggi=260, judul=f"{LABEL[f]} (r = {f2(r, 2)})"))


# ------------------------------------------------------------------
# 4. ANALISIS KESEHATAN
# ------------------------------------------------------------------
def page_kesehatan():
    st.title("❤️ Analisis Kesehatan")
    st.write("Pilih satu indikator untuk melihat hubungannya dengan **Life Expectancy**.")

    c1, c2 = st.columns([2, 3])
    f = c1.selectbox("Pilih indikator", FEATURES, format_func=lambda x: LABEL[x])
    th = c2.slider("Rentang tahun", TAHUN_MIN, TAHUN_MAX, (TAHUN_MIN, TAHUN_MAX))
    pilih_reg = st.multiselect("Filter region", sorted(DF["region"].unique()), placeholder="Semua region")

    d = DF[DF["year"].between(*th)]
    if pilih_reg:
        d = d[d["region"].isin(pilih_reg)]
    d = d.dropna(subset=[f, TARGET])
    st.caption(KETERANGAN[f])
    if len(d) < 3:
        st.warning("Data pada filter ini terlalu sedikit.")
        return

    pr, sr = d[f].corr(d[TARGET]), spearman(d[f], d[TARGET])
    m1, m2, m3 = st.columns(3)
    m1.metric("Pearson r", f2(pr, 3))
    m2.metric("Spearman ρ", f2(sr, 3))
    m3.metric("Jumlah data", f"{len(d):,}")

    kiri, kanan = st.columns(2)
    with kiri:
        tampil(scatter_reg(d, f, TARGET, color="region", tinggi=420, judul=f"{LABEL[f]} vs Life Expectancy"))
    with kanan:
        t = d.groupby("year")[[f, TARGET]].mean().reset_index()
        base = alt.Chart(t).encode(x=sumbu_tahun())
        a = base.mark_line(point=True, color=ORANYE).encode(
            y=alt.Y(f"{f}:Q", scale=alt.Scale(zero=False), title=LABEL[f],
                    axis=alt.Axis(titleColor=ORANYE)))
        b = base.mark_line(point=True, color=BIRU).encode(
            y=alt.Y(f"{TARGET}:Q", scale=alt.Scale(zero=False), title="Life expectancy (tahun)",
                    axis=alt.Axis(titleColor=BIRU)))
        tampil((a + b).resolve_scale(y="independent").properties(
            height=420, title="Tren rata-rata: indikator (oranye) vs life expectancy (biru)"))

    insight(f"Hubungan **{LABEL[f]}** dengan life expectancy **{corr_desc(pr)}** (r = {f2(pr, 3)}). "
            f"Korelasi tidak otomatis berarti sebab-akibat; faktor lain seperti pendapatan negara dapat berpengaruh.")

    st.markdown(f"##### Negara dengan {LABEL[f]} terendah & tertinggi (rata-rata periode terpilih)")
    rata = d.groupby("country_name")[[f, TARGET]].mean().sort_values(f)
    a, b = st.columns(2)
    a.write("Terendah")
    a.dataframe(rata.head(10).round(2))
    b.write("Tertinggi")
    b.dataframe(rata.tail(10).iloc[::-1].round(2))


# ------------------------------------------------------------------
# 5. PREDIKSI
# ------------------------------------------------------------------
def prediksi(paket, nilai: dict):
    """Satu baris input -> prediksi life expectancy (nilai kosong diisi median data latih)."""
    fitur = paket["fitur"]
    X = pd.DataFrame([{c: nilai.get(c, np.nan) for c in fitur}], columns=fitur)
    return float(predict_rf(paket, X)[0])


def page_prediksi():
    st.title("🤖 Prediksi Angka Harapan Hidup")
    paket, err = get_paket()
    if paket is None:
        st.error(f"model_rf.npz gagal dibaca: {err}")
        return
    fitur = paket["fitur"]
    med, rng = paket["median_fitur"], paket["rentang_fitur"]
    mae = paket["metrik_uji"]["MAE"]
    st.write("Model **Random Forest Regression** memprediksi `life_expectancy` dari 6 indikator kesehatan.")

    tab_a, tab_b = st.tabs(["✍️ Input manual", "🌎 Dari data negara"])

    with tab_a:
        st.caption("Nilai awal = median data latih. Ubah sesuai skenario yang ingin dicoba.")
        nilai, luar = {}, []
        cols = st.columns(3)
        for i, c in enumerate(fitur):
            lo, hi = rng[c]
            with cols[i % 3]:
                nilai[c] = st.number_input(
                    LABEL[c], min_value=0.0, value=float(round(med[c], 2)), step=1.0,
                    help=f"{KETERANGAN.get(c, '')}. Rentang data latih: {lo:,.2f} – {hi:,.2f}", key=f"in_{c}",
                )
            if not lo <= nilai[c] <= hi:
                luar.append(c)
        if luar:
            st.warning("Di luar rentang data latih (hasil kurang andal): " + ", ".join(LABEL[c] for c in luar))
        if st.button("🔮 Prediksi", type="primary", key="btn_manual"):
            hasil = prediksi(paket, nilai)
            st.success(f"### Prediksi life expectancy: **{hasil:.2f} tahun**")
            st.caption(f"Perkiraan kisaran ±MAE model ({mae:.2f} tahun): {hasil - mae:.2f} – {hasil + mae:.2f} tahun.")
            rata = DF.loc[DF["year"] == TAHUN_MAX, TARGET].mean()
            st.write(f"Pembanding: rata-rata seluruh negara pada {TAHUN_MAX} = **{rata:.2f} tahun** "
                     f"(selisih {hasil - rata:+.2f}).")

    with tab_b:
        st.caption("Ambil indikator dari data asli, lalu bandingkan prediksi dengan nilai sebenarnya.")
        c1, c2 = st.columns(2)
        negara = c1.selectbox("Negara", sorted(DF["country_name"].unique()))
        sub = DF[DF["country_name"] == negara]
        tahun = c2.selectbox("Tahun", sorted(sub["year"].unique(), reverse=True))
        baris = sub[sub["year"] == tahun].iloc[0]

        st.dataframe(pd.DataFrame({
            "Indikator": [LABEL[c] for c in fitur],
            "Nilai": [baris[c] for c in fitur],
            "Keterangan": ["kosong → diisi median" if pd.isna(baris[c]) else "dari data" for c in fitur],
        }), hide_index=True, width="stretch")
        hasil = prediksi(paket, {c: baris[c] for c in fitur})
        aktual = baris[TARGET]
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediksi", f"{hasil:.2f} th")
        m2.metric("Aktual", f"{aktual:.2f} th")
        m3.metric("Selisih (prediksi − aktual)", f"{hasil - aktual:+.2f} th")
        if tahun < TAHUN_UJI_MULAI:
            st.caption("Tahun ini termasuk **data latih**, jadi akurasinya cenderung lebih tinggi dari data uji.")

    with st.expander("ℹ️ Catatan penggunaan"):
        st.markdown(
            "- Model hanya memakai 6 indikator di atas (tanpa kode negara dan tahun).\n"
            "- Nilai kosong diisi median data latih lewat `SimpleImputer`.\n"
            "- Prediksi adalah **asosiasi statistik**, bukan bukti sebab-akibat."
        )


# ------------------------------------------------------------------
# 6. MODEL
# ------------------------------------------------------------------
def page_model():
    st.title("🧠 Model — Random Forest Regressor")
    paket, err = get_paket()
    if paket is None:
        st.error(f"model_rf.npz gagal dibaca: {err}")
        return
    fitur = paket["fitur"]
    m = paket["metrik_uji"]

    st.markdown(
        "**Target:** `life_expectancy` · **Fitur:** " + ", ".join(f"`{c}`" for c in fitur) + "  \n"
        f"**Algoritma:** Random Forest ({paket['n_pohon']} pohon) · "
        f"**Imputasi:** median data latih · **Split temporal:** latih ≤ {TAHUN_UJI_MULAI - 1}, uji ≥ {TAHUN_UJI_MULAI}"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (data uji)", f"{m['MAE']:.2f} th", help="Rata-rata selisih mutlak prediksi vs aktual")
    c2.metric("RMSE (data uji)", f"{m['RMSE']:.2f} th", help="Lebih menghukum kesalahan besar")
    c3.metric("R² (data uji)", f"{m['R2']:.3f}", help="1 = sempurna, 0 = setara menebak rata-rata")

    uji = hasil_uji()
    if uji.empty:
        st.info("Tidak ada data tahun uji pada dataset.")
    else:
        st.subheader("Actual vs Predicted (data uji)")
        lo = float(min(uji[TARGET].min(), uji["prediksi"].min()) - 1)
        hi = float(max(uji[TARGET].max(), uji["prediksi"].max()) + 1)
        skala = alt.Scale(domain=[lo, hi])
        a, b = st.columns(2)
        with a:
            pts = alt.Chart(uji).mark_circle(size=40, opacity=0.6).encode(
                x=alt.X(f"{TARGET}:Q", scale=skala, title="Actual (tahun)"),
                y=alt.Y("prediksi:Q", scale=skala, title="Predicted (tahun)"),
                color=alt.Color("year:N", title="Tahun"),
                tooltip=["country_name", "year", alt.Tooltip(f"{TARGET}:Q", format=".2f"),
                         alt.Tooltip("prediksi:Q", format=".2f")])
            diag = alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})).mark_line(
                color=MERAH, strokeDash=[6, 4]).encode(x="x:Q", y="y:Q")
            tampil((pts + diag).properties(height=380, title="Garis merah = prediksi sempurna (y = x)"))
        with b:
            tampil(alt.Chart(uji).mark_bar(color="#9467bd").encode(
                x=alt.X("error:Q", bin=alt.Bin(maxbins=30), title="Error (Predicted − Actual, tahun)"),
                y=alt.Y("count()", title="Frekuensi"),
            ).properties(height=380, title="Distribusi Error"))
        insight(f"Pada {len(uji)} baris data uji, rata-rata error (bias) {f2(uji['error'].mean())} tahun; "
                f"90% prediksi selisihnya ≤ {f2(uji['abs_error'].quantile(0.9))} tahun.")

        with st.expander("Tabel hasil prediksi (data uji)"):
            tabel = uji[["country_name", "year", TARGET, "prediksi", "error"]].rename(
                columns={"country_name": "negara", TARGET: "aktual"})
            st.dataframe(tabel.round(2), hide_index=True, width="stretch")

    st.subheader("Feature Importance")
    fi = pd.DataFrame({"fitur": [LABEL[c] for c in fitur], "importance": paket["importance"]})
    tampil(alt.Chart(fi).mark_bar(color=ORANYE).encode(
        x=alt.X("importance:Q", title="Importance"), y=alt.Y("fitur:N", sort="-x", title=None),
        tooltip=["fitur", alt.Tooltip("importance:Q", format=".3f")]).properties(height=280))
    top = fi.loc[fi["importance"].idxmax()]
    insight(f"Fitur paling berpengaruh: **{top['fitur']}** ({top['importance'] * 100:.1f}%). "
            "Importance bukan bukti sebab-akibat, dan fitur yang saling berkorelasi bisa saling berbagi nilai.")

    with st.expander("Keterbatasan"):
        st.markdown(
            "- Korelasi dan feature importance bukan bukti sebab-akibat.\n"
            "- Data uji hanya 2020–2021 (dipengaruhi pandemi).\n"
            "- Adult mortality berhubungan sangat erat dengan life expectancy, sehingga R² tinggi tidak otomatis "
            "berarti model cocok untuk menentukan intervensi.\n"
            "- Tidak ada variabel sosial-ekonomi lain karena hanya 4 CSV yang dipakai."
        )


# ------------------------------------------------------------------
# ROUTER
# ------------------------------------------------------------------
{
    "🏠 Dashboard": page_dashboard,
    "📊 Data": page_data,
    "🌎 Profil Negara": page_negara,
    "📈 EDA": page_eda,
    "❤️ Analisis Kesehatan": page_kesehatan,
    "🤖 Prediksi": page_prediksi,
    "🧠 Model": page_model,
}[halaman]()
