"""
Analisis dan Prediksi Angka Harapan Hidup Negara — SDG 3 (Good Health and Well-being)

Aplikasi Streamlit yang dibuat dari notebook Colab:
  Load CSV -> Cleaning -> Merge -> EDA -> Visualisasi -> Machine Learning -> Evaluasi
Model (Random Forest) TIDAK dilatih ulang di sini, hanya dibaca dari model.pkl.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Angka Harapan Hidup | SDG 3",
    page_icon="❤️",
    layout="wide",
)

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

    # WHO: pakai kategori "Both sexes"
    who = _kode(raw["who_table"])
    who = who[who["sex"].astype(str).str.lower().str.contains("both")].copy()
    who.loc[(who[TARGET] <= 0) | (who[TARGET] > 120), TARGET] = np.nan
    who = who.groupby(["country_code", "year"], as_index=False)[TARGET].mean()

    def siapkan(nama, kolom):
        d = _kode(raw[nama])
        for c in kolom:
            d.loc[d[c] < 0, c] = np.nan  # nilai negatif tidak masuk akal -> missing
        return d.groupby(["country_code", "year"], as_index=False)[kolom].mean()

    wdi = siapkan("wdi_table", ["adult_mortality", "neonatal_mortality", "health_expenditure"])
    owid = siapkan(
        "owid_table", ["cardiovascular_death_rate", "cancer_death_rate", "diabetes_death_rate"]
    )

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


@st.cache_resource
def load_model():
    """Baca paket model dari model.pkl (dict: model, imputer, fitur, metrik_uji, dst)."""
    return joblib.load(BASE / "model.pkl")


def f2(x, nd=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:,.{nd}f}"


def corr_desc(r):
    if pd.isna(r):
        return "tidak dapat dihitung"
    a = abs(r)
    kuat = (
        "sangat kuat" if a >= 0.8 else "kuat" if a >= 0.6 else "sedang" if a >= 0.4
        else "lemah" if a >= 0.2 else "sangat lemah"
    )
    return f"{kuat} dan {'positif' if r > 0 else 'negatif'}"


def tampil_fig(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def insight(teks):
    st.info("**Insight:** " + teks)


def scatter_reg(ax, d, x, y, color=None, alpha=0.3):
    """Scatter + garis regresi linear sederhana."""
    d = d[[x, y]].dropna()
    ax.scatter(d[x], d[y], alpha=alpha, s=14, color=color)
    if len(d) >= 3 and d[x].nunique() > 1:
        m, b = np.polyfit(d[x], d[y], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 100)
        ax.plot(xs, m * xs + b, color="red", linewidth=1.8)
    return d


# ------------------------------------------------------------------
# MUAT DATA + MODEL
# ------------------------------------------------------------------
try:
    RAW = load_raw()
    DF = build_final()
except FileNotFoundError as e:
    st.error(f"File data tidak ditemukan: {e}. Pastikan 4 file CSV berada satu folder dengan app.py.")
    st.stop()

try:
    PAKET = load_model()
    MODEL_OK = True
except Exception as e:  # noqa: BLE001
    PAKET, MODEL_OK = None, False
    MODEL_ERR = e

TAHUN_MIN, TAHUN_MAX = int(DF["year"].min()), int(DF["year"].max())

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.title("🌍 SDG 3")
st.sidebar.caption("Good Health and Well-being")
MENU = [
    "🏠 Dashboard",
    "📊 Data",
    "📈 EDA",
    "❤️ Analisis Kesehatan",
    "🤖 Prediksi",
    "🧠 Model",
]
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
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(yearly.index, yearly["mean"], marker="o", color="tab:blue", label="Rata-rata")
    std = yearly["std"].fillna(0)
    ax.fill_between(yearly.index, yearly["mean"] - std, yearly["mean"] + std,
                    alpha=0.15, color="tab:blue", label="± 1 simpangan baku")
    ax.set(xlabel="Tahun", ylabel="Life expectancy (tahun)")
    ax.grid(alpha=0.3)
    ax.legend()
    tampil_fig(fig)

    d_ref = DF[DF["year"] == akhir].dropna(subset=[TARGET])
    top, bot = d_ref.loc[d_ref[TARGET].idxmax()], d_ref.loc[d_ref[TARGET].idxmin()]
    st.markdown("##### Insight utama")
    st.markdown(
        f"- Rata-rata angka harapan hidup naik dari **{f2(m_awal)} tahun ({awal})** menjadi "
        f"**{f2(m_akhir)} tahun ({akhir})**, yaitu **{f2(m_akhir - m_awal)} tahun**.\n"
        f"- Pada {akhir}, tertinggi: **{top['country_name']} ({f2(top[TARGET])})**, "
        f"terendah: **{bot['country_name']} ({f2(bot[TARGET])})**, selisih "
        f"**{f2(top[TARGET] - bot[TARGET])} tahun**.\n"
        f"- Rata-rata {int(yearly['mean'].idxmin())} adalah yang terendah "
        f"({f2(yearly['mean'].min())}); perhatikan dampak pandemi pada tahun-tahun akhir."
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
        # letakkan nama negara & region setelah country_code
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
        regs = sorted(df["region"].dropna().unique())
        pilih_reg = c2.multiselect("Pilih region", regs, placeholder="Semua region")
        if pilih_reg:
            df = df[df["region"].isin(pilih_reg)]
    if "country_name" in df.columns:
        negs = sorted(df["country_name"].dropna().unique())
        pilih_neg = c3.multiselect("Pilih negara", negs, placeholder="Semua negara")
        if pilih_neg:
            df = df[df["country_name"].isin(pilih_neg)]

    st.caption(f"{len(df):,} baris × {df.shape[1]} kolom")
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
# 3. EDA
# ------------------------------------------------------------------
def page_eda():
    st.title("📈 EDA — Exploratory Data Analysis")
    tab1, tab2, tab3 = st.tabs(["Q1 · Perkembangan", "Q2 · Adult Mortality", "Q3 · Indikator Lain"])

    # ---- Q1
    with tab1:
        st.markdown("**Bagaimana perkembangan rata-rata angka harapan hidup negara dari tahun 2000 sampai 2021?**")
        yearly = DF.groupby("year")[TARGET].agg(["count", "mean", "median", "min", "max", "std"]).round(2)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(yearly.index, yearly["mean"], marker="o", color="tab:blue")
        ax.set(title="Rata-rata Angka Harapan Hidup per Tahun", xlabel="Tahun", ylabel="Life expectancy (tahun)")
        ax.grid(alpha=0.3)
        tampil_fig(fig)
        insight(f"Rata-rata berubah dari **{f2(yearly['mean'].iloc[0])}** ({yearly.index[0]}) menjadi "
                f"**{f2(yearly['mean'].iloc[-1])}** ({yearly.index[-1]}); puncaknya {f2(yearly['mean'].max())} "
                f"pada {int(yearly['mean'].idxmax())}.")

        st.markdown("**Perbandingan antar region**")
        tren = DF.groupby(["year", "region"])[TARGET].mean().unstack()
        fig, ax = plt.subplots(figsize=(10, 5))
        for r in tren.columns:
            ax.plot(tren.index, tren[r], marker="o", markersize=3, label=r)
        ax.set(xlabel="Tahun", ylabel="Life expectancy (tahun)", title="Tren Rata-rata per Region")
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.grid(alpha=0.3)
        tampil_fig(fig)

        st.markdown("**10 negara tertinggi dan terendah**")
        tahun = st.select_slider("Pilih tahun", sorted(DF["year"].unique()), value=TAHUN_MAX, key="eda_q1_year")
        d = DF[DF["year"] == tahun].dropna(subset=[TARGET])
        top10 = d.nlargest(10, TARGET).sort_values(TARGET)
        bot10 = d.nsmallest(10, TARGET).sort_values(TARGET, ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].barh(top10["country_name"], top10[TARGET], color="tab:green")
        axes[0].set(title=f"10 Tertinggi ({tahun})", xlabel="Life expectancy (tahun)")
        axes[1].barh(bot10["country_name"], bot10[TARGET], color="tab:red")
        axes[1].set(title=f"10 Terendah ({tahun})", xlabel="Life expectancy (tahun)")
        fig.tight_layout()
        tampil_fig(fig)
        with st.expander("Tabel ringkasan per tahun"):
            st.dataframe(yearly, width="stretch")

    # ---- Q2
    with tab2:
        st.markdown("**Bagaimana hubungan adult mortality dengan life expectancy?**")
        d = DF[["adult_mortality", TARGET, "region"]].dropna(subset=["adult_mortality", TARGET])
        fig, ax = plt.subplots(figsize=(9, 5.5))
        scatter_reg(ax, d, "adult_mortality", TARGET)
        ax.set(title="Adult Mortality vs Life Expectancy", xlabel="Adult mortality", ylabel="Life expectancy (tahun)")
        ax.grid(alpha=0.3)
        tampil_fig(fig)
        pr = d["adult_mortality"].corr(d[TARGET])
        sr = d["adult_mortality"].corr(d[TARGET], method="spearman")
        slope = np.polyfit(d["adult_mortality"], d[TARGET], 1)[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Pearson r", f2(pr, 3))
        c2.metric("Spearman ρ", f2(sr, 3))
        c3.metric("Jumlah data (n)", f"{len(d):,}")
        insight(f"Hubungannya **{corr_desc(pr)}**. Setiap kenaikan 1 satuan adult mortality berasosiasi dengan "
                f"perubahan life expectancy sebesar {f2(slope, 3)} tahun. Ini asosiasi, bukan bukti sebab-akibat.")

    # ---- Q3
    with tab3:
        st.markdown("**Bagaimana hubungan adult mortality, neonatal mortality, health expenditure, "
                    "cardiovascular, cancer, dan diabetes death rate dengan life expectancy?**")
        corr = DF[[TARGET] + FEATURES].corr()
        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        nama = [LABEL[c].replace(" (tahun)", "") for c in corr.columns]
        ax.set_xticks(range(len(nama)), nama, rotation=45, ha="right")
        ax.set_yticks(range(len(nama)), nama)
        for i in range(len(nama)):
            for j in range(len(nama)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        tampil_fig(fig)

        ct = corr[TARGET].drop(TARGET).sort_values()
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.barh([LABEL[c] for c in ct.index], ct.values, color=["tab:red" if v < 0 else "tab:green" for v in ct])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(title="Korelasi Pearson terhadap Life Expectancy", xlabel="r")
        fig.tight_layout()
        tampil_fig(fig)
        terkuat = ct.abs().idxmax()
        insight("; ".join(f"{LABEL[c]}: r = {f2(v, 2)} ({corr_desc(v)})" for c, v in ct.items()) +
                f". Hubungan linear terkuat: **{LABEL[terkuat]}**.")

        st.markdown("**Scatter plot per indikator**")
        lain = [f for f in FEATURES if f != "adult_mortality"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, f in zip(axes.flatten(), lain + ["adult_mortality"]):
            d = scatter_reg(ax, DF, f, TARGET, alpha=0.25)
            ax.set(title=f"{LABEL[f]} (r = {f2(d[f].corr(d[TARGET]), 2)})", xlabel=f, ylabel="Life expectancy")
        fig.tight_layout()
        tampil_fig(fig)


# ------------------------------------------------------------------
# 4. ANALISIS KESEHATAN
# ------------------------------------------------------------------
def page_kesehatan():
    st.title("❤️ Analisis Kesehatan")
    st.write("Pilih satu indikator untuk melihat hubungannya dengan **Life Expectancy**.")

    c1, c2 = st.columns([2, 3])
    f = c1.selectbox("Pilih indikator", FEATURES, format_func=lambda x: LABEL[x])
    th = c2.slider("Rentang tahun", TAHUN_MIN, TAHUN_MAX, (TAHUN_MIN, TAHUN_MAX))
    regs = sorted(DF["region"].unique())
    pilih_reg = st.multiselect("Filter region", regs, placeholder="Semua region")

    d = DF[DF["year"].between(*th)]
    if pilih_reg:
        d = d[d["region"].isin(pilih_reg)]
    d = d.dropna(subset=[f, TARGET])
    st.caption(KETERANGAN[f])
    if len(d) < 3:
        st.warning("Data pada filter ini terlalu sedikit.")
        return

    pr, sr = d[f].corr(d[TARGET]), d[f].corr(d[TARGET], method="spearman")
    m1, m2, m3 = st.columns(3)
    m1.metric("Pearson r", f2(pr, 3))
    m2.metric("Spearman ρ", f2(sr, 3))
    m3.metric("Jumlah data", f"{len(d):,}")

    kiri, kanan = st.columns(2)
    with kiri:
        fig, ax = plt.subplots(figsize=(7, 5))
        for r, g in d.groupby("region"):
            ax.scatter(g[f], g[TARGET], s=14, alpha=0.5, label=r)
        m, b = np.polyfit(d[f], d[TARGET], 1)
        xs = np.linspace(d[f].min(), d[f].max(), 100)
        ax.plot(xs, m * xs + b, color="red", linewidth=2, label="Regresi linear")
        ax.set(title=f"{LABEL[f]} vs Life Expectancy", xlabel=LABEL[f], ylabel="Life expectancy (tahun)")
        ax.grid(alpha=0.3)
        if len(d["region"].unique()) <= 8:
            ax.legend(fontsize=6)
        tampil_fig(fig)
    with kanan:
        tren = d.groupby("year")[[f, TARGET]].mean()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(tren.index, tren[f], color="tab:orange", marker="o", markersize=3)
        ax.set_ylabel(LABEL[f], color="tab:orange")
        ax.set_xlabel("Tahun")
        ax2 = ax.twinx()
        ax2.plot(tren.index, tren[TARGET], color="tab:blue", marker="o", markersize=3)
        ax2.set_ylabel("Life expectancy (tahun)", color="tab:blue")
        ax.set_title("Tren Rata-rata: Indikator vs Life Expectancy")
        ax.grid(alpha=0.3)
        tampil_fig(fig)

    insight(f"Hubungan **{LABEL[f]}** dengan life expectancy **{corr_desc(pr)}** (r = {f2(pr, 3)}). "
            f"Korelasi tidak otomatis berarti sebab-akibat; faktor lain seperti pendapatan negara dapat berpengaruh.")

    st.markdown(f"##### Negara dengan {LABEL[f]} tertinggi & terendah (rata-rata periode terpilih)")
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
    """Imputer (median data latih) -> Random Forest, urutan kolom sesuai paket['fitur']."""
    fitur = paket["fitur"]
    X = pd.DataFrame([{c: nilai.get(c, np.nan) for c in fitur}], columns=fitur)
    X_imp = pd.DataFrame(paket["imputer"].transform(X), columns=fitur)
    return float(paket["model"].predict(X_imp)[0])


def page_prediksi():
    st.title("🤖 Prediksi Angka Harapan Hidup")
    if not MODEL_OK:
        st.error(f"model.pkl gagal dibaca: {MODEL_ERR}")
        return
    paket = PAKET
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

        tampil = pd.DataFrame({
            "Indikator": [LABEL[c] for c in fitur],
            "Nilai": [baris[c] for c in fitur],
            "Keterangan": ["kosong → diisi median" if pd.isna(baris[c]) else "dari data" for c in fitur],
        })
        st.dataframe(tampil, hide_index=True, width="stretch")
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
    if not MODEL_OK:
        st.error(f"model.pkl gagal dibaca: {MODEL_ERR}")
        return
    paket = PAKET
    fitur, model, imputer = paket["fitur"], paket["model"], paket["imputer"]
    m = paket["metrik_uji"]

    st.markdown(
        "**Target:** `life_expectancy` · **Fitur:** " + ", ".join(f"`{c}`" for c in fitur) + "  \n"
        f"**Algoritma:** Random Forest ({model.n_estimators} pohon) · "
        f"**Imputasi:** median data latih · **Split temporal:** latih ≤ {TAHUN_UJI_MULAI - 1}, uji ≥ {TAHUN_UJI_MULAI}"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (data uji)", f"{m['MAE']:.2f} th", help="Rata-rata selisih mutlak prediksi vs aktual")
    c2.metric("RMSE (data uji)", f"{m['RMSE']:.2f} th", help="Lebih menghukum kesalahan besar")
    c3.metric("R² (data uji)", f"{m['R2']:.3f}", help="1 = sempurna, 0 = setara menebak rata-rata")

    # prediksi ulang pada data uji (tanpa training ulang)
    uji = DF[DF["year"] >= TAHUN_UJI_MULAI].copy()
    if uji.empty:
        st.info("Tidak ada data tahun uji pada dataset.")
    else:
        Xu = pd.DataFrame(imputer.transform(uji[fitur]), columns=fitur, index=uji.index)
        uji["prediksi"] = model.predict(Xu)
        uji["error"] = uji["prediksi"] - uji[TARGET]
        uji["abs_error"] = uji["error"].abs()

        st.subheader("Actual vs Predicted (data uji)")
        lo = min(uji[TARGET].min(), uji["prediksi"].min()) - 1
        hi = max(uji[TARGET].max(), uji["prediksi"].max()) + 1
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(6, 5.5))
            for t, g in uji.groupby("year"):
                ax.scatter(g[TARGET], g["prediksi"], s=22, alpha=0.6, label=str(t))
            ax.plot([lo, hi], [lo, hi], "r--", label="y = x")
            ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Actual (tahun)", ylabel="Predicted (tahun)")
            ax.legend(title="Tahun")
            ax.grid(alpha=0.3)
            tampil_fig(fig)
        with b:
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.hist(uji["error"], bins=30, color="tab:purple", alpha=0.8)
            ax.axvline(0, color="red", linestyle="--")
            ax.set(title="Distribusi Error (Predicted − Actual)", xlabel="Error (tahun)", ylabel="Frekuensi")
            tampil_fig(fig)
        insight(f"Pada {len(uji)} baris data uji, rata-rata error (bias) {f2(uji['error'].mean())} tahun; "
                f"90% prediksi selisihnya ≤ {f2(uji['abs_error'].quantile(0.9))} tahun.")

        with st.expander("Tabel hasil prediksi (data uji)"):
            tabel = uji[["country_name", "year", TARGET, "prediksi", "error"]].rename(
                columns={"country_name": "negara", TARGET: "aktual"})
            st.dataframe(tabel.round(2), hide_index=True, width="stretch")

    st.subheader("Feature Importance")
    fi = pd.DataFrame({"fitur": fitur, "importance": model.feature_importances_}).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh([LABEL[c] for c in fi["fitur"]], fi["importance"], color="tab:orange")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    tampil_fig(fig)
    top = fi.iloc[-1]
    insight(f"Fitur paling berpengaruh: **{LABEL[top['fitur']]}** ({top['importance'] * 100:.1f}%). "
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
    "📈 EDA": page_eda,
    "❤️ Analisis Kesehatan": page_kesehatan,
    "🤖 Prediksi": page_prediksi,
    "🧠 Model": page_model,
}[halaman]()
