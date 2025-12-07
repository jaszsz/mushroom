import streamlit as st
import pandas as pd
import plotly.express as px
import kagglehub
from kagglehub import KaggleDatasetAdapter

st.set_page_config(page_title="Mushroom Dashboard", layout="wide")

# =============================
# Load Dataset
# =============================
@st.cache_data
def load_data():
    file_path = "mushrooms.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "uciml/mushroom-classification",
        file_path
    )
    return df

df = load_data()

# =============================
# Dictionary Penjelasan Fitur (Bahasa Indonesia + Emoji)
# =============================

feature_description = {
    "cap-shape": "🍄 Bentuk Tudung: bell=lonceng (b), conical=kerucut (c), convex=cembung (x), flat=datar (f), knobbed=bertombol (k), sunken=cekung (s)",
    "cap-surface": "🧱 Permukaan Tudung: fibrous=berserat (f), grooves=beralur (g), scaly=bersisik (y), smooth=halus (s)",
    "cap-color": "🎨 Warna Tudung: brown=coklat (n), buff=kuning pucat (b), cinnamon=kayu manis (c), gray=abu-abu (g), green=hijau (r), pink=merah muda (p), purple=ungu (u), red=merah (e), white=putih (w), yellow=kuning (y)",
    "bruises": "💥 Perubahan Warna Saat Memar: t=ya, f=tidak",
    "odor": "👃 Bau: almond=kacang almond (a), anise=adas manis (l), creosote=kreosot (c), fishy=amis (y), foul=busuk (f), musty=apek (m), none=tidak ada bau (n), pungent=menyengat (p), spicy=pedas (s)",
    "gill-attachment": "🦴 Keterikatan Bilah: attached=menempel (a), descending=menurun (d), free=bebas (f), notched=berlekuk (n)",
    "gill-spacing": "📏 Jarak Bilah: close=rapat (c), crowded=sangat rapat (w), distant=jarang (d)",
    "gill-size": "📐 Ukuran Bilah: broad=lebar (b), narrow=sempit (n)",
    "gill-color": "🎨 Warna Bilah: black=hitam (k), brown=coklat (n), buff=kuning pucat (b), chocolate=chokelat tua (h), gray=abu-abu (g), green=hijau (r), orange=oranye (o), pink=merah muda (p), purple=ungu (u), red=merah (e), white=putih (w), yellow=kuning (y)",
    "stalk-shape": "🏗 Bentuk Batang: enlarging=melebar ke bawah (e), tapering=menyempit ke bawah (t)",
    "stalk-root": "🌱 Akar Batang: bulbous=membulat (b), club=seperti gada (c), cup=cawan (u), equal=seragam (e), rhizomorphs=serabut panjang (z), rooted=berakar (r), missing=tidak diketahui (?)",
    "stalk-surface-above-ring": "🧵 Permukaan Batang Atas Cincin: fibrous=berserat (f), scaly=bersisik (y), silky=sutra (k), smooth=halus (s)",
    "stalk-surface-below-ring": "🧶 Permukaan Batang Bawah Cincin: fibrous=berserat (f), scaly=bersisik (y), silky=sutra (k), smooth=halus (s)",
    "stalk-color-above-ring": "🎨 Warna Batang Atas Cincin: brown=coklat (n), buff=kuning pucat (b), cinnamon=kayu manis (c), gray=abu-abu (g), orange=oranye (o), pink=merah muda (p), red=merah (e), white=putih (w), yellow=kuning (y)",
    "stalk-color-below-ring": "🎨 Warna Batang Bawah Cincin: brown=coklat (n), buff=kuning pucat (b), cinnamon=kayu manis (c), gray=abu-abu (g), orange=oranye (o), pink=merah muda (p), red=merah (e), white=putih (w), yellow=kuning (y)",
    "veil-type": "🧥 Tipe Selubung: partial=sebagian (p), universal=menyeluruh (u)",
    "veil-color": "🎨 Warna Selubung: brown=coklat (n), orange=oranye (o), white=putih (w), yellow=kuning (y)",
    "ring-number": "🔵 Jumlah Cincin: none=tidak ada (n), one=satu (o), two=dua (t)",
    "ring-type": "⭕ Tipe Cincin: cobwebby=jaring laba-laba (c), evanescent=memudar (e), flaring=melebar (f), large=besar (l), none=tanpa cincin (n), pendant=menjuntai (p), sheathing=menyelubungi (s), zone=berzona (z)",
    "spore-print-color": "🧬 Warna Cetakan Spora: black=hitam (k), brown=coklat (n), buff=kuning pucat (b), chocolate=chokelat tua (h), green=hijau (r), orange=oranye (o), purple=ungu (u), white=putih (w), yellow=kuning (y)",
    "population": "👥 Populasi: abundant=melimpah (a), clustered=bergerombol (c), numerous=banyak (n), scattered=menyebar (s), several=beberapa (v), solitary=tunggal (y)",
    "habitat": "🌲 Habitat: grasses=rumput (g), leaves=daun (l), meadows=padang rumput (m), paths=jalan setapak (p), urban=perkotaan (u), waste=limbah/tanah rusak (w), woods=hutan (d)"
}

feature_title = {
    "cap-shape": "🍄 Bentuk Tudung: menggambarkan bentuk fisik bagian atas jamur yang dapat membantu membedakan spesiesnya.",
    "cap-surface": "🧱 Permukaan Tudung: menunjukkan tekstur luar tudung jamur, apakah halus, bersisik, atau berlendir.",
    "cap-color": "🎨 Warna Tudung: mendeskripsikan variasi warna pada bagian tudung yang sering menjadi ciri utama identifikasi.",
    "bruises": "💥 Perubahan Warna Saat Memar: menunjukkan apakah permukaan jamur berubah warna ketika ditekan atau rusak.",
    "odor": "👃 Bau Jamur: menggambarkan aroma khas yang dihasilkan jamur dan kerap menjadi indikator penting dalam klasifikasi.",
    "gill-attachment": "🦴 Keterikatan Bilah: menjelaskan bagaimana bilah jamur melekat atau terhubung pada batang.",
    "gill-spacing": "📏 Jarak Bilah: menunjukkan seberapa rapat atau renggang susunan bilah di bagian bawah tudung.",
    "gill-size": "📐 Ukuran Bilah: menggambarkan besar-kecilnya bilah yang dapat mempengaruhi karakteristik morfologi jamur.",
    "gill-color": "🎨 Warna Bilah: mendeskripsikan warna bilah jamur yang sering berubah seiring usia atau spora.",
    "stalk-shape": "🌱 Bentuk Batang: menunjukkan bentuk geometris batang jamur, apakah meruncing, membulat, atau mengembang.",
    "stalk-root": "🌰 Akar Batang: menjelaskan kondisi atau struktur dasar batang yang terhubung dengan tanah atau substrat.",
    "stalk-surface-above-ring": "🧵 Permukaan Batang (atas cincin): mendeskripsikan tekstur batang di area atas cincin jamur.",
    "stalk-surface-below-ring": "🧵 Permukaan Batang (bawah cincin): menunjukkan karakter permukaan batang pada bagian di bawah cincin.",
    "stalk-color-above-ring": "🎨 Warna Batang (atas cincin): menggambarkan warna batang pada area yang berada di atas cincin jamur.",
    "stalk-color-below-ring": "🎨 Warna Batang (bawah cincin): menjelaskan warna batang di bagian bawah cincin sebagai pembeda spesies.",
    "veil-type": "🪶 Jenis Selubung: menunjukkan tipe selubung pelindung yang mengitari jamur saat muda.",
    "veil-color": "🎨 Warna Selubung: mendeskripsikan warna selubung jamur yang biasanya tampak sebelum jamur dewasa.",
    "ring-number": "🔘 Jumlah Cincin: menjelaskan berapa banyak cincin yang muncul pada batang jamur.",
    "ring-type": "🔔 Jenis Cincin: menggambarkan bentuk atau karakteristik cincin yang mengelilingi batang jamur.",
    "spore-print-color": "🧬 Warna Spora: menunjukkan warna hasil cetakan spora yang sangat penting untuk identifikasi spesies.",
    "population": "👥 Populasi: menjelaskan jumlah atau kepadatan kemunculan jamur di lingkungan tertentu.",
    "habitat": "🌳 Habitat: menggambarkan jenis lingkungan tempat jamur tumbuh, seperti hutan, padang rumput, atau area lembap."
}

import pandas as pd

import re

def convert_description_to_table(desc: str):
    desc_clean = desc.split(":")[1] if ":" in desc else desc
    items = desc_clean.split(",")

    rows = []
    for item in items:
        if "=" in item:
            meaning, symbol = item.split("=")
            meaning = meaning.strip().capitalize()
            
            # Ambil kode di dalam tanda kurung
            match = re.search(r"\((.*?)\)", symbol)
            kode = match.group(1) if match else ""
            
            # Hapus kode dan kurung dari symbol
            kata = re.sub(r"\(.*?\)", "", symbol).strip().capitalize()

            rows.append({
                "Kode": kode,
                "Kata": kata,
                "Arti": meaning
            })
    return pd.DataFrame(rows)


# =================================
# Sidebar Menu
# =================================
st.sidebar.title("📌 Menu Sidebar")

# =================================
# Sidebar Menu
# =================================

page = st.sidebar.radio(
    "Pilih Halaman:",
    ["Informasi Dataset", "Berita & Informasi Terkini", "Hasil Analisis"]
)

# =================================
# Halaman 1: Informasi Dataset
# =================================
if page == "Informasi Dataset":

    st.header("📘 Informasi Dataset")
    st.markdown("""
    Dataset **UCI Mushroom Classification** berisi 8124 sampel jamur dengan 22 fitur kategorikal untuk membedakan jamur:

    - **edible (e)** = aman dimakan  
    - **poisonous (p)** = beracun / berbahaya  
                  
    """)
    col1, col2 = st.columns(2)

    with col1:
        with open("mushrooms.csv", "rb") as f:
            st.download_button(
                label="📥 Download CSV",
                data=f,
                file_name="mushrooms.csv",
                mime="text/csv"
            )

    with col2:
        st.markdown("""
            <a href="https://www.kaggle.com/datasets/uciml/mushroom-classification" target="_blank">
                <div style="
                    display:inline-block;
                    padding:12px 22px;
                    background:#7D3C98;
                    color:white;
                    border-radius:8px;
                    font-weight:600;
                    text-decoration:none;
                    text-align:center;
                    width:100%;
                    box-shadow:0 2px 6px rgba(0,0,0,0.2);
                ">
                    🌐 Lihat di Kaggle
                </div>
            </a>
        """, unsafe_allow_html=True)

    # =============================
    # Title
    # =============================
    st.title("🍄 Mushroom Classification Dashboard")
    # =============================
    # Data table
    # =============================
    st.subheader("📄 Tabel Data")
    st.dataframe(df)
    
    # ---- SELECTBOX DILETAKKAN DI SIDEBAR HALAMAN INI ----
    column_selected = st.selectbox(
        "Pilih fitur untuk dianalisis:",
        df.columns
    )
    # =============================
    # Basic Info
    # =============================
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", len(df))
    col2.metric("Jumlah Kolom", len(df.columns))
    col3.metric("Fitur Dipilih", column_selected)

    # =============================
    # Penjelasan Fitur Dipilih (Emoji + Tabel)
    # =============================
    
    st.subheader(f"📝 Penjelasan Fitur: **{column_selected}**")

    if column_selected in feature_description:
        st.markdown(f"""
        <div style="padding:10px; background-color:#0E2848; border-radius:8px; color:white;">
            <b>{feature_title.get(column_selected, "Informasi Fitur")}</b>
        </div>
        """, unsafe_allow_html=True)

        # Tabel rapi
        df_desc = convert_description_to_table(feature_description[column_selected])
        st.dataframe(df_desc, use_container_width=True)
    else:
        st.warning("Belum ada deskripsi untuk fitur ini.")

    # =============================
    # Distribution Chart
    # =============================
    st.subheader(f"Distribusi Nilai: **{column_selected}**")
    fig = px.histogram(df, x=column_selected)
    st.plotly_chart(fig, use_container_width=True)

    # =============================
    # Pie Chart Distribusi Fitur
    # =============================
    st.subheader(f"Pie Chart: Distribusi nilai pada fitur **{column_selected}**")
    
    pie_data = df[column_selected].value_counts().reset_index()
    pie_data.columns = [column_selected, "count"]

    fig_pie = px.pie(
        pie_data,
        names=column_selected,
        values="count",
        title=f"Distribusi Pie Chart untuk fitur: {column_selected}",
        hole=0.3  # Menjadi donut chart (opsional)
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    # =============================
    # Relationship to Class (edible/poisonous)
    # =============================
    st.subheader(f"Hubungan fitur **{column_selected}** dengan class (edible/poisonous)")

    fig2 = px.histogram(
        df,
        x=column_selected,
        color="class",
        barmode="group"
    )
    st.plotly_chart(fig2, use_container_width=True)



# =================================
# Halaman 2: Berita & Informasi Terkini
# =================================
elif page == "Berita & Informasi Terkini":

    # ====== INFO PENGETAHUAN JAMUR ======
    st.subheader("📚 Fakta Menarik Tentang Jamur")

    fakta_jamur = [
        "Jamur beracun sering punya warna cerah sebagai peringatan alami.",
        "Tidak ada satu pun ciri universal untuk membedakan jamur aman vs beracun.",
        "Beberapa jamur beracun tidak memiliki bau yang mencurigakan.",
        "Amanita phalloides disebut **Death Cap**—salah satu jamur paling mematikan di dunia.",
        "Lebih dari 90% keracunan fatal disebabkan oleh kelompok Amanita.",
        "Jamur termasuk dalam kerajaan biologisnya sendiri, yaitu Fungi. Mereka sebenarnya lebih dekat hubungannya dengan hewan daripada tumbuhan.",
        "Tidak seperti tumbuhan, jamur tidak memiliki klorofil dan tidak dapat membuat makanannya sendiri menggunakan sinar matahari. Mereka menyerap nutrisi dari bahan organik di sekitarnya.",
        "Sekitar 90% tubuh jamur terdiri dari air.",
        "Beberapa jenis jamur dapat memecah plastik, menawarkan solusi potensial untuk polusi."
    ]

    if st.button("🎲 Dapatkan Fakta Jamur Acak"):
        import random
        st.success(random.choice(fakta_jamur))

    from newspaper import Article

# -------------------------------
# 1. List URL Berita (Fixed)
# -------------------------------
    news_list = [
        {
            "title": "Ilmuwan Temukan Jamur Terpahit di Dunia, Aman Dimakan",
            "url": "https://food.detik.com/info-kuliner/d-7888479/ilmuwan-temukan-jamur-terpahit-di-dunia-aman-dimakan"
        },
        {
            "title": "Jamur Liar Bikin Warga Garut dan Subang Masuk Rumah Sakit",
            "url": "https://www.detik.com/jabar/berita/d-7759235/jamur-liar-bikin-warga-garut-dan-subang-masuk-rumah-sakit"
        },
        {
            "title": "Jamur Liar Pembawa Petaka di Garut",
            "url": "https://www.detik.com/jabar/berita/d-7755993/jamur-liar-pembawa-petaka-di-garut"
        },
        {
            "title": "Sekeluarga di Garut Keracunan Usai Santap Jamur Liar",
            "url": "https://www.detik.com/jabar/berita/d-7755127/sekeluarga-di-garut-keracunan-usai-santap-jamur-liar"
        },
        {
            "title": "Geger Jamur Liar Renggut 3 Nyawa di Australia",
            "url": "https://news.detik.com/internasional/d-6867154/geger-jamur-liar-renggut-3-nyawa-di-australia"
        }
    ]

    st.subheader("📰 Berita & Informasi Terkini")

    for item in news_list:
        st.markdown(f"""
        <div style="
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 12px;
            background: #f7f7f7;
            border: 1px solid #e0e0e0;
        ">
            <a href="{item['url']}" target="_blank" style="text-decoration: none; color: #333;">
                <h4 style="margin-bottom: 4px;">{item['title']}</h4>
                <small style="color:#555;">Klik untuk membaca selengkapnya →</small>
            </a>
        </div>
        """, unsafe_allow_html=True)