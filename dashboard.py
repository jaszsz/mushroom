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
    "bruises": "💥 Perubahan Warna Saat Memar: true=ya (t), false=tidak (f)",
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
    ["Informasi Dataset", "Hasil Analisis", "Prediksi Jamur", "Penelitian Serupa", "Berita & Informasi Terkini"]
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
if page == "Berita & Informasi Terkini":

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


# -------------------------------
# 1. List URL Berita (Fixed)
# -------------------------------
    import base64

    def img_to_base64(path):
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    
    news_items = [
        {
            "title": "Ilmuwan Temukan Jamur Terpahit di Dunia, Aman Dimakan",
            "img": img_to_base64("gambar/ilmuwan.png"),
            "url": "https://food.detik.com/info-kuliner/d-7888479/ilmuwan-temukan-jamur-terpahit-di-dunia-aman-dimakan"
        },
        {
            "title": "Jamur Liar Bikin Warga Garut dan Subang Masuk Rumah Sakit",
            "img": img_to_base64("gambar/keracunan.jpeg"),
            "url": "https://www.detik.com/jabar/berita/d-7759235/jamur-liar-bikin-warga-garut-dan-subang-masuk-rumah-sakit"
        },
        {
            "title": "Jamur Liar Pembawa Petaka di Garut",
            "img": img_to_base64("gambar/petaka.jpeg"),
            "url": "https://www.detik.com/jabar/berita/d-7755993/jamur-liar-pembawa-petaka-di-garut"
        },
        {
            "title": "Sekeluarga di Garut Keracunan Usai Santap Jamur Liar",
            "img": img_to_base64("gambar/sekeluarga.jpeg"),
            "url": "https://www.detik.com/jabar/berita/d-7755127/sekeluarga-di-garut-keracunan-usai-santap-jamur-liar"
        },
        {
            "title": "Geger Jamur Liar Renggut 3 Nyawa di Australia",
            "img": img_to_base64("gambar/geger.jpeg"),
            "url": "https://news.detik.com/internasional/d-6867154/geger-jamur-liar-renggut-3-nyawa-di-australia"
        }
    ]


    st.header("📰 Berita & Informasi Terkini")
    # --- Swiper + CSS ---
    st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.css"/>
    <style>
    .swiper {
        width: 100%;
        height: 480px;
    }
    .swiper-slide {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
    }
    .slide-info {
        position: absolute;
        bottom: 0;
        width: 100%;
        padding: 18px 20px;
        background: linear-gradient(to top, rgba(0,0,0,0.75), rgba(0,0,0,0));
        color: white;
    }
    .slide-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .slide-title a {
        color: white;
        text-decoration: none;
    }
    .slide-title a:hover {
        text-decoration: underline;
    }
    .slide-img {
        width: 100%;
        height: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Generate slides ---
    for item in news_items:
        st.markdown(f"""
        <div class="swiper-slide">
            <img class="slide-img" src="data:image/jpeg;base64,{item['img']}"/>
            <div class="slide-info">
                <div class="slide-title">
                    <a href="{item['url']}" target="_blank">{item['title']}</a>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Close HTML + Swiper JS ---
    st.markdown("""
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.js"></script>
    <script>
    const swiper = new Swiper('.swiper', {
        loop: true,
        autoplay: {
            delay: 3000,
            disableOnInteraction: false,
        },
        slidesPerView: 1,
        spaceBetween: 20,
    });
    </script>
    """, unsafe_allow_html=True)


elif page == "Hasil Analisis":
    st.title("🎯 Evaluasi Kinerja Model Machine Learning")
    st.markdown("""
    Pada penelitian ini dibandingkan dua model Machine Learning, yaitu **XGBoost** dan **Random Forest**, dalam memprediksi klasifikasi jamur (edible/poisonous). Dataset yang digunakan memiliki pola yang sangat terstruktur sehingga kedua model mencapai performa sempurna pada data uji.          
    """)
    import time

    if st.button("Reveal Best Model"):
        with st.spinner("Evaluating models..."):
            time.sleep(1.2)
        st.success("🌳 Best Model: **Random Forest** ✨")

    # ============================
    # Header
    # ============================
    st.markdown("""
    <style>
    .metric-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #1E1E1E;
        color: white;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 16px;
        opacity: 0.7;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================
# Performance Table
# ============================
    df = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest"],
        "Accuracy": [1.0, 1.0],
        "Sensitivity": [1.0, 1.0],
        "Specificity": [1.0, 1.0],
        "AUC": [1.0, 1.0],
        "MCC": [1.0, 1.0],
    })

    st.subheader("📋 Tabel Perbandingan Metrik Model")
    st.dataframe(df, use_container_width=True)

# ============================
# Metric Cards (Highlight MCC)
# ============================
    st.subheader("🏆 Ringkasan Performa Model")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">XGBoost</div>
            <br>
            <div class="metric-label">MCC</div>
            <div class="metric-value">1.00</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">Random Forest</div>
            <br>
            <div class="metric-label">MCC</div>
            <div class="metric-value">1.00</div>
        </div>
        """, unsafe_allow_html=True)


# ============================
# Chart Visualization
# ============================
    st.subheader("📊 Visualisasi Perbandingan Metrik")

    df_chart = df.set_index("Model")
    st.bar_chart(df_chart, use_container_width=True)

# ============================
# Section: Visualisasi Confusion Matrix & Feature Importance
# ============================

    st.subheader("🖼️ Visualisasi")

    st.markdown("### 🔍 Confusion Matrix")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Random Forest**")
        st.image("gambar/cm rf.png", width=1200)

    with col2:
        st.markdown("**XGBoost**")
        st.image("gambar/cm xgboost.png", width=1200)


    st.markdown("### 🌿 Feature Importance")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Random Forest**")
        st.image("gambar/fi rf.png", width=1200)

    with col4:
        st.markdown("**XGBoost**")
        st.image("gambar/fi xgboost.png", width=1200)

# ============================
# Additional Insight Section
# ============================
    st.subheader("📝 Insight Tambahan")
    st.info("""
    Dataset jamur memiliki fitur kategorikal yang sangat informatif  seperti **odor**, **gill-color**, dan **spore-print-color**. Fitur-fitur ini sangat mudah dipisahkan oleh algoritma tree-based sehingga hampir semua model bahkan model sederhana pun dapat mencapai performa sempurna.
    """)

df_training = pd.DataFrame({
    "Output": ["Correctly Classified", "MAE", "RMSE"],
    "C4.5": ["100%", "0", "0"],
    "Naive Bayes": ["95.887%", "0.0405", "0.1718"],
    "Support Vector Machine": ["100%", "0", "0"]
})

df_testing = pd.DataFrame({
    "Output": ["Correctly Classified", "MAE", "RMSE"],
    "C4.5": ["100%", "0", "0"],
    "Naive Bayes": ["95.827%", "0.0419", "0.1757"],
    "Support Vector Machine": ["100%", "0", "0"]
})

if page == "Penelitian Serupa":

    st.header("Penelitian Serupa")

# --- Pilihan Penelitian ---
    options = {
        "— Pilih Penelitian —": "opsi",
        "Classification Algorithm for Edible Mushroom Identification (Agung Wibowo dkk.)": "penelitian1",
        "Accuracy of classification poisonous or edible of mushroom using Naïve Bayes and KNN (Roni Hamonangan dkk.)": "penelitian2"
    }

    selected = st.selectbox("Pilih Penelitian:", list(options.keys()))

    st.markdown("---")

# --- Layout Utama ---
    col_left, col_right = st.columns([1.3, 2])

# ============================================
# PENELITIAN 1
# ============================================
    if options[selected] == "penelitian1":
    # ---------- KIRI: Penulis ----------
        with col_left:
            st.subheader("Penulis")
            st.markdown("""
            - Agung Wibowo  
            - Yuri Rahayu
            - Andi Riyanto
            - Taufik Hidayatulloh
            """)


    # ---------- KANAN: Card kota | tahun | metode ----------
        with col_right:
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("""
                <div style='padding:10px; border-radius:10px; background:#1E1E1E; text-align:center;'>
                    <b>Kota</b><br>
                    Yogyakarta, Indonesia
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown("""
                <div style='padding:10px; border-radius:10px; background:#1E1E1E; text-align:center;'>
                    <b>Tahun</b><br>
                    2018
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown("""
                <div style='padding:10px; border-radius:10px; background:#1E1E1E; text-align:center;'>
                    <b>Metode</b><br>
                    C4.5, Naive Bayes, Support Vector Machine
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### Hasil Evaluasi")
        pilihan_tabel = st.radio("Pilih tabel:", ["Training Set", "Testing Set"])
        if pilihan_tabel == "Training Set":
            st.dataframe(df_training)
        else:
            st.dataframe(df_testing)


# ============================================
# PENELITIAN 2
# ============================================
    elif options[selected] == "penelitian2":
    # ---------- KIRI: Penulis ----------
        with col_left:
            st.subheader("Penulis")
            st.markdown("""
            - Roni Hamonangan  
            - Meidika Bagus Saputro
            - Cecep Bagus Surya Dinata Karta Atmaja
            """)

    # ---------- KANAN: Card kota | tahun | metode ----------
        with col_right:
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("""
                <div style='padding:10px; border-radius:10px; background:#1E1E1E; text-align:center;'>
                    <b>Kota</b><br>
                    Semarang, Indonesia
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown("""
                <div style='padding:10px; border-radius:10px; background:#1E1E1E; text-align:center;'>
                    <b>Tahun</b><br>
                    2021
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown("""
                <div style='padding:10px; border-radius:10px; background:#1E1E1E; text-align:center;'>
                    <b>Metode</b><br>
                    Naive Bayes, K-Nearest Neighbors
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### Hasil Evaluasi")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="metric-value">Naive Bayes</div>
                <br>
                <div class="metric-label">Akurasi</div>
                <div class="metric-value">90,2%</div>
                <br>
                <div class="metric-label">Confusion Matrix</div>
            </div>
            """, unsafe_allow_html=True,)
            st.image("gambar/cm nb.png", width=1200)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="metric-value">K-Nearest Neighbors</div>
                <br>
                <div class="metric-label">Akurasi</div>
                <div class="metric-value">100%</div>
                <br>
                <div class="metric-label">Confusion Matrix</div>
            </div>
            """, unsafe_allow_html=True)
            st.image("gambar/cm knn.png", width=1200)


import pickle

# ===============================
# LOAD MODEL + ENCODERS
# ===============================

with open("rf_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
encoders = saved["encoders"]          # fitur
class_encoder = saved["class_encoder"]   # khusus class
feature_cols = saved["features"]


if page == "Prediksi Jamur":
    st.title("🍄 Prediksi Jamur: Aman atau Beracun")

    st.write("Silakan isi 22 fitur jamur di bawah ini untuk memprediksi apakah jamur aman (edible) atau beracun (poisonous).")
    user_input = {}
    
    c1, c2, c3 = st.columns(3)
    with c1:
        user_input["cap-shape"] = st.selectbox("cap-shape", encoders["cap-shape"].classes_, help=feature_description["cap-shape"])
        user_input["cap-surface"] = st.selectbox("cap-surface", encoders["cap-surface"].classes_, help=feature_description["cap-surface"])
        user_input["cap-color"] = st.selectbox("cap-color", encoders["cap-color"].classes_, help=feature_description["cap-color"])
        user_input["bruises"] = st.selectbox("bruises", encoders["bruises"].classes_, help=feature_description["bruises"])
        user_input["odor"] = st.selectbox("odor", encoders["odor"].classes_, help=feature_description["odor"])
        user_input["ring-number"] = st.selectbox("ring-number", encoders["ring-number"].classes_, help=feature_description["ring-number"])
        user_input["ring-type"] = st.selectbox("ring-type", encoders["ring-type"].classes_, help=feature_description["ring-type"])
        user_input["spore-print-color"] = st.selectbox("spore-print-color", encoders["spore-print-color"].classes_, help=feature_description["spore-print-color"])
    with c2:
        user_input["gill-attachment"] = st.selectbox("gill-attachment", encoders["gill-attachment"].classes_, help=feature_description["gill-attachment"])
        user_input["gill-spacing"] = st.selectbox("gill-spacing", encoders["gill-spacing"].classes_, help=feature_description["gill-spacing"])
        user_input["gill-size"] = st.selectbox("gill-size", encoders["gill-size"].classes_, help=feature_description["gill-size"])
        user_input["gill-color"] = st.selectbox("gill-color", encoders["gill-color"].classes_, help=feature_description["gill-color"])
        user_input["veil-type"] = st.selectbox("veil-type", encoders["veil-type"].classes_, help=feature_description["veil-type"])
        user_input["veil-color"] = st.selectbox("veil-color", encoders["veil-color"].classes_, help=feature_description["veil-color"])
        user_input["population"] = st.selectbox("population", encoders["population"].classes_, help=feature_description["population"])
        user_input["habitat"] = st.selectbox("habitat", encoders["habitat"].classes_, help=feature_description["habitat"])
    with c3:
        user_input["stalk-shape"] = st.selectbox("stalk-shape", encoders["stalk-shape"].classes_, help=feature_description["stalk-shape"])
        user_input["stalk-root"] = st.selectbox("stalk-root", encoders["stalk-root"].classes_, help=feature_description["stalk-root"])
        user_input["stalk-surface-above-ring"] = st.selectbox("stalk-surface-above-ring", encoders["stalk-surface-above-ring"].classes_, help=feature_description["stalk-surface-above-ring"])
        user_input["stalk-surface-below-ring"] = st.selectbox("stalk-surface-below-ring", encoders["stalk-surface-below-ring"].classes_, help=feature_description["stalk-surface-below-ring"])
        user_input["stalk-color-above-ring"] = st.selectbox("stalk-color-above-ring", encoders["stalk-color-above-ring"].classes_, help=feature_description["stalk-color-above-ring"])
        user_input["stalk-color-below-ring"] = st.selectbox("stalk-color-below-ring", encoders["stalk-color-below-ring"].classes_, help=feature_description["stalk-color-below-ring"])


    # ====== TOMBOL PREDIKSI ======
    if st.button("Prediksi Jamur"):
        # convert ke DataFrame
        df = pd.DataFrame([user_input])

        # ENCODE input sesuai label encoder training
        for col in df.columns:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])

        # Prediksi
        df = df.reindex(columns=feature_cols)
        pred = model.predict(df)[0]   # hasil angka: 0 atau 1
        class_map = {0: "Edible (e)", 1: "Poisonous (p)"}
        pred_label = class_map[pred]

# Tampilkan hasil
        if pred == 0:
            st.success(f"🌱 **EDIBLE — Jamur Aman Dimakan**\n\nHasil model: {pred_label}")
        elif pred == 1:
            st.error(f"☠️ **POISONOUS — Jamur Beracun**\n\nHasil model: {pred_label}")
