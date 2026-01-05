# app_streamlit.py
"""
Streamlit App: Boxing Training System
Fitur:
1. Klasifikasi Atlet (Input Nama + Tahun Lahir untuk Unik)
2. Tampilan Rencana Latihan Detail (Markdown)
3. Profil Sasana
4. Data Management (Admin Login + Statistik)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import ast
import json
import sqlite3
import hashlib
from datetime import datetime

# ---------- Konfigurasi ----------
MODELS_DIR = "models"
PLANS_DIR = "Training_Plans"
ASSETS_DIR = "assets"

SCALE_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "selected_model.pkl")
META_PATH = os.path.join(MODELS_DIR, "selected_model_meta.json")

DB_PATH = "athletes.db"

# Setup Page Config
st.set_page_config(
    page_title="Boxing MBC Training System",
    page_icon="🥊",
    layout="wide"
)

# ---------- Helpers: Security & Auth ----------
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

# ---------- Helpers: Artifacts ----------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    artifacts["scaler"] = joblib.load(SCALE_PATH) if os.path.exists(SCALE_PATH) else None
    artifacts["model"] = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    artifacts["meta"] = None
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                artifacts["meta"] = json.load(f)
        except Exception:
            artifacts["meta"] = None

    artifacts["plans"] = {}
    if os.path.isdir(PLANS_DIR):
        for fname in os.listdir(PLANS_DIR):
            if fname.lower().endswith(".txt"):
                role = os.path.splitext(fname)[0]
                path = os.path.join(PLANS_DIR, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        artifacts["plans"][role] = f.read()
                except Exception:
                    artifacts["plans"][role] = None
    return artifacts

def safe_parse_plan(raw_text):
    if raw_text is None:
        return None
    try:
        parsed = ast.literal_eval(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return None

def plan_week1_to_rows(plan_dict):
    rows = []
    if not plan_dict or "week1" not in plan_dict:
        return rows

    week1 = plan_dict["week1"]
    for day_obj in week1:
        day = day_obj.get("day", "")
        fokus = day_obj.get("focus", "")
        items = day_obj.get("items", [])
        bentuk_list = []
        ket_list = []
        if isinstance(items, list):
            for it in items:
                name = it.get("name", "") if isinstance(it, dict) else str(it)
                bentuk_list.append(name)
                detail_parts = []
                if isinstance(it, dict):
                    for key in ["sets","reps","rounds","duration_min","round_duration_min","rest_s","exercises","duration_each_s"]:
                        if key in it:
                            val = it[key]
                            if isinstance(val, list):
                                val = ", ".join(map(str, val))
                            detail_parts.append(f"{key}: {val}")
                ket_text = f"{name} ({'; '.join(detail_parts)})" if detail_parts else name
                ket_list.append(ket_text)
        else:
            bentuk_list.append(str(items))
            ket_list.append(str(items))

        rows.append({
            "HARI": day,
            "TUJUAN": fokus,
            "BENTUK_LIST": bentuk_list,
            "KET_LIST": ket_list
        })
    return rows

def predict_single(values, artifacts):
    scaler = artifacts.get("scaler")
    model = artifacts.get("model")
    plans_raw = artifacts.get("plans", {})
    meta = artifacts.get("meta", {})

    if scaler is None or model is None:
        return {"error": "Model error: Artifacts not found."}

    arr = np.array(values, dtype=float).reshape(1, -1)
    try:
        x_scaled = scaler.transform(arr)[0]
    except Exception:
        return {"error": "Scaling error"}

    try:
        cluster = int(model.predict([x_scaled])[0])
    except Exception:
        return {"error": "Inference error"}

    cluster_to_role = {}
    if isinstance(meta, dict) and "cluster_to_role" in meta:
        cluster_to_role = {int(k): v for k, v in meta["cluster_to_role"].items()}
    
    default_map = {0: "kuat", 1: "mampu", 2: "kurang_optimal"}
    role = cluster_to_role.get(cluster, default_map.get(cluster, "mampu"))

    raw_plan = plans_raw.get(role)
    plan_dict = safe_parse_plan(raw_plan)
    rows = plan_week1_to_rows(plan_dict) if plan_dict is not None else []

    return {"cluster": cluster, "role": role, "plan_rows": rows}

# ---------- Helpers: SQLite (Data & User) ----------
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    
    # Table Data Atlet
    cur.execute("""
    CREATE TABLE IF NOT EXISTS athlete_inputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        custom_id TEXT,
        athlete_name TEXT,
        created_at TEXT,
        VO2_Max REAL,
        Sprint30m_sec REAL,
        PushUp_MaxReps INTEGER,
        CMJ_Height_cm REAL,
        Punch36_Speed REAL,
        Resting_HR_bpm INTEGER,
        cluster INTEGER,
        role TEXT,
        plan_json TEXT
    )
    """)
    
    # --- AUTO MIGRATION (Untuk update DB lama jika ada) ---
    try:
        cur.execute("SELECT custom_id FROM athlete_inputs LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE athlete_inputs ADD COLUMN custom_id TEXT")
            cur.execute("ALTER TABLE athlete_inputs ADD COLUMN athlete_name TEXT")
            conn.commit()
        except Exception:
            pass 
    
    # Table User (Untuk Login Admin)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
    """)
    conn.commit()
    return conn

def create_user(conn, username, password):
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users(username, password) VALUES (?,?)", (username, make_hashes(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False 

def login_user(conn, username, password):
    cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE username = ?", (username,))
    data = cur.fetchone()
    if data:
        if check_hashes(password, data[0]):
            return True
    return False

def save_record(conn, name, birth_year, features, cluster, role, plan_rows):
    cur = conn.cursor()
    now = datetime.now()
    ts = now.isoformat(sep=" ", timespec="seconds")
    
    # --- LOGIKA IDENTITAS UNIK (SOLUSI NAMA SAMA) ---
    # Nama disimpan gabung dengan tahun lahir: "RIAN (2000)"
    clean_name = name.strip().upper()
    if not clean_name: clean_name = "ATLET"
    
    full_identity_name = f"{clean_name} ({birth_year})"
    
    # Custom ID Unik untuk tracking sesi
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    custom_id = f"{date_str}-{clean_name}{birth_year}-{time_str}"
    
    plan_json = json.dumps(plan_rows, ensure_ascii=False)
    
    cur.execute("""
        INSERT INTO athlete_inputs (
            custom_id, athlete_name, created_at, 
            VO2_Max, Sprint30m_sec, PushUp_MaxReps, CMJ_Height_cm, Punch36_Speed, Resting_HR_bpm,
            cluster, role, plan_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        custom_id, full_identity_name, ts,
        features["VO2_Max"], features["Sprint30m_sec"], features["PushUp_MaxReps"],
        features["CMJ_Height_cm"], features["Punch36_Speed"], features["Resting_HR_bpm"],
        int(cluster), role, plan_json
    ))
    conn.commit()
    return custom_id, full_identity_name

def fetch_records(conn, limit=1000):
    return pd.read_sql_query(f"SELECT * FROM athlete_inputs ORDER BY id DESC LIMIT {limit}", conn)

# ---------- Page: Header ----------
def show_header():
    col1, col2 = st.columns([1, 6])
    with col1:
        # Ganti URL ini dengan st.image("assets/logo.png") jika ada
        st.image("img/MBC_LOGO.png", use_container_width=True)
    with col2:
        st.title("MBC Boxing Camp System")
        st.markdown("**Sistem Cerdas Klasifikasi & Manajemen Data Atlet**")
    st.markdown("---")

# ---------- Page: Profil Sasana (TAMPILAN TETAP) ----------
def show_profile():
    st.header("Profil Sasana & Aplikasi")
    
    # Foto Sasana
    st.image("img/Boxing.jpg", 
             caption="Boxing", use_container_width=True)
    
    # Deskripsi Sasana
    st.subheader("Tentang MBC Boxing Camp")
    st.write("""
MBC Boxing Camp adalah pusat pelatihan tinju amatir terkemuka yang beroperasi di bawah naungan resmi organisasi Persatuan Tinju Amatir Indonesia (PERTINA). Kami berfokus pada pengembangan atlet secara holistik, mulai dari pembentukan kemampuan dasar dan fundamental hingga mencapai tingkat keahlian kompetitif tertinggi.
Program pembinaan kami dirancang secara sistematis untuk menghasilkan atlet-atlet berprestasi yang siap dipertandingkan. Setiap atlet di MBC akan menjalani proses seleksi dan evaluasi berkelanjutan untuk kemudian dipilih dan dipersiapkan secara intensif guna mewakili daerah dan sasana di berbagai kompetisi tinju amatir tingkat regional hingga nasional di bawah bendera PERTINA.
MBC berkomitmen menjadi pilar dalam menciptakan generasi petinju profesional Indonesia masa depan.
    """)

    st.markdown("---")

    # Profil Pembuat
    st.subheader("Pengembang Sistem")
    col_a, col_b = st.columns([1, 3])
    
    with col_a:
        st.image("img/ryantonyc.jpg", caption="Developer", use_container_width=True)
        
    with col_b:
        st.markdown("""
        **RYANTONY CANAVARRO (232102565)**
        Sistem ini saya bangun untuk membantu pelatih (Coach) dalam menentukan program latihan yang efektif serta cepat melihat dan menentukan batas kemampuan atlet.
        Menggunakan algoritma *Machine Learning*, sistem ini mengelompokkan atlet menjadi:
        - **Kuat**
        - **Sedang (Mampu)**
        - **Lemah (Kurang Optimal)**
        """)

# ---------- Page: Klasifikasi (Coach) ----------
def show_classification(art, conn):
    st.header("🔍 Klasifikasi Atlet & Rencana Latihan")
    
    # Form Input
    with st.form("athlete_form"):
        st.markdown("### 1. Identitas Atlet")
        c1, c2 = st.columns([3, 1])
        with c1:
            name_input = st.text_input("Nama Panggilan", placeholder="Contoh: Rian")
        with c2:
            birth_year = st.number_input("Tahun Lahir", min_value=1970, max_value=2015, value=2000, step=1)
        
        st.caption(f"Identitas Unik: {name_input.upper()} ({birth_year})")

        st.markdown("### 2. Parameter Fisik")
        col1, col2 = st.columns(2)
        with col1:
            vo2 = st.number_input("VO2 Max", 10.0, 100.0, 55.0)
            sprint = st.number_input("Sprint 30m (s)", 2.0, 15.0, 5.0)
            push = st.number_input("Push Up Max", 0, 200, 50)
        with col2:
            cmj = st.number_input("CMJ Height (cm)", 5.0, 100.0, 30.0)
            punch = st.number_input("Punch Speed", 0.1, 5.0, 1.5)
            hr = st.number_input("Resting HR", 30, 200, 70)
        
        save_check = st.checkbox("Simpan ke Database", value=True)
        submitted = st.form_submit_button("Analisis", type="primary")

    if submitted:
        if not name_input.strip():
            st.warning("⚠️ Harap isi Nama Atlet.")
        else:
            vals = [vo2, sprint, push, cmj, punch, hr]
            res = predict_single(vals, art)
            
            if "error" in res:
                st.error(res["error"])
            else:
                role = res["role"]
                st.success(f"Hasil Klasifikasi: **{role.upper()}** (Cluster {res['cluster']})")
                
                # --- TAMPILAN RENCANA LATIHAN DETAIL (SEPERTI SEBELUMNYA) ---
                st.markdown("### 📅 Rencana Latihan (Minggu 1)")
                if not res["plan_rows"]:
                    st.write("Rencana Minggu 1 tidak tersedia untuk role ini.")
                else:
                    for r in res["plan_rows"]:
                        hari = r.get("HARI", "")
                        tujuan = r.get("TUJUAN", "")
                        bentuk = r.get("BENTUK_LIST", [])
                        ket = r.get("KET_LIST", [])

                        # Format Markdown Detail
                        md = f"#### **{hari}**\n"
                        if tujuan:
                            md += f"**Tujuan:** {tujuan}\n\n"
                        if bentuk:
                            md += "**Bentuk Latihan:**\n"
                            for b in bentuk:
                                md += f"- {b}\n"
                            md += "\n"
                        if ket:
                            md += "**Detail/Keterangan:**\n"
                            for k in ket:
                                md += f"- {k}\n"
                        
                        st.markdown(md)
                        st.markdown("---") # Garis pemisah antar hari

                # Simpan ke DB
                if save_check:
                    feat = {"VO2_Max":vo2, "Sprint30m_sec":sprint, "PushUp_MaxReps":push, "CMJ_Height_cm":cmj, "Punch36_Speed":punch, "Resting_HR_bpm":hr}
                    
                    # Panggil fungsi save dengan tahun lahir
                    new_id, saved_name = save_record(conn, name_input, birth_year, feat, res["cluster"], role, res["plan_rows"])
                    
                    st.toast(f"Data Tersimpan! ID: {new_id}")
                    st.info(f"✅ Data tersimpan untuk **{saved_name}**.")

# ---------- Page: Data Management (Admin) ----------
def show_data_management(conn):
    st.header("🔐 Data Management Center")

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # Login System
    if not st.session_state['logged_in']:
        tab_login, tab_reg = st.tabs(["Login", "Sign Up"])
        with tab_login:
            st.subheader("Login Admin")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Masuk"):
                if login_user(conn, username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Gagal Login.")
        with tab_reg:
            st.subheader("Daftar Akun Baru")
            new_user = st.text_input("Username Baru", key="reg_user")
            new_pass = st.text_input("Password Baru", type="password", key="reg_pass")
            if st.button("Daftar"):
                if create_user(conn, new_user, new_pass):
                    st.success("Akun dibuat. Silakan login.")
                else:
                    st.warning("Username sudah ada.")
    
    else:
        # Dashboard Admin
        st.success(f"Login sebagai: **{st.session_state['username']}**")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()
        
        st.markdown("---")
        df = fetch_records(conn, limit=1000)
        
        if df.empty:
            st.info("Belum ada data.")
            return

        # 1. Statistik
        col_a, col_b = st.columns(2)
        col_a.metric("Total Tes", len(df))
        unique_athletes = df['athlete_name'].nunique() if 'athlete_name' in df.columns else 0
        col_b.metric("Total Atlet Unik", unique_athletes)

        # 2. Visualisasi
        st.subheader("📊 Statistik")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Distribusi Klasifikasi")
            st.bar_chart(df['role'].value_counts(), color="#FF4B4B")
        with c2:
            st.caption("Tren Input Harian")
            df['date'] = pd.to_datetime(df['created_at']).dt.date
            st.line_chart(df['date'].value_counts().sort_index())

        # 3. Tabel Data (Dengan Nama & ID)
        st.subheader("📋 Data Atlet")
        cols_show = ["custom_id", "athlete_name","role", "created_at"]
        valid_cols = [c for c in cols_show if c in df.columns]
        
        st.dataframe(df[valid_cols], use_container_width=True, hide_index=True)
        
        # 3. Tabel Data & Download
        st.subheader("📋 Database Lengkap")
        
        # 4. Menampilkan Preview (5 Baris Teratas, Semua Kolom)
        st.write("Preview 5 Data Terbaru (Menampilkan semua kolom):")
        

        st.dataframe(df.head(5), use_container_width=True, hide_index=True)
        
        # B. Tombol Unduh (Mengunduh SELURUH Data, bukan cuma 5)
        csv = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Unduh CSV (Semua Data Lengkap)",
            data=csv,
            file_name="data_atlet_full.csv",
            mime="text/csv",
            type="primary" # Membuat tombol berwarna merah/utama agar terlihat jelas
        )

# ---------- Main Execution ----------
def main():
    art = load_artifacts()
    conn = init_db(DB_PATH)
    
    with st.sidebar:
        st.header("Menu Utama")
        menu = st.radio("Pilih Halaman:", 
            ["Klasifikasi (Coach)", "Profil Sasana", "Data Management (Admin)"]
        )
        st.markdown("---")
        st.caption("MBC System v1.0")

    show_header()

    if menu == "Klasifikasi (Coach)":
        show_classification(art, conn)
    elif menu == "Profil Sasana":
        show_profile()
    elif menu == "Data Management (Admin)":
        show_data_management(conn)




if __name__ == "__main__":
    main()