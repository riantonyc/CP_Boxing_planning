# app_streamlit.py
"""
Streamlit (Coach view) + Penyimpanan SQLite
- Menampilkan hasil klasifikasi (kuat/mampu/kurang_optimal)
  dan Rencana Minggu 1 (Senin-Jumat) setiap hari sebagai blok Markdown
- Menyimpan setiap input + hasil cluster ke SQLite (athletes.db)
- Sidebar: lihat data tersimpan dan unduh CSV
Komentar dan string dalam Bahasa Indonesia.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import ast
import json
import sqlite3
from datetime import datetime

# ---------- Konfigurasi ----------
MODELS_DIR = "models"
PLANS_DIR = "Training_Plans"

SCALE_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "selected_model.pkl")
META_PATH = os.path.join(MODELS_DIR, "selected_model_meta.json")

DB_PATH = "athletes.db"  # file sqlite
FEATURE_COLS = ["VO2_Max","Sprint30m_sec","PushUp_MaxReps","CMJ_Height_cm","Punch36_Speed","Resting_HR_bpm"]

st.set_page_config(page_title="Boxing", layout="centered")

# ---------- Helpers: Artifacts ----------
@st.cache_resource
def load_artifacts():
    """Muat scaler, model, meta dan plans (string)"""
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

    # load training plans (stringified dict) dari Training_Plans/*.txt
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
    """Parse teks plan (stringified dict) menjadi dict python atau None jika gagal."""
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
    """
    Konversi plan['week1'] -> list baris:
    Setiap baris: {'HARI','TUJUAN','BENTUK_LIST','KET_LIST'}
    """
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
                # keterangan detail
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
    """
    Inference wrapper:
    - scale input, prediksi cluster, map cluster->role (meta preferred), ambil plan week1
    - kembalikan dict {cluster, role, plan_rows, raw_plan_text}
    """
    scaler = artifacts.get("scaler")
    model = artifacts.get("model")
    plans_raw = artifacts.get("plans", {})
    meta = artifacts.get("meta", {})

    if scaler is None or model is None:
        return {"error": "Model belum tersedia. Jalankan run_pipeline.py terlebih dahulu untuk membuat model."}

    arr = np.array(values, dtype=float).reshape(1, -1)
    try:
        x_scaled = scaler.transform(arr)[0]
    except Exception as e:
        return {"error": f"Scaling error: {e}"}

    try:
        cluster = int(model.predict([x_scaled])[0])
    except Exception as e:
        return {"error": f"Model inference error: {e}"}

    # mapping cluster->role: prioritas meta jika ada
    cluster_to_role = {}
    if isinstance(meta, dict) and "cluster_to_role" in meta and isinstance(meta["cluster_to_role"], dict):
        try:
            cluster_to_role = {int(k): v for k, v in meta["cluster_to_role"].items()}
        except Exception:
            cluster_to_role = {}
    default_map = {0: "kuat", 1: "mampu", 2: "kurang_optimal"}
    role = cluster_to_role.get(cluster, default_map.get(cluster, "mampu"))

    raw_plan = plans_raw.get(role)
    plan_dict = safe_parse_plan(raw_plan)
    rows = plan_week1_to_rows(plan_dict) if plan_dict is not None else []

    return {"cluster": cluster, "role": role, "plan_rows": rows, "raw_plan_text": raw_plan}

# ---------- Helpers: SQLite ----------
def init_db(db_path=DB_PATH):
    """Inisialisasi DB dan buat table jika belum ada."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS athlete_inputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    conn.commit()
    return conn

def save_record(conn, features: dict, cluster: int, role: str, plan_rows: list):
    """Simpan satu record ke tabel athlete_inputs."""
    cur = conn.cursor()
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    plan_json = json.dumps(plan_rows, ensure_ascii=False)
    cur.execute("""
        INSERT INTO athlete_inputs (
            created_at, VO2_Max, Sprint30m_sec, PushUp_MaxReps, CMJ_Height_cm, Punch36_Speed, Resting_HR_bpm,
            cluster, role, plan_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        ts,
        features.get("VO2_Max"), features.get("Sprint30m_sec"), features.get("PushUp_MaxReps"),
        features.get("CMJ_Height_cm"), features.get("Punch36_Speed"), features.get("Resting_HR_bpm"),
        int(cluster), role, plan_json
    ))
    conn.commit()
    return cur.lastrowid

def fetch_records(conn, limit=1000):
    """Ambil semua record sebagai pandas DataFrame (limit opsional)."""
    df = pd.read_sql_query(f"SELECT * FROM athlete_inputs ORDER BY id DESC LIMIT {limit}", conn)
    if "plan_json" in df.columns:
        try:
            df["plan_preview"] = df["plan_json"].apply(lambda x: (json.loads(x)[0]["HARI"] + " ..." if x and json.loads(x) else "") if x else "")
        except Exception:
            df["plan_preview"] = None
    return df

# ---------- UI ----------
st.title("Coach Dashboard — Week 1 Plan (Simpan ke DB)")
st.write("Masukkan data atlet. Hanya tampilkan hasil (kuat/mampu/kurang optimal) dan Rencana Minggu 1. Anda dapat menyimpan hasil ke database lokal (SQLite).")

# muat artifact
art = load_artifacts()

# inisialisasi DB connection
conn = init_db(DB_PATH)

# Input form
with st.form("athlete_form"):
    st.subheader("Input data atlet")
    vo2 = st.number_input("VO2_Max (ml/kg/min)", min_value=10.0, max_value=100.0, value=55.0, step=0.1)
    sprint = st.number_input("Sprint30m_sec (detik)", min_value=2.5, max_value=10.0, value=5.0, step=0.01)
    push = st.number_input("PushUp_MaxReps", min_value=0, max_value=200, value=50, step=1)
    cmj = st.number_input("CMJ_Height_cm", min_value=5.0, max_value=60.0, value=30.0, step=0.1)
    punch = st.number_input("Punch36_Speed", min_value=0.5, max_value=3.0, value=1.5, step=0.01)
    hr = st.number_input("Resting_HR_bpm", min_value=30, max_value=140, value=70, step=1)

    save_after = st.checkbox("Simpan hasil ke database (SQLite) setelah klasifikasi", value=True)
    submitted = st.form_submit_button("Klasifikasikan & (opsional) Simpan")

if submitted:
    values = [vo2, sprint, push, cmj, punch, hr]
    res = predict_single(values, art)
    if "error" in res:
        st.error(res["error"])
    else:
        cluster = res["cluster"]
        role = res["role"]
        plan_rows = res["plan_rows"]
        st.success(f"Hasil klasifikasi: **{role.upper()}** (cluster {cluster})")

        # tampilkan rencana minggu1
        if not plan_rows:
            st.write("Rencana Minggu 1 tidak tersedia untuk role ini.")
        else:
            st.write("Rencana Latihan — Minggu 1 (Senin–Jumat):")
            for r in plan_rows:
                hari = r.get("HARI", "")
                tujuan = r.get("TUJUAN", "")
                bentuk = r.get("BENTUK_LIST", [])
                ket = r.get("KET_LIST", [])

                md = f"### **{hari}**\n\n"
                if tujuan:
                    md += f"**Tujuan:** {tujuan}\n\n"
                if bentuk:
                    md += "**Bentuk Latihan:**\n"
                    for b in bentuk:
                        md += f"- {b}\n"
                    md += "\n"
                if ket:
                    md += "**Keterangan:**\n"
                    for k in ket:
                        md += f"- {k}\n"
                    md += "\n"
                md += "---\n"
                st.markdown(md)

        # simpan jika diminta
        if save_after:
            features = {
                "VO2_Max": vo2, "Sprint30m_sec": sprint, "PushUp_MaxReps": push,
                "CMJ_Height_cm": cmj, "Punch36_Speed": punch, "Resting_HR_bpm": hr
            }
            try:
                rowid = save_record(conn, features, cluster, role, plan_rows)
                st.success(f"Hasil tersimpan ke database (id={rowid}).")
            except Exception as e:
                st.error(f"Gagal menyimpan ke DB: {e}")

# ---------- Sidebar: lihat & unduh data ----------
st.sidebar.header("Data Tersimpan (SQLite)")
with st.sidebar.expander("Lihat data terakhir"):
    try:
        df_saved = fetch_records(conn, limit=500)
        if df_saved.shape[0] == 0:
            st.write("Belum ada data tersimpan.")
        else:
            st.dataframe(df_saved[["id","created_at","VO2_Max","Sprint30m_sec","PushUp_MaxReps","CMJ_Height_cm","Punch36_Speed","Resting_HR_bpm","cluster","role"]])
            csv = df_saved.to_csv(index=False).encode("utf-8")
            st.download_button("Unduh CSV (semua kolom)", csv, "athletes_saved.csv", "text/csv")
    except Exception as e:
        st.write("Gagal membaca DB:", e)

# with st.sidebar.expander("Hapus semua data (opsional)"):
#     if st.button("Hapus semua entri (PERMANEN)"):
#         try:
#             cur = conn.cursor()
#             cur.execute("DELETE FROM athlete_inputs")
#             conn.commit()
#             st.sidebar.success("Semua data telah dihapus.")
#         except Exception as e:
#             st.sidebar.error(f"Gagal menghapus: {e}")

st.sidebar.caption("Database: " + DB_PATH)
st.caption("Catatan: Gunakan penilaian pelatih untuk finalisasi. Aplikasi hanya memberi rekomendasi berbasis cluster.")
