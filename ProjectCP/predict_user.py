# predict_user.py (versi robust, komentar Bahasa Indonesia)
"""
Loader inference untuk model terpilih (KMeans/GMM).
Tahan jika file hasil pipeline hilang / ada layout sedikit berbeda.
Logika:
- muat scaler & model jika ada
- bangun mapping cluster->role dari (prioritas):
    1) meta selected_model_meta.json (jika berisi cluster_to_role)
    2) results/cluster_strength_scores.csv
    3) results/profile_kmeans.csv (hitung skor dari profil)
    4) fallback default {0:kuat,1:mampu,2:kurang_optimal}
- fungsi utama: predict_from_list(values, check_distance_to_center=False, distance_threshold=None)
"""

import joblib
import json
import pandas as pd
import numpy as np
import os

# Fitur input (urut)
FEATURE_COLS = ["VO2_Max","Sprint30m_sec","PushUp_MaxReps","CMJ_Height_cm","Punch36_Speed","Resting_HR_bpm"]

# Direktori / path
models_dir = "models"
results_dir = "results"
selected_model_path = os.path.join(models_dir, "selected_model.pkl")
meta_path = os.path.join(models_dir, "selected_model_meta.json")

# Muat scaler jika ada
if os.path.exists(os.path.join(models_dir, "scaler.pkl")):
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
else:
    scaler = None
    print("Warning: models/scaler.pkl tidak ditemukan. Jalankan run_pipeline.py terlebih dahulu untuk menghasilkan artifact.")

# Muat model yang dipilih
selected_model = None
selected_method = None
if os.path.exists(selected_model_path):
    selected_model = joblib.load(selected_model_path)
else:
    print("Warning: models/selected_model.pkl tidak ditemukan. Jalankan run_pipeline.py untuk melatih & menyimpan model.")

# Muat meta jika ada (akan membantu mapping cluster->role)
meta = {}
if os.path.exists(meta_path):
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            selected_method = meta.get("selected_method", None)
    except Exception:
        meta = {}
        selected_method = None

# Path untuk file hasil pipeline
score_csv = os.path.join(results_dir, "cluster_strength_scores.csv")
profile_csv = os.path.join(results_dir, "profile_kmeans.csv")

# -------------------------
# Fungsi bantu untuk membangun mapping
# -------------------------
def build_mapping_from_meta(meta_obj):
    """Coba ambil mapping cluster_to_role dari meta (jika ada)."""
    try:
        ctr = meta_obj.get("cluster_to_role")
        if isinstance(ctr, dict) and len(ctr) > 0:
            # pastikan keys -> int
            out = {int(k): v for k, v in ctr.items()}
            return out
    except Exception:
        pass
    return None


def build_mapping_from_score_csv(path):
    """Baca cluster_strength_scores.csv (fleksibel) dan kembalikan score_df terurut."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # jika ada kolom 'kmeans_label'
    if "kmeans_label" in df.columns:
        try:
            df2 = df.set_index("kmeans_label").sort_values("strength_score", ascending=False)
            return df2
        except Exception:
            pass
    # heuristik: cari kolom yang tampak seperti label
    for col in df.columns:
        if col.lower().startswith("kmean") or col.lower().startswith("cluster") or col.lower().startswith("index"):
            try:
                df2 = df.set_index(col).sort_values("strength_score", ascending=False)
                return df2
            except Exception:
                continue
    # jika tidak ditemukan struktur, coba asumsi index sebagai label
    if df.shape[1] >= 1:
        try:
            if "strength_score" in df.columns:
                df2 = df.set_index(df.columns[0]).sort_values("strength_score", ascending=False)
                return df2
        except Exception:
            pass
    return None


def build_mapping_from_profile_csv(path):
    """Jika score CSV tidak ada, gunakan profile_kmeans.csv untuk menghitung skor dan urutkan."""
    try:
        prof = pd.read_csv(path, index_col=0)
    except Exception:
        return None
    try:
        def strength_score_row(r):
            return r["VO2_Max"] - r["Sprint30m_sec"] + r["PushUp_MaxReps"] + r["CMJ_Height_cm"] + r["Punch36_Speed"] - r["Resting_HR_bpm"]
        scores = {}
        for idx in prof.index:
            row = prof.loc[idx]
            scores[int(idx)] = float(strength_score_row(row))
        score_df = pd.DataFrame.from_dict(scores, orient="index", columns=["strength_score"])
        score_df = score_df.sort_values("strength_score", ascending=False)
        return score_df
    except Exception:
        return None

# -------------------------
# Bentuk akhir mapping cluster->role (prioritas: meta -> score csv -> profile -> default)
# -------------------------
cluster_to_role = {}

# 1) coba meta
meta_map = build_mapping_from_meta(meta)
if meta_map is not None:
    cluster_to_role = meta_map
else:
    # 2) coba score csv
    score_df = None
    if os.path.exists(score_csv):
        score_df = build_mapping_from_score_csv(score_csv)
    # 3) coba profile csv jika score_df tidak ada
    if score_df is None and os.path.exists(profile_csv):
        score_df = build_mapping_from_profile_csv(profile_csv)
    # 4) bangun mapping dari score_df jika ada
    if score_df is not None:
        roles_order = ["kuat", "mampu", "kurang_optimal"]
        for i, lbl in enumerate(score_df.index.astype(int)):
            cluster_to_role[int(lbl)] = roles_order[i] if i < len(roles_order) else "mampu"
    else:
        # fallback default
        print("Warning: cluster strength scores / profile tidak ditemukan. Mapping default dipakai: 0->kuat,1->mampu,2->kurang_optimal")
        cluster_to_role = {0: "kuat", 1: "mampu", 2: "kurang_optimal"}


# -------------------------
# Muat rencana latihan
# -------------------------
def load_training_plan(role):
    """Muat file plan text dari folder Training_Plans (role.txt)."""
    path = os.path.join("Training_Plans", f"{role}.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None
    return None


# -------------------------
# Fungsi publik: predict_from_list
# -------------------------
def predict_from_list(values, check_distance_to_center=False, distance_threshold=None):
    """
    values: list 6 fitur sesuai FEATURE_COLS
    check_distance_to_center: jika True, tambahkan jarak ke centroid terdekat
    distance_threshold: jika diberikan, juga beri flag apakah jauh dari centroid
    Return:
      dict berisi cluster, role, training_plan (string) atau error
    """
    # validasi panjang input
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != len(FEATURE_COLS):
        return {"error": "Masukkan 6 nilai: " + ", ".join(FEATURE_COLS)}

    if scaler is None or selected_model is None:
        return {"error": "Model/scaler tidak ditemukan. Jalankan run_pipeline.py untuk melatih & menyimpan model."}

    # buat DataFrame kecil dari input (untuk urutan kolom yang benar)
    df_input = pd.DataFrame([dict(zip(FEATURE_COLS, values))], columns=FEATURE_COLS)

    # lakukan scaling - gunakan .values untuk menghindari warning nama feature
    try:
        x_scaled = scaler.transform(df_input.values)[0]
    except Exception as e:
        return {"error": f"Scaling gagal: {e}"}

    # prediksi cluster dan temukan centroid/means bila mungkin
    try:
        # KMeans & GMM sama-sama punya predict pada instance; GMM juga punya means_
        cluster = int(selected_model.predict([x_scaled])[0])
        # ambil centers: KMeans.cluster_centers_ atau GMM.means_
        centers = None
        if hasattr(selected_model, "cluster_centers_"):
            centers = np.array(getattr(selected_model, "cluster_centers_"))
        elif hasattr(selected_model, "means_"):
            centers = np.array(getattr(selected_model, "means_"))
        elif hasattr(selected_model, "means"):
            centers = np.array(getattr(selected_model, "means"))
        else:
            centers = None
    except Exception as e:
        return {"error": f"Prediksi model gagal: {e}"}

    role = cluster_to_role.get(cluster, "mampu")
    plan_text = load_training_plan(role)

    result = {"model_method": selected_method, "cluster": int(cluster), "role": role, "training_plan": plan_text}

    if check_distance_to_center:
        if centers is not None and getattr(centers, "size", 0) > 0:
            dists = np.linalg.norm(centers - x_scaled, axis=1)
            nearest = float(np.min(dists))
            result["dist_to_nearest_center"] = round(nearest, 6)
            if distance_threshold is not None:
                result["is_far_from_centroid"] = nearest > distance_threshold
        else:
            result["dist_to_nearest_center"] = None

    return result


# Jika file ini dijalankan sebagai script, contoh panggilan cepat
if __name__ == "__main__":
    example = [60.0, 4.6, 75, 36.0, 1.75, 55] # ini contoh input jika file code ini dijalankan langsung
    print("Contoh input:", example)
    print("Hasil prediksi:", predict_from_list(example, check_distance_to_center=True, distance_threshold=3.0))
