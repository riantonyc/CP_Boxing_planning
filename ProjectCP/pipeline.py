# pipeline.py
"""
Pipeline with automated model selection among:
1) KMeans (k-means++ n_init=50)
2) KMeans initialized from DBSCAN centroids
3) KMeans trained on augmented (oversampled) data
4) Gaussian Mixture Model (GMM)

Metrics: silhouette_score (higher better) & davies_bouldin_score (lower better).
Selection rule: highest silhouette; tiebreaker lowest DB index.

Saves:
- selected model -> models/selected_model.pkl
- metadata -> models/selected_model_meta.json
- model comparison CSV -> results/model_comparison.csv
- scaler/encoder -> models/
- training outputs -> results/
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

# default dirs
DEFAULT_MODELS_DIR = "models"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_PLANS_DIR = "Training_Plans"


# -------------------------
# PREPROCESSING HELPERS
# -------------------------
def ensure_dirs(*dirs: str):
    """Pastikan direktori ada (jika belum ada, buat)."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def fit_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[StandardScaler, object, pd.DataFrame]:
    """
    Fit scaler & encoder, kembalikan (scaler, encoder_or_None, df_scaled)
    - Konversi kolom ke numeric jika memungkinkan (coerce -> NaN)
    - Scaling numeric via StandardScaler
    - (opsional) encode categorical via OrdinalEncoder (tidak digunakan di proyek ini)
    """
    df_use = df.loc[:, feature_cols].copy()

    # Coerce semua feature menjadi numeric jika mungkin
    for c in df_use.columns:
        df_use[c] = pd.to_numeric(df_use[c], errors="coerce")

    # identifikasi kolom numeric (setelah coercion)
    num_cols = df_use.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = []  # tidak diharapkan ada categorical untuk fitur ini

    scaler = StandardScaler()
    if len(num_cols) > 0:
        scaled = scaler.fit_transform(df_use[num_cols].values)
        # assign kembali sebagai DataFrame agar dtype tidak bermasalah
        df_use.loc[:, num_cols] = pd.DataFrame(scaled, columns=num_cols, index=df_use.index)

    encoder = None
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder()
        df_use.loc[:, cat_cols] = encoder.fit_transform(df_use[cat_cols])

    if len(num_cols) > 0 and df_use[num_cols].isna().any().any():
        print("WARNING: NaN present in numeric features after conversion/scale. Consider cleaning input data.")

    return scaler, encoder, df_use


# -------------------------
# DBSCAN HELPERS (preprocessing)
# -------------------------
def grid_search_dbscan(X: np.ndarray, eps_vals, min_samples_vals) -> pd.DataFrame:
    """
    Grid-search sederhana untuk DBSCAN; kembalikan DataFrame hasil.
    Kolom: eps, min_samples, n_clusters, n_noise, noise_ratio, silhouette (hanya non-noise).
    """
    rows = []
    n_total = X.shape[0]
    for eps in eps_vals:
        for ms in min_samples_vals:
            db = DBSCAN(eps=eps, min_samples=ms).fit(X)
            labels = db.labels_
            n_noise = int((labels == -1).sum())
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            sil = None
            if n_clusters >= 2:
                mask = labels != -1
                try:
                    sil = silhouette_score(X[mask], labels[mask])
                except Exception:
                    sil = None
            rows.append({
                "eps": float(eps), "min_samples": int(ms),
                "n_clusters": int(n_clusters), "n_noise": int(n_noise),
                "noise_ratio": float(n_noise / n_total),
                "silhouette": float(sil) if sil is not None else None
            })
    return pd.DataFrame(rows)


def apply_dbscan_and_get_centroids(X: np.ndarray, eps: float, min_samples: int):
    """
    Fit DBSCAN dan hitung centroid untuk cluster non-noise.
    Return: (db_obj, labels, centroids ndarray (c x dim), mask_non_noise (bool array))
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    mask_non_noise = labels != -1
    unique_labels = sorted([lab for lab in set(labels) if lab != -1])
    centroids = []
    for lab in unique_labels:
        centroids.append(np.mean(X[labels == lab], axis=0))
    centroids_arr = np.vstack(centroids) if len(centroids) > 0 else np.empty((0, X.shape[1]))
    return db, labels, centroids_arr, mask_non_noise


# -------------------------
# centroid reduce/expand
# -------------------------
def reduce_or_expand_centroids(centroids_dbscan: np.ndarray, X_clean: np.ndarray, k: int):
    """
    Pastikan jumlah centroid awal sama dengan k:
    - m == k -> return copy
    - m > k  -> cluster centroid menjadi k center (KMeans kecil)
    - m < k  -> tambah center dari titik terjauh (farthest-point) sampai k
    """
    m = centroids_dbscan.shape[0]
    if m == 0:
        return None
    if m == k:
        return centroids_dbscan.copy()
    if m > k:
        kmc = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmc.fit(centroids_dbscan)
        return kmc.cluster_centers_
    # m < k
    centers = list(centroids_dbscan.copy())
    needed = k - m
    X_remaining = X_clean.copy()
    for _ in range(needed):
        if X_remaining.shape[0] == 0:
            break
        dists = np.min(np.linalg.norm(X_remaining[:, None, :] - np.vstack(centers)[None, :, :], axis=2), axis=1)
        idx = int(np.argmax(dists))
        centers.append(X_remaining[idx])
        X_remaining = np.delete(X_remaining, idx, axis=0)
    return np.vstack(centers)[:k]


# -------------------------
# MODEL TRYERS
# -------------------------
def try_kmeans_pp(X_train: np.ndarray, k=3) -> Dict:
    """KMeans dengan init k-means++ dan n_init=50."""
    km = KMeans(n_clusters=k, init='k-means++', n_init=50, random_state=42).fit(X_train)
    labels = km.labels_
    sil = silhouette_score(X_train, labels) if len(set(labels)) > 1 else -1.0
    dbi = davies_bouldin_score(X_train, labels) if len(set(labels)) > 1 else float("inf")
    return {"model": km, "labels": labels, "silhouette": sil, "dbi": dbi, "method": "kmeans_pp"}


def try_kmeans_dbscan_init(X_train: np.ndarray, centroids_dbscan: np.ndarray, k=3) -> Dict:
    """KMeans yang di-inisialisasi dari centroid DBSCAN (reduce/expand agar k)."""
    init_centers = reduce_or_expand_centroids(centroids_dbscan, X_train, k)
    if init_centers is None:
        return try_kmeans_pp(X_train, k=k)
    km = KMeans(n_clusters=k, init=init_centers, n_init=10, random_state=42).fit(X_train)
    labels = km.labels_
    sil = silhouette_score(X_train, labels) if len(set(labels)) > 1 else -1.0
    dbi = davies_bouldin_score(X_train, labels) if len(set(labels)) > 1 else float("inf")
    return {"model": km, "labels": labels, "silhouette": sil, "dbi": dbi, "method": "kmeans_dbscan_init"}


def try_oversample_then_kmeans(X_train: np.ndarray, initial_labels=None, k=3) -> Dict:
    """
    Oversampling sintetis (Gaussian kecil di sekitar center) untuk menyeimbangkan ukuran cluster,
    lalu latih KMeans pada data augmentasi.
    """
    if initial_labels is None:
        base_km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42).fit(X_train)
        labels = base_km.labels_
    else:
        labels = initial_labels

    if len(labels) != X_train.shape[0]:
        labels = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42).fit_predict(X_train)

    counts = np.bincount(labels, minlength=k)
    target = counts.max()
    centers = np.array([X_train[labels == i].mean(axis=0) if (labels == i).sum() > 0 else X_train[np.random.choice(len(X_train))] for i in range(k)])
    X_aug = X_train.copy()
    for i, cnt in enumerate(counts):
        if cnt < target:
            need = int(target - cnt)
            if cnt > 1:
                sigma = np.std(X_train[labels == i], axis=0) * 0.05
                sigma[sigma == 0] = 0.01
            else:
                sigma = np.std(X_train, axis=0) * 0.05
            synth = centers[i] + np.random.normal(scale=sigma, size=(need, X_train.shape[1]))
            X_aug = np.vstack([X_aug, synth])
    km = KMeans(n_clusters=k, init='k-means++', n_init=50, random_state=42).fit(X_aug)
    labels_final = km.predict(X_train)
    sil = silhouette_score(X_train, labels_final) if len(set(labels_final)) > 1 else -1.0
    dbi = davies_bouldin_score(X_train, labels_final) if len(set(labels_final)) > 1 else float("inf")
    return {"model": km, "labels": labels_final, "silhouette": sil, "dbi": dbi, "method": "kmeans_oversample"}


def try_gmm(X_train: np.ndarray, k=3) -> Dict:
    """Gaussian Mixture Model (GMM)."""
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42).fit(X_train)
    labels = gmm.predict(X_train)
    sil = silhouette_score(X_train, labels) if len(set(labels)) > 1 else -1.0
    dbi = davies_bouldin_score(X_train, labels) if len(set(labels)) > 1 else float("inf")
    return {"model": gmm, "labels": labels, "silhouette": sil, "dbi": dbi, "method": "gmm"}


# -------------------------
# MODEL SELECTION ORCHESTRATION
# -------------------------
def run_full_pipeline_with_selection(
    df: pd.DataFrame,
    feature_cols: List[str],
    out_dir=".",
    k=3,
    eps_vals=None,
    min_samples_vals=None,
    save_dbscan_audit=True
) -> Dict:
    """
    Alur end-to-end:
    1) fit preprocessor (scaler)
    2) grid-search DBSCAN -> pilih eps/min_samples
    3) apply DBSCAN -> dapatkan centroid & mask_non_noise (clean set)
    4) coba 4 strategi model (kmeans_pp, kmeans_dbscan_init, oversample_kmeans, gmm)
    5) pilih model terbaik berdasarkan silhouette (utama) & davies_bouldin (penurun)
    6) simpan model terpilih + metadata
    7) beri label pada semua baris (predict pada full data yang sudah discale)
    8) simpan profile, skor kekuatan, dan rencana latihan (week1 Mon-Fri)
    """
    ensure_dirs(out_dir, DEFAULT_MODELS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_PLANS_DIR)
    models_dir = os.path.join(out_dir, DEFAULT_MODELS_DIR)
    results_dir = os.path.join(out_dir, DEFAULT_RESULTS_DIR)
    plans_dir = os.path.join(out_dir, DEFAULT_PLANS_DIR)

    # 1) Preprocess
    scaler, encoder, df_scaled = fit_preprocessor(df, feature_cols)
    X = df_scaled.values

    # 2) DBSCAN grid search (default range)
    if eps_vals is None:
        eps_vals = np.linspace(0.2, 1.2, 11)
    if min_samples_vals is None:
        min_samples_vals = [3, 4, 5]

    grid = grid_search_dbscan(X, eps_vals, min_samples_vals)
    try:
        grid.to_csv(os.path.join(results_dir, "dbscan_grid_search.csv"), index=False)
    except Exception:
        pass

    # Pilih baris terbaik heuristik: prefer highest silhouette (lebih stabil), tie-breaker rendah noise_ratio
    if grid["silhouette"].notna().any():
        best_row = grid.dropna(subset=["silhouette"]).sort_values(["silhouette", "noise_ratio"], ascending=[False, True]).iloc[0]
    else:
        best_row = grid.sort_values("noise_ratio", ascending=True).iloc[0]
    best_eps = float(best_row["eps"])
    best_ms = int(best_row["min_samples"])

    # 3) apply DBSCAN
    db, db_labels, centroids_dbscan, mask_non_noise = apply_dbscan_and_get_centroids(X, eps=best_eps, min_samples=best_ms)
    if save_dbscan_audit:
        try:
            joblib.dump(db, os.path.join(models_dir, "dbscan_for_audit.pkl"))
        except Exception:
            pass

    X_clean = X[mask_non_noise]
    n_clean = int(mask_non_noise.sum())

    # jika data "clean" terlalu kecil, fallback gunakan seluruh X untuk training model
    if X_clean.shape[0] < max(5, k):
        print("Warning: cleaned set small; using full dataset for model trials.")
        X_train_for_models = X.copy()
        mask_used = np.ones(X.shape[0], dtype=bool)
    else:
        X_train_for_models = X_clean.copy()
        mask_used = mask_non_noise.copy()

    # 4) Try models
    results = []
    res1 = try_kmeans_pp(X_train_for_models, k=k); results.append(res1)
    res2 = try_kmeans_dbscan_init(X_train_for_models, centroids_dbscan, k=k); results.append(res2)
    init_labels = res1["labels"] if res1 is not None else None
    res3 = try_oversample_then_kmeans(X_train_for_models, initial_labels=init_labels, k=k); results.append(res3)
    res4 = try_gmm(X_train_for_models, k=k); results.append(res4)

    # 5) Compare & pilih terbaik (silhouette desc, davies_bouldin asc)
    comp_rows = []
    for r in results:
        comp_rows.append({
            "method": r["method"],
            "silhouette": float(r["silhouette"]) if r["silhouette"] is not None else None,
            "davies_bouldin": float(r["dbi"]) if r["dbi"] is not None else None
        })
    comp_df = pd.DataFrame(comp_rows).sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
    try:
        comp_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    except Exception:
        pass

    best_method = comp_df.iloc[0]["method"]
    best_res = next(r for r in results if r["method"] == best_method)
    selected_model = best_res["model"]

    # 6) Save selected model + meta
    ensure_dirs(models_dir)
    try:
        joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    except Exception:
        pass
    if encoder is not None:
        try:
            joblib.dump(encoder, os.path.join(models_dir, "encoder.pkl"))
        except Exception:
            pass

    try:
        joblib.dump(selected_model, os.path.join(models_dir, "selected_model.pkl"))
    except Exception:
        pass

    meta = {
        "selected_method": best_method,
        "k": int(k),
        "dbscan_params": {"eps": best_eps, "min_samples": best_ms, "dbscan_clusters": int(centroids_dbscan.shape[0])},
        "n_raw": int(X.shape[0]),
        "n_clean": int(n_clean),
        "metrics": comp_rows
    }

    # 7) Produce labels for ALL rows using selected model (predict pada full scaled X)
    try:
        all_labels = selected_model.predict(X)
    except Exception:
        # fallback: semua ke 0
        all_labels = np.zeros(X.shape[0], dtype=int)

    df_out = df.copy()
    df_out["kmeans_label"] = all_labels.astype(int)

    # 8) Profiles, scores, training plan saving
    try:
        profile = df_out.groupby("kmeans_label")[feature_cols].mean().round(3)
        profile.to_csv(os.path.join(results_dir, "profile_kmeans.csv"))
    except Exception:
        profile = pd.DataFrame()

    # fungsi skor kekuatan (rule sederhana)
    def strength_score_row(r: pd.Series) -> float:
        # formula: VO2 tinggi (+), sprint time rendah (-), pushup tinggi (+),
        # CMJ tinggi (+), punch speed tinggi (+), resting HR rendah (-)
        return (
            float(r.get("VO2_Max", 0.0))
            - float(r.get("Sprint30m_sec", 0.0))
            + float(r.get("PushUp_MaxReps", 0.0))
            + float(r.get("CMJ_Height_cm", 0.0))
            + float(r.get("Punch36_Speed", 0.0))
            - float(r.get("Resting_HR_bpm", 0.0))
        )

    scores = {}
    if not profile.empty:
        for idx in profile.index:
            scores[int(idx)] = float(strength_score_row(profile.loc[idx]))
    else:
        # fallback: hitung dari df_out mean per label (jika profile gagal)
        grouped = df_out.groupby("kmeans_label")[feature_cols].mean().round(3)
        for idx in grouped.index:
            scores[int(idx)] = float(strength_score_row(grouped.loc[idx]))

    score_df = pd.DataFrame.from_dict(scores, orient="index", columns=["strength_score"])
    if not score_df.empty:
        score_df = score_df.sort_values("strength_score", ascending=False)
        score_df["rank"] = range(1, len(score_df) + 1)
    try:
        score_df.to_csv(os.path.join(results_dir, "cluster_strength_scores.csv"))
    except Exception:
        pass

    # map rank -> role (1 strongest => 'kuat', 2 => 'mampu', 3 => 'kurang_optimal')
    rank_to_role = {1: "kuat", 2: "mampu", 3: "kurang_optimal"}
    cluster_to_role = {}
    if not score_df.empty:
        for _, row in score_df.reset_index().iterrows():
            cluster_label = int(row["index"])
            rank = int(row["rank"])
            cluster_to_role[cluster_label] = rank_to_role.get(rank, "mampu")
    else:
        # default mapping jika score_df kosong
        unique_labels = sorted(list(set(all_labels.tolist())))
        for i, lbl in enumerate(unique_labels):
            cluster_to_role[int(lbl)] = rank_to_role.get(i + 1, "mampu")

    # tambahkan cluster_to_role ke meta
    meta["cluster_to_role"] = cluster_to_role

    # 9) Write training plans (week1 Mon-Fri) to files (overwrite)
    # setiap hari minimal 7 item (warmup, drills, utama, accessory, core, conditioning, cooldown)
    plans = {
        "kuat": {
            "week1": [
                {"day": "Senin",  "focus": "Speed & Power", "items": [
                    {"name": "Warm-up (mobilitas + jogging ringan)", "duration_min": 8},
                    {"name": "Dynamic drills (leg swings, arm circles)", "reps": ""},
                    {"name": "Sprint 30m", "sets": 8, "reps": 1, "rest_s": 90},
                    {"name": "Plyometric box jump", "sets": 4, "reps": 6, "rest_s": 60},
                    {"name": "Explosive medicine ball throws", "sets": 4, "reps": 6},
                    {"name": "Short core circuit (plank/hollow)", "sets": 3, "duration_each_s": 45},
                    {"name": "Cool-down & stretching", "duration_min": 8}
                ]},
                {"day": "Selasa", "focus": "Padwork & Technique", "items": [
                    {"name": "Warm-up (skipping + shadowboxing)", "duration_min": 8},
                    {"name": "Footwork ladder / cones", "sets": 4},
                    {"name": "Padwork high intensity", "rounds": 6, "round_duration_min": 3},
                    {"name": "Combination drills (power+speed)", "sets": 6, "reps": 8},
                    {"name": "Defensive drills (slips/rolls)", "duration_min": 12},
                    {"name": "Accessory upper body (band pull-aparts)", "sets": 3, "reps": 15},
                    {"name": "Mobility & breathing work", "duration_min": 6}
                ]},
                {"day": "Rabu",   "focus": "Strength (Explosive)", "items": [
                    {"name": "Warm-up + activation", "duration_min": 10},
                    {"name": "Kettlebell swing / Olympic variation", "sets": 4, "reps": 4},
                    {"name": "Trap bar / deadlift variation (power)", "sets": 4, "reps": 3, "rest_s": 120},
                    {"name": "Unilateral explosive (split jump)", "sets": 3, "reps": 6},
                    {"name": "Core heavy (weighted plank/hanging leg raise)", "sets": 3},
                    {"name": "Neck & upper-back stability", "sets": 3, "reps": 12},
                    {"name": "Cooldown & mobility", "duration_min": 8}
                ]},
                {"day": "Kamis",  "focus": "Conditioning & Skill", "items": [
                    {"name": "Warm-up (easy run + drills)", "duration_min": 10},
                    {"name": "Interval running (400/200m)", "sets": 6, "rest_s": 90},
                    {"name": "Heavy bag power rounds", "rounds": 6, "round_duration_min": 3},
                    {"name": "Speed endurance mitts/pad", "rounds": 4, "round_duration_min": 3},
                    {"name": "Core circuit (anti-rotation)", "sets": 3},
                    {"name": "Grip & forearm work", "sets": 3, "reps": 12},
                    {"name": "Stretching & recovery pulses", "duration_min": 10}
                ]},
                {"day": "Jumat",  "focus": "Recovery Active + Sparring IQ", "items": [
                    {"name": "Warm-up (mobility + light bike)", "duration_min": 10},
                    {"name": "Technical sparring (light, focus on timing)", "rounds": 6, "round_duration_min": 2},
                    {"name": "Padwork technical rounds", "rounds": 4, "round_duration_min": 3},
                    {"name": "Low-intensity steady state cardio", "duration_min": 20},
                    {"name": "Movement & footwork drills", "duration_min": 12},
                    {"name": "Breathing + relaxation drills", "duration_min": 8},
                    {"name": "Extended mobility & soft tissue work", "duration_min": 12}
                ]}
            ]
        },

        "mampu": {
            "week1": [
                {"day": "Senin",  "focus": "Speed Endurance", "items": [
                    {"name": "Warm-up (joint mobility)", "duration_min": 8},
                    {"name": "Activation (skipping/quick feet)", "duration_min": 5},
                    {"name": "Sprint 30m", "sets": 6, "reps": 1, "rest_s": 90},
                    {"name": "Agility ladder / cone drills", "sets": 4},
                    {"name": "Short plyo (tuck jumps)", "sets": 3, "reps": 8},
                    {"name": "Core dynamic (mountain climbers)", "sets": 3, "reps": 30},
                    {"name": "Cool-down & stretching", "duration_min": 8}
                ]},
                {"day": "Selasa", "focus": "Technical Padwork", "items": [
                    {"name": "Warm-up (shadowboxing)", "duration_min": 8},
                    {"name": "Footwork drills", "duration_min": 10},
                    {"name": "Padwork moderate", "rounds": 5, "round_duration_min": 3},
                    {"name": "Combination repetition (3-step combos)", "sets": 6},
                    {"name": "Speed drills (reaction)", "sets": 4},
                    {"name": "Accessory pull/push (rows/push-ups)", "sets": 3, "reps": 10},
                    {"name": "Mobility & breathing", "duration_min": 8}
                ]},
                {"day": "Rabu",   "focus": "Strength (General)", "items": [
                    {"name": "Warm-up", "duration_min": 8},
                    {"name": "Compound lifts (squat/bench/row)", "sets": 3, "reps": 8, "rest_s": 90},
                    {"name": "Romanian deadlift / hinge", "sets": 3, "reps": 6},
                    {"name": "Accessory core (anti-rotation)", "sets": 3},
                    {"name": "Posterior chain band work", "sets": 3, "reps": 12},
                    {"name": "Conditioning finisher (tabata)", "rounds": 4},
                    {"name": "Stretch & cool-down", "duration_min": 8}
                ]},
                {"day": "Kamis",  "focus": "Anaerobic Conditioning", "items": [
                    {"name": "Warm-up (dynamic)", "duration_min": 8},
                    {"name": "Interval runs (200m)", "sets": 8, "rest_s": 60},
                    {"name": "Heavy bag technique rounds", "rounds": 5, "round_duration_min": 3},
                    {"name": "Speed-endurance circuits (bodyweight)", "sets": 4},
                    {"name": "Jump rope endurance", "duration_min": 6},
                    {"name": "Core & hip stability", "sets": 3},
                    {"name": "Cooldown mobility", "duration_min": 8}
                ]},
                {"day": "Jumat",  "focus": "Technique & Recovery", "items": [
                    {"name": "Warm-up (easy shadow)", "duration_min": 8},
                    {"name": "Technical drills (defense/offense)", "duration_min": 12},
                    {"name": "Light sparring / partner drills", "rounds": 4, "round_duration_min": 2},
                    {"name": "Mobility flow", "duration_min": 15},
                    {"name": "Breathing & core activation", "duration_min": 8},
                    {"name": "Optional light aerobic", "duration_min": 15},
                    {"name": "Stretch & foam rolling", "duration_min": 10}
                ]}
            ]
        },

        "kurang_optimal": {
            "week1": [
                {"day": "Senin",  "focus": "Cardio dasar", "items": [
                    {"name": "Warm-up (walking + mobility)", "duration_min": 8},
                    {"name": "Brisk jog / bike", "duration_min": 25},
                    {"name": "Low-impact plyo (step-ups)", "sets": 3, "reps": 8},
                    {"name": "Bodyweight circuit (squat/push-up/rows)", "sets": 3, "reps": 10},
                    {"name": "Core basics (plank/side plank)", "sets": 3},
                    {"name": "Coordination drills (cones)", "duration_min": 10},
                    {"name": "Cooldown & stretching", "duration_min": 8}
                ]},
                {"day": "Selasa", "focus": "Teknik dasar", "items": [
                    {"name": "Warm-up (mobility)", "duration_min": 8},
                    {"name": "Shadowboxing (teknik dasar)", "rounds": 5, "round_duration_min": 2},
                    {"name": "Footwork basics", "duration_min": 15},
                    {"name": "Light padwork (focus combinations)", "rounds": 4, "round_duration_min": 2},
                    {"name": "Balance & coordination (single-leg)", "sets": 3, "reps": 8},
                    {"name": "Core activation", "sets": 3},
                    {"name": "Stretching", "duration_min": 10}
                ]},
                {"day": "Rabu",   "focus": "Kekuatan tubuh dasar", "items": [
                    {"name": "Warm-up (band work)", "duration_min": 8},
                    {"name": "Bodyweight squat / lunges", "sets": 3, "reps": 12},
                    {"name": "Push-up progressions", "sets": 3, "reps": 10},
                    {"name": "Inverted row / band row", "sets": 3, "reps": 8},
                    {"name": "Core basics (deadbug)", "sets": 3},
                    {"name": "Low-load conditioning (step-ups)", "sets": 3},
                    {"name": "Mobility cooldown", "duration_min": 8}
                ]},
                {"day": "Kamis",  "focus": "Kardio ringan + koordinasi", "items": [
                    {"name": "Warm-up (dynamic stretches)", "duration_min": 8},
                    {"name": "Interval walking/jog (30/60s)", "sets": 10},
                    {"name": "Coordination drills (agility cones)", "duration_min": 12},
                    {"name": "Low-impact plyo (box step)", "sets": 3, "reps": 8},
                    {"name": "Core stability", "sets": 3},
                    {"name": "Breathing practice", "duration_min": 6},
                    {"name": "Stretching & cooldown", "duration_min": 8}
                ]},
                {"day": "Jumat",  "focus": "Skill ringan & recovery", "items": [
                    {"name": "Warm-up (mobility + light shadow)", "duration_min": 8},
                    {"name": "Light padwork (technique)", "rounds": 4, "round_duration_min": 2},
                    {"name": "Footwork & balance drills", "duration_min": 12},
                    {"name": "Light core circuit", "sets": 3},
                    {"name": "Breathing & relaxation", "duration_min": 8},
                    {"name": "Optional easy aerobic", "duration_min": 15},
                    {"name": "Foam rolling & stretching", "duration_min": 10}
                ]}
            ]
        }
    }

    # Simpan plans ke folder Training_Plans sebagai file teks (stringified dict)
    ensure_dirs(plans_dir)
    for role, plan in plans.items():
        try:
            with open(os.path.join(plans_dir, f"{role}.txt"), "w", encoding="utf-8") as f:
                f.write(str(plan))
        except Exception as e:
            print(f"Warning: failed to write plan for {role}: {e}")

    # Simpan df_out (hasil clustering)
    try:
        df_out.to_csv(os.path.join(results_dir, "clustered_athletes_with_labels.csv"), index=False)
    except Exception as e:
        print(f"Warning: failed to save clustered_athletes_with_labels.csv: {e}")

    # simpan metadata lengkap (include cluster_to_role)
    meta["cluster_to_role"] = cluster_to_role
    try:
        with open(os.path.join(models_dir, "selected_model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    # return ringkasan hasil
    return {
        "models_dir": models_dir,
        "results_dir": results_dir,
        "plans_dir": plans_dir,
        "selected_method": best_method,
        "meta": meta,
        "comparison_csv": os.path.join(results_dir, "model_comparison.csv")
    }


# Jika dijalankan langsung (debug cepat)
if __name__ == "__main__":
    example_features = ["VO2_Max","Sprint30m_sec","PushUp_MaxReps","CMJ_Height_cm","Punch36_Speed","Resting_HR_bpm"]
    if os.path.exists("data.csv"):
        df_data = pd.read_csv("data.csv")
        out = run_full_pipeline_with_selection(df_data, example_features, out_dir=".")
        print("Pipeline finished. Meta:", out)
    else:
        print("No data.csv found. To run pipeline from script, place a CSV named 'data.csv' or call run_full_pipeline_with_selection programmatically.")
