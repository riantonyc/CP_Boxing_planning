"""


File tunggal untuk:
- Evaluasi metrics clustering K-Means
- Audit DBSCAN (density check & outlier)
- Menampilkan Silhouette, DBI, CHI, Inertia
- Stability testing (random seed)
- Output siap untuk BAB 3.4 Testing
"""

import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ============================================================
# KONFIGURASI
# ============================================================

DATASET_PATH = "../DatasetFinal/boxing_dataset_extended_10k.csv"
RESULTS_DIR = "results"

N_CLUSTERS = 3
RANDOM_STATE = 42

DBSCAN_EPS_LIST = [0.25, 0.30, 0.35, 0.40]
DBSCAN_MIN_SAMPLES = 5

STABILITY_SEEDS = [10, 20, 30]

FEATURES = [
    "VO2_Max",
    "Sprint30m_sec",
    "PushUp_MaxReps",
    "CMJ_Height_cm",
    "Punch36_Speed",
    "Resting_HR_bpm"
]

# ============================================================
# UTILITIES
# ============================================================

def evaluate_metrics(X, labels, model=None):
    """Hitung metrik clustering internal"""
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return None

    metrics = {
        "silhouette_score": silhouette_score(X, labels),
        "davies_bouldin_index": davies_bouldin_score(X, labels),
        "calinski_harabasz_index": calinski_harabasz_score(X, labels),
        "inertia": model.inertia_ if model is not None and hasattr(model, "inertia_") else None
    }
    return metrics


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n=== CLUSTERING TESTING PIPELINE (FINAL) ===\n")

    # --------------------------------------------------------
    # 0. Persiapan Folder Output
    # --------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load & Cleaning Dataset
    # --------------------------------------------------------
    df = pd.read_csv(DATASET_PATH)

    # Ambil hanya fitur yang digunakan
    df = df[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()

    X = df.values

    print(f"Dataset ready: {X.shape[0]} rows × {X.shape[1]} features")

    # --------------------------------------------------------
    # 2. Normalisasi Min-Max
    # --------------------------------------------------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print("Preprocessing: Min-Max Scaling ✓")

    # --------------------------------------------------------
    # 3. K-MEANS TRAINING
    # --------------------------------------------------------
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        init="k-means++",
        n_init=20,
        random_state=RANDOM_STATE
    )
    kmeans_labels = kmeans.fit_predict(X_scaled)

    print("\n--- K-MEANS CLUSTER DISTRIBUTION ---")
    for c, cnt in zip(*np.unique(kmeans_labels, return_counts=True)):
        print(f"Cluster {c}: {cnt} data")

    # --------------------------------------------------------
    # 4. K-MEANS METRICS
    # --------------------------------------------------------
    kmeans_metrics = evaluate_metrics(X_scaled, kmeans_labels, model=kmeans)

    print("\nK-MEANS METRICS:")
    for k, v in kmeans_metrics.items():
        print(f"{k:28s}: {v:.4f}" if v is not None else f"{k:28s}: None")

    # --------------------------------------------------------
    # 5. DBSCAN AUDIT (GRID SEARCH)
    # --------------------------------------------------------
    print("\n--- DBSCAN AUDIT ---")

    dbscan_results = []

    for eps in DBSCAN_EPS_LIST:
        dbscan = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES)
        labels = dbscan.fit_predict(X_scaled)

        n_noise = np.sum(labels == -1)
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]

        n_clusters = len(set(valid_labels))

        print(f"eps={eps:.2f} → clusters={n_clusters}, noise={n_noise}")

        if n_clusters >= 2 and len(valid_labels) >= 10:
            metrics = evaluate_metrics(X_scaled[valid_mask], valid_labels)
            metrics.update({
                "eps": eps,
                "clusters": n_clusters,
                "noise": int(n_noise)
            })
            dbscan_results.append(metrics)

    # Simpan audit DBSCAN
    if dbscan_results:
        pd.DataFrame(dbscan_results).to_csv(
            os.path.join(RESULTS_DIR, "dbscan_audit_metrics.csv"),
            index=False
        )
        print("DBSCAN audit saved ✓")
    else:
        print("DBSCAN tidak menghasilkan cluster valid")

    # --------------------------------------------------------
    # 6. STABILITY TESTING
    # --------------------------------------------------------
    print("\n--- K-MEANS STABILITY TEST ---")

    stability_results = []

    for seed in STABILITY_SEEDS:
        km = KMeans(
            n_clusters=N_CLUSTERS,
            n_init=10,
            random_state=seed
        )
        lbl = km.fit_predict(X_scaled)
        s = silhouette_score(X_scaled, lbl)

        stability_results.append({
            "random_seed": seed,
            "silhouette_score": s
        })

        print(f"seed={seed} → silhouette={s:.4f}")

    pd.DataFrame(stability_results).to_csv(
        os.path.join(RESULTS_DIR, "kmeans_stability_test.csv"),
        index=False
    )

    # --------------------------------------------------------
    # 7. SIMPAN METRICS FINAL
    # --------------------------------------------------------
    final_metrics = {
        "model": "KMeans (init=k-means++)",
        **kmeans_metrics
    }

    pd.DataFrame([final_metrics]).to_csv(
        os.path.join(RESULTS_DIR, "clustering_metrics_summary.csv"),
        index=False
    )

    print("\nFinal metrics saved ✓")
    print("\n=== TESTING SELESAI ===\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
