# run_pipeline.py
"""
Run pipeline and model selection.
"""
import os
import pandas as pd
from pipeline import run_full_pipeline_with_selection

DATA_CSV_PATH = "../DatasetFinal/boxing_dataset_extended_10k.csv"
OUT_DIR = "."

os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_CSV_PATH):
    # contoh data kecil
    sample = {
        "VO2_Max": [59.24, 57.65, 59.62, 61.81, 57.41, 57.41, 61.95, 59.92, 56.83, 59.36, 56.84, 56.84],
        "Sprint30m_sec": [4.79, 4.56, 5.03, 4.95, 4.96, 4.62, 5.08, 4.52, 4.91, 5.23, 4.60, 4.69],
        "PushUp_MaxReps": [72, 79, 74, 62, 82, 67, 79, 75, 72, 77, 69, 78],
        "CMJ_Height_cm": [35.93, 36.11, 36.38, 39.19, 35.49, 36.25, 37.73, 36.94, 35.28, 37.80, 34.25, 35.55],
        "Punch36_Speed": [1.65, 1.81, 1.82, 1.63, 1.79, 1.66, 1.66, 1.72, 1.73, 1.70, 1.57, 1.69],
        "Resting_HR_bpm": [55, 63, 63, 56, 57, 59, 59, 55, 62, 62, 58, 61]
    }
    df = pd.DataFrame(sample)
    df.to_csv(DATA_CSV_PATH, index=False)
else:
    df = pd.read_csv(DATA_CSV_PATH)

feature_cols = ["VO2_Max","Sprint30m_sec","PushUp_MaxReps","CMJ_Height_cm","Punch36_Speed","Resting_HR_bpm"]

meta = run_full_pipeline_with_selection(df, feature_cols, out_dir=OUT_DIR, k=3, save_dbscan_audit=True)
print("Pipeline finished. Meta:", meta)
print("Check results/model_comparison.csv and models/selected_model_meta.json")
