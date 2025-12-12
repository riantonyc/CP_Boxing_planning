# test_input.py
"""
Simple harness to test inference on sample inputs.
Run: python test_input.py
"""
from predict_user import predict_from_list

samples = [
    [68.0, 4.35, 88, 39.0, 1.90, 56],   # kuat (expected)
    [54.0, 5.05, 52, 30.0, 1.50, 68],   # mampu (expected)
    [66.0, 4.90, 80, 30.0, 1.75, 69]   # kurang optimal (expected)
]

for s in samples:
    print("INPUT:", s)
    out = predict_from_list(s, check_distance_to_center=True, distance_threshold=None)
    print("OUTPUT:", out)
    print("-" * 60)
