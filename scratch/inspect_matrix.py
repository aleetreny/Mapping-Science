import pandas as pd
from pathlib import Path

matrix_path = Path("outputs/07_morphological_similarity/matrices/field_static_morphological_distance_matrix.csv")
if not matrix_path.exists():
    print(f"Matrix file not found at {matrix_path}")
else:
    df = pd.read_csv(matrix_path, index_col=0)
    print("Matrix Shape:", df.shape)
    print("Index (Row Names):", list(df.index))
    print("Columns:", list(df.columns))
