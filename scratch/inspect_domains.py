import pandas as pd
from pathlib import Path

parquet_path = Path("outputs/04_reduced_metric_core/reduced_interpretable_core_metrics.parquet")
if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    print("Columns:", list(df.columns))
    # Print distinct combinations of domain and field
    mapping = df[["domain_display_name", "field_display_name"]].drop_duplicates().sort_values(["domain_display_name", "field_display_name"])
    print("\nDomain-to-Field Mapping:")
    for idx, row in mapping.iterrows():
        print(f"  Domain: {row['domain_display_name']} -> Field: {row['field_display_name']}")
else:
    print(f"File not found: {parquet_path}")
