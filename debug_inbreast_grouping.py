"""
Debug script to check INbreast breast grouping issue.
"""

import pandas as pd
from pathlib import Path
from collections import Counter

# Read the INbreast CSV (update path as needed)
INBREAST_CSV = r"results/zeroshot_results/inbreast_with_patient_id.csv"

if not Path(INBREAST_CSV).exists():
    print(f"ERROR: File not found: {INBREAST_CSV}")
    print("Please update the path or run the notebook up to cell 14 first")
else:
    df = pd.read_csv(INBREAST_CSV)

    print("="*80)
    print("INBREAST DATAFRAME DIAGNOSIS")
    print("="*80)

    print(f"\nTotal rows: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")

    # Check Patient ID
    print(f"\n--- PATIENT ID CHECK ---")
    if 'Patient ID' in df.columns:
        patient_ids = df['Patient ID'].astype(str).str.strip()
        print(f"Unique Patient IDs: {patient_ids.nunique()}")
        print(f"Patient ID value counts:")
        print(patient_ids.value_counts().head(10))

        # Check for problematic values
        if patient_ids.nunique() < 10:
            print("\n⚠️ WARNING: Very few unique Patient IDs!")
            print("Unique values:", patient_ids.unique())
    else:
        print("❌ Patient ID column NOT FOUND")

    # Check Laterality
    print(f"\n--- LATERALITY CHECK ---")
    if 'Laterality' in df.columns:
        laterality = df['Laterality'].astype(str).str.strip().str.upper()
        print(f"Unique Lateralities: {laterality.nunique()}")
        print(f"Laterality value counts:")
        print(laterality.value_counts())

        # Check for problematic values
        if laterality.nunique() < 2:
            print("\n⚠️ WARNING: Only one laterality value!")
            print("Unique values:", laterality.unique())
    else:
        print("❌ Laterality column NOT FOUND")

    # Check View
    print(f"\n--- VIEW CHECK ---")
    if 'View' in df.columns:
        views = df['View'].astype(str).str.strip().str.upper()
        print(f"Unique Views: {views.nunique()}")
        print(f"View value counts:")
        print(views.value_counts())

    # Simulate breast grouping
    print(f"\n--- SIMULATED BREAST GROUPING ---")
    if 'Patient ID' in df.columns and 'Laterality' in df.columns:
        patient_ids = df['Patient ID'].astype(str).str.strip()
        laterality = df['Laterality'].astype(str).str.strip().str.upper()

        # Create breast ID
        df['breast_id'] = patient_ids + "_" + laterality

        print(f"\nUnique breast IDs: {df['breast_id'].nunique()}")

        # Count views per breast
        views_per_breast = df.groupby('breast_id').size()
        distribution = Counter(views_per_breast)

        print(f"\nViews per breast distribution:")
        for n_views in sorted(distribution.keys()):
            count = distribution[n_views]
            pct = count / len(views_per_breast) * 100
            print(f"  {n_views} view(s): {count:>3d} ({pct:>5.1f}%)")

        print(f"\nSample breast IDs (first 10):")
        print(df['breast_id'].head(10).tolist())

        # Check for most common breast IDs
        print(f"\nMost common breast IDs:")
        print(df['breast_id'].value_counts().head(10))

    # Show sample rows
    print(f"\n--- SAMPLE ROWS ---")
    cols_to_show = ['File Name', 'Patient ID', 'Laterality', 'View', 'Bi-Rads']
    cols_available = [c for c in cols_to_show if c in df.columns]
    print(df[cols_available].head(10).to_string())
