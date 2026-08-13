"""
Load INbreast Dataset from .xls file with Patient ID

This script shows how to load INbreast using the .xls file which contains
patient ID information for proper breast-level grouping.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the .xls file
xls_path = "INbreast.xls"

print("="*80)
print("LOADING INBREAST.XLS")
print("="*80)

# Read the .xls file
df = pd.read_excel(xls_path)

print(f"\nLoaded {len(df)} rows")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# Check for patient ID column
patient_id_cols = [col for col in df.columns if 'patient' in col.lower() or 'id' in col.lower()]
print(f"\nPotential patient ID columns: {patient_id_cols}")

# Show unique values for key columns
print("\n" + "="*80)
print("COLUMN ANALYSIS")
print("="*80)

for col in df.columns:
    n_unique = df[col].nunique()
    print(f"\n{col}:")
    print(f"  Unique values: {n_unique}")
    if n_unique <= 20:
        print(f"  Values: {df[col].unique()[:10]}")
    print(f"  Sample: {df[col].iloc[0]}")

# Analyze breast grouping structure
print("\n" + "="*80)
print("BREAST GROUPING STRUCTURE")
print("="*80)

# Try to identify patient/breast grouping
if 'Patient_ID' in df.columns or 'PatientID' in df.columns:
    patient_col = 'Patient_ID' if 'Patient_ID' in df.columns else 'PatientID'
    print(f"\nFound patient ID column: {patient_col}")
    print(f"Number of unique patients: {df[patient_col].nunique()}")

    # Count images per patient
    images_per_patient = df.groupby(patient_col).size()
    print(f"\nImages per patient distribution:")
    print(images_per_patient.value_counts().sort_index())

# Check for laterality information
laterality_cols = [col for col in df.columns if 'lat' in col.lower() or 'side' in col.lower()]
print(f"\nLaterality columns: {laterality_cols}")

# Check for view information
view_cols = [col for col in df.columns if 'view' in col.lower()]
print(f"View columns: {view_cols}")

print("\n" + "="*80)
