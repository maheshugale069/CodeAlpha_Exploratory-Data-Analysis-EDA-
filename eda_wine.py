
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr

wr.filterwarnings('ignore')

# ---------------------------
# Configuration
# ---------------------------
CSV_FILE = "WineQT.csv"   # Put WineQT.csv in the SAME folder as this script
OUTPUT_DIR = "outputs"    # All reports/plots will be saved here

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Couldn't find {CSV_FILE}. Place it next to this script and try again.")

df = pd.read_csv(CSV_FILE)

# ---------------------------
# Quick look
# ---------------------------
print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== INFO =====")
# info() prints directly; capture to string via buffer if needed, but printing is fine:
df.info()

print("\n===== DESCRIBE (all dtypes) =====")
try:
    desc = df.describe(include='all', datetime_is_numeric=True).T
except TypeError:
    # For older pandas without datetime_is_numeric
    desc = df.describe(include='all').T
print(desc)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== MISSING VALUES (count) =====")
print(df.isnull().sum())

print("\n===== UNIQUE VALUES PER COLUMN =====")
print(df.nunique())

# ---------------------------
# Save tabular reports
# ---------------------------
missing = df.isna().sum().to_frame('missing_count')
missing['missing_pct'] = (missing['missing_count'] / len(df)) * 100
missing.to_csv(os.path.join(OUTPUT_DIR, "missing_report.csv"))

nunique = df.nunique().to_frame('unique_count')
nunique.to_csv(os.path.join(OUTPUT_DIR, "unique_counts.csv"))

dtypes = df.dtypes.astype(str).to_frame('dtype')
dtypes.to_csv(os.path.join(OUTPUT_DIR, "dtypes.csv"))

desc.to_csv(os.path.join(OUTPUT_DIR, "summary_describe.csv"))

dup_count = df.duplicated().sum()
with open(os.path.join(OUTPUT_DIR, "duplicates.txt"), "w", encoding="utf-8") as f:
    f.write(f"Duplicate rows: {dup_count}\n")

# ---------------------------
# Visualizations
# ---------------------------
numeric_cols = df.select_dtypes(include=[np.number])
if numeric_cols.shape[1] > 0:
    # Histograms
    ax_array = numeric_cols.hist(figsize=(14, 10), bins=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histograms.png"), dpi=150)
    plt.show()

    # Correlation heatmap
    corr = numeric_cols.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "corr_heatmap.png"), dpi=150)
    plt.show()

    # Boxplot (horizontal for readability)
    plt.figure(figsize=(12, 8))
    try:
        sns.boxplot(data=numeric_cols, orient='h')
    except Exception:
        # Fallback to pandas if seaborn has issues
        numeric_cols.plot(kind='box', vert=False, figsize=(12, 8))
    plt.title("Outlier Detection (Boxplot)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot.png"), dpi=150)
    plt.show()
else:
    print("\n(No numeric columns found; skipping numeric plots.)")

print(f"\nAll reports/plots saved to: {os.path.abspath(OUTPUT_DIR)}")
