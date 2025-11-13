#!/bin/bash
# ==========================================================
# download_nba_data.sh (Simplified, Reliable Version)
# Downloads the NBA shots dataset, extracts CSVs,
# converts them to a single Parquet file, and
# performs cleanup. Idempotent on repeated runs.
# ==========================================================

set -euo pipefail

# --- Setup ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
ZIP_FILE="$DATA_DIR/nba-shots.zip"
PARQUET_FILE="$DATA_DIR/shots.parquet"
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/techbaron13/nba-shots-dataset-2001-present"

mkdir -p "$DATA_DIR"

# ==========================================================
# 1. If parquet exists → skip everything
# ==========================================================
if [[ -f "$PARQUET_FILE" ]]; then
    echo "[INFO] Parquet already exists at $PARQUET_FILE"
    echo "[INFO] Skipping download and CSV processing."
    exit 0
fi

# ==========================================================
# 2. Download ZIP using Kaggle API token
# ==========================================================
echo "[INFO] Downloading dataset from Kaggle..."
curl -L -o "$ZIP_FILE" \
     --header "Authorization: Bearer $(jq -r '.key' ~/.kaggle/kaggle.json)" \
     "$DATASET_URL"

# ==========================================================
# 3. Extract ZIP
# ==========================================================
echo "[INFO] Extracting ZIP..."
unzip -o "$ZIP_FILE" -d "$DATA_DIR" >/dev/null

# ==========================================================
# 4. Flatten CSV structure (works on all shells)
# ==========================================================
echo "[INFO] Flattening CSV files..."
find "$DATA_DIR" -type f -name "*.csv" -exec mv {} "$DATA_DIR" \;

# ==========================================================
# 5. Convert all CSVs into ONE Parquet file
# ==========================================================
echo "[INFO] Creating Parquet file..."

python3 - << EOF
import pandas as pd
import pathlib

DATA_DIR = pathlib.Path("$DATA_DIR")
PARQUET_FILE = DATA_DIR / "shots.parquet"

csv_files = sorted(DATA_DIR.glob("*.csv"))
print(f"Found {len(csv_files)} CSVs")

if not csv_files:
    raise RuntimeError("No CSV files found in data directory.")

dfs = []
for f in csv_files:
    print(f"[Extracting] Reading {f.name}")
    dfs.append(pd.read_csv(f, low_memory=False))

df = pd.concat(dfs, ignore_index=True)
df.to_parquet(PARQUET_FILE, index=False)

print(f"[PY] Saved Parquet to: {PARQUET_FILE}")
EOF

# ==========================================================
# 6. Cleanup
# ==========================================================
echo "[INFO] Cleaning up CSVs and ZIP..."
rm -f "$ZIP_FILE"
rm -f "$DATA_DIR"/*.csv

echo "[DONE] Dataset is ready at:"
echo "       $PARQUET_FILE"
