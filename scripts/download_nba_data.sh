#!/bin/bash
# ==========================================================
#  download_nba_data.sh
#  Downloads the NBA shots dataset via Kaggle API endpoint
#  using curl, and extracts it into ./data/
# ==========================================================

set -euo pipefail

# --- Setup ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
ZIP_FILE="$DATA_DIR/nba-shots-dataset.zip"
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/techbaron13/nba-shots-dataset-2001-present"

echo "[INFO] Ensuring data directory exists at: $DATA_DIR"
mkdir -p "$DATA_DIR"

# --- Download using Kaggle API token ---
# Requires ~/.kaggle/kaggle.json
echo "[INFO] Downloading dataset from Kaggle..."
curl -L -o "$ZIP_FILE" --header "Authorization: Bearer $(jq -r '.key' ~/.kaggle/kaggle.json)" "$DATASET_URL"

# --- Unzip ---
if [[ -f "$ZIP_FILE" ]]; then
  echo "[INFO] Extracting files..."
  unzip -o "$ZIP_FILE" -d "$DATA_DIR" >/dev/null
else
  echo "[ERROR] Download failed. No zip file found."
  exit 1
fi

# --- Move CSVs up from nested folders if needed ---
if compgen -G "$DATA_DIR/**/*.csv" > /dev/null; then
  echo "[INFO] Flattening CSV structure..."
  find "$DATA_DIR" -type f -name "*.csv" -exec mv {} "$DATA_DIR" \;
fi

# --- Cleanup ---
rm -f "$ZIP_FILE"
find "$DATA_DIR" -type d -empty -delete

echo "[DONE] All CSVs ready in $DATA_DIR/"
