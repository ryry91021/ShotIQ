import pandas as pd
import pathlib
import os

def main():
    # Define paths
    BASE_DIR = pathlib.Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    PARQUET_FILE = DATA_DIR / "shots.parquet"

    print(f"Checking data directory: {DATA_DIR}")

    # 1. Check if Parquet already exists
    if PARQUET_FILE.exists():
        print(f"[SUCCESS] Parquet file already exists at: {PARQUET_FILE}")
        print("You can now run 'python src/main.py'")
        return

    # 2. Find CSV files (handling nested folders from the zip)
    if not DATA_DIR.exists():
        print(f"[ERROR] Data directory not found. Please create {DATA_DIR} and unzip the Kaggle data there.")
        return

    print("Searching for CSV files...")
    csv_files = list(DATA_DIR.rglob("*.csv")) # rglob searches recursively

    if not csv_files:
        print("[ERROR] No CSV files found.")
        print("1. Download the dataset: https://www.kaggle.com/datasets/techbaron13/nba-shots-dataset-2001-present")
        print("2. Unzip it into the 'data' folder.")
        return

    print(f"Found {len(csv_files)} CSV files. Combining into Parquet...")

    # 3. Combine CSVs into one DataFrame
    dfs = []
    for i, f in enumerate(csv_files):
        try:
            # Print progress every 100 files
            if i % 100 == 0:
                print(f"Processing file {i}/{len(csv_files)}...")
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    # 4. Save as Parquet
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        full_df.to_parquet(PARQUET_FILE, index=False)
        print("------------------------------------------------")
        print(f"[DONE] Successfully created: {PARQUET_FILE}")
        print("You can now run the main program.")
    else:
        print("[ERROR] Failed to combine CSVs.")

if __name__ == "__main__":
    main()