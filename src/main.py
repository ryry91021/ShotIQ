from process_data import ShotDataLoader
from clean_data import ShotDataCleaner
from plots import CourtPlotter
from pathlib import Path
import time

def main():
    start_time = time.perf_counter()

    BASE_DIR = Path(__file__).parent

    # Load data
    loader = ShotDataLoader(base_dir=BASE_DIR, data_subdir="../data")
    df = loader.load_parquet(BASE_DIR / "../data/shots.parquet")
    if df is None:
        print("No data loaded. Exiting.")
        return

    # Clean data
    cleaner = ShotDataCleaner()
    cleaned_df = cleaner.clean(df)

    # Plot data for a specific player
    plotter = CourtPlotter()
    plotter.plot_shot_data(cleaned_df, player="Josh Hart")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()