from process_data import ShotDataLoader
from clean_data import ShotDataCleaner
# from plots import CourtPlotter  <-- Replaced with interactive version
from interactive_plots import InteractiveCourtPlotter
from model import ShotOutcomePredictor
from pathlib import Path
import time

def main():
    start_time = time.perf_counter()
    BASE_DIR = Path(__file__).parent

    # 1. Load Data
    loader = ShotDataLoader(base_dir=BASE_DIR, data_subdir="../data")
    # Ensure this path matches your local structure
    parquet_path = BASE_DIR / "../data/shots.parquet"
    
    if not parquet_path.exists():
        print(f"Error: {parquet_path} not found. Please run download_nba_data.sh first.")
        return

    df = loader.load_parquet(parquet_path)
    if df is None:
        print("No data loaded. Exiting.")
        return

    # 2. Clean Data
    cleaner = ShotDataCleaner()
    cleaned_df = cleaner.clean(df)

    # 3. Train Machine Learning Model [cite: 57]
    # We filter for a subset or train on all. Training on 4000+ files might be slow,
    # so for the "Progress Report" demo, we can train on the loaded dataframe.
    predictor = ShotOutcomePredictor()
    accuracy = predictor.train(cleaned_df)
    
    # Example Prediction (User Input Simulation)
    # Predicting a 25ft shot for Carmelo Anthony
    prob = predictor.predict_probability(
        shotX=25, shotY=25, distance=25, player="Carmelo Anthony", shot_type=3
    )
    print(f"Prediction: Carmelo Anthony has a {prob*100:.1f}% chance of making a 25ft shot.")

    # 4. Interactive Visualization [cite: 55]
    # This prepares the figure which can be rendered in Jupyter or exported to HTML
    plotter = InteractiveCourtPlotter()
    fig = plotter.plot_shot_data(cleaned_df, player="Carmelo Anthony")
    fig.show()
    
    # To view the plot, typically you would use fig.show() or write to HTML
    # fig.write_html("carmelo_interactive.html")
    print("Interactive plot object created.")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()