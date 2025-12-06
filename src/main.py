from process_data import ShotDataLoader
from clean_data import ShotDataCleaner
from interactive_plots import InteractiveCourtPlotter
from model import ShotOutcomePredictor
from pathlib import Path
import time


def main():
    start_time = time.perf_counter()
    BASE_DIR = Path(__file__).parent

    # 1. Load Data
    loader = ShotDataLoader(base_dir=BASE_DIR, data_subdir="../data")
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

    # 3. Choose player (before training)
    player_name = input("Enter player name for shot probability prediction: ")

    # 4. Train per-player Machine Learning Model
    predictor = ShotOutcomePredictor(min_samples=100)
    try:
        accuracy = predictor.train(cleaned_df, player=player_name)
    except ValueError as e:
        print(e)
        return

    print(f"Trained per-player model for {player_name}. Accuracy: {accuracy:.4f}")

    # 5. Example Prediction (User Input Simulation)
    prob = predictor.predict_probability(
        shotX=25,
        shotY=25,
        distance=25,
        shot_type=3  # adjust to match your shot_type encoding
    )
    print(f"Prediction: {player_name} has a {prob*100:.1f}% chance of making a 25ft shot.")

    # 6. Interactive Visualization (historical shots for that player)
    plotter = InteractiveCourtPlotter()
    fig = plotter.plot_shot_data(cleaned_df, player=player_name)
    fig.show()

    print("Interactive plot object created.")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
