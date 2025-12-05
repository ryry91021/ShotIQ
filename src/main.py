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

    # Get player name before training
    player_name = input("Enter player name for shot probability prediction: ")

    # 3. Train per-player Machine Learning Model
    predictor = ShotOutcomePredictor(min_samples=100)
    try:
        accuracy = predictor.train(cleaned_df, player=player_name)
    except ValueError as e:
        print(e)
        return

    # Example Prediction (User Input Simulation)
    prob = predictor.predict_probability(
        shotX=25,
        shotY=25,
        distance=25,
        player=player_name,
        shot_type=3  # whatever encoding you're using for shot_type
    )
    print(f"Prediction: {player_name} has a {prob*100:.1f}% chance of making a 25ft shot.")

    # 4. Interactive Visualization (historical shots for that player)
    plotter = InteractiveCourtPlotter()
    fig = plotter.plot_shot_data(cleaned_df, player=player_name)
    fig.show()

    print("Interactive plot object created.")
    elapsed_time = time.perf_counter() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
