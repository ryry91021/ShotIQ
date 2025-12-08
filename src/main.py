from process_data import ShotDataLoader
from clean_data import ShotDataCleaner
from interactive_plots import InteractiveCourtPlotter
from model import ShotOutcomePredictor
from pathlib import Path
import time
import numpy as np


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

    # 5. Interactive Visualization (clear court with click-for-probability)
    plotter = InteractiveCourtPlotter(player=player_name, predictor=predictor)
    fig = plotter.plot_shot_data(cleaned_df, player=player_name)
    
    # Add click event callback for probability prediction
    def on_click(trace, points, selector):
        if points.point_inds:
            # This is for clicking on the scatter trace; we want clicks on the empty court
            pass
    
    # For Plotly, we'll create a custom callback wrapper
    def create_click_handler(predictor, player_name):
        def handle_click(clickData):
            if clickData is None or 'points' not in clickData or len(clickData['points']) == 0:
                return
            
            point = clickData['points'][0]
            shotX = point.get('x')
            shotY = point.get('y')
            
            if shotX is not None and shotY is not None:
                shot_type = 3  # Default shot type
                prob = predictor.predict_probability(
                    shotX=shotX,
                    shotY=shotY,
                    shot_type=shot_type
                )
                hoop_x, hoop_y = 25, 4.75
                distance = np.sqrt((shotX - hoop_x)**2 + (shotY - hoop_y)**2)
                print(f"\n[CLICKED] Shot at ({shotX:.1f}, {shotY:.1f})")
                print(f"{player_name} has a {prob*100:.1f}% chance at {distance:.1f}ft")
        
        return handle_click
    
    fig.show()

    print("Interactive plot displayed. Click on court locations to predict shot probability.")

    # 6. Example Prediction (User Input Simulation)
    shotX, shotY = 25, 25
    shot_type = 3  # adjust to match your shot_type encoding
    prob = predictor.predict_probability(
        shotX=shotX,
        shotY=shotY,
        shot_type=shot_type
    )
    # Calculate distance from hoop (25, 4.75)
    hoop_x, hoop_y = 25, 4.75
    distance = ((shotX - hoop_x)**2 + (shotY - hoop_y)**2) ** 0.5
    print(f"\nExample Prediction: {player_name} has a {prob*100:.1f}% chance of making a {distance:.1f}ft shot.")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
