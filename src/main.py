from process_data import ShotDataLoader
from clean_data import ShotDataCleaner
from model import ShotOutcomePredictor
from interactive_dash_app import InteractiveDashApp
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

    # 3. Get list of available players
    available_players = sorted(cleaned_df['player'].unique())
    print("\nAvailable players:")
    for i, player in enumerate(available_players, 1):
        print(f"  {i}. {player}")
    
    player_input = input("\nEnter player name(s) to train (comma-separated, or press Enter for all): ").strip()
    
    if player_input.lower() == 'all' or player_input == '':
        players_to_train = available_players
    else:
        players_to_train = [p.strip() for p in player_input.split(',')]
    
    # Filter to only players that exist in the data
    players_to_train = [p for p in players_to_train if p in available_players]
    
    if not players_to_train:
        print("Error: No valid players selected.")
        return

    # 4. Train models for selected players
    predictors_dict = {}
    print(f"\nTraining models for {len(players_to_train)} player(s)...")
    
    for player_name in players_to_train:
        predictor = ShotOutcomePredictor(min_samples=100)
        try:
            accuracy = predictor.train(cleaned_df, player=player_name)
            predictors_dict[player_name] = predictor
            print(f"  ✓ {player_name}: Accuracy = {accuracy:.4f}")
        except ValueError as e:
            print(f"  ✗ {player_name}: {e}")

    if not predictors_dict:
        print("Error: Could not train any models.")
        return

    # 5. Launch Interactive Dash App
    print("\n" + "="*60)
    print("Launching interactive shot probability predictor...")
    print("="*60)
    
    app = InteractiveDashApp(cleaned_df, predictors_dict)
    app.run(debug=False, port=8050)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
