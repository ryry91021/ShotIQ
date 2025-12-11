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

    # 3. Choose player(s) (before training)
    def ask_yes_no(prompt, default=False):
        """Prompt the user for a yes/no answer. Returns True for yes, False for no.

        Keeps asking until the user enters a recognizable response. If the user
        just hits enter and a default is provided, the default is returned.
        """
        yes = {"y", "yes"}
        no = {"n", "no"}
        while True:
            resp = input(prompt).strip().lower()
            if resp == "" and default is not None:
                return bool(default)
            if resp in yes:
                return True
            if resp in no:
                return False
            print("Please enter 'y' or 'n' (or 'yes'/'no').")

    while True:
        player_name = input("Enter player name (E.g. LeBron James, Josh Hart) for shot probability prediction (or 'q' to quit): ").strip()
        if player_name.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break
        if not player_name:
            print("No player name entered — please try again.")
            continue

        # 4. Train per-player Machine Learning Model
        predictor = ShotOutcomePredictor(min_samples=100)
        try:
            accuracy = predictor.train(cleaned_df, player=player_name)
        except ValueError as e:
            print(e)
            # Ask user if they want to try another player
            if ask_yes_no("Do you want to try a different player? (y/n): ", default=True):
                continue
            else:
                break

        # 5. Example Prediction (User Input Simulation)
        shotX, shotY = 25, 25
        default_shot_type = 3  # adjust to match your shot_type encoding
        prob = predictor.predict_probability(
            shotX=shotX,
            shotY=shotY,
            shot_type=default_shot_type
        )
        # Calculate distance from hoop (25, 4.75)
        hoop_x, hoop_y = 25, 4.75
        distance = ((shotX - hoop_x)**2 + (shotY - hoop_y)**2) ** 0.5
        print(f"Prediction: {player_name} has a {prob*100:.1f}% chance of making a {distance:.1f}ft shot.")

        # 5b. Allow user to manually input coordinates for on-demand predictions
        while True:
            if not ask_yes_no("Would you like to input shot coordinates for a prediction? (y/n): ", default=False):
                break

            coord_str = input("Enter coordinates as 'x,y' or 'x y' (or 'b' to go back): ").strip()
            if coord_str.lower() in {"b", "back", "q", "quit", "exit"}:
                break

            # Normalize and parse coordinates
            coord_str = coord_str.replace(',', ' ')
            parts = coord_str.split()
            if len(parts) != 2:
                print("Invalid input. Please enter two numeric values for x and y, e.g. '25 20' or '25,20'.")
                continue

            try:
                sx = float(parts[0])
                sy = float(parts[1])
            except ValueError:
                print("Coordinates must be numeric. Try again.")
                continue

            # Optional shot type input
            st_input = input(f"Enter shot_type (press Enter to use default {default_shot_type}): ").strip()
            if st_input == "":
                st = default_shot_type
            else:
                try:
                    st = int(st_input)
                except ValueError:
                    print("Invalid shot_type; using default.")
                    st = default_shot_type

            try:
                prob = predictor.predict_probability(shotX=sx, shotY=sy, shot_type=st)
                distance = ((sx - hoop_x)**2 + (sy - hoop_y)**2) ** 0.5
                print(f"{player_name} chance at ({sx:.1f}, {sy:.1f}) [{distance:.1f}ft]: {prob*100:.1f}%")
            except Exception as e:
                print(f"Could not compute prediction: {e}")
                continue

        # 6. Interactive Visualization (historical shots for that player)
        plotter = InteractiveCourtPlotter()
        fig = plotter.plot_shot_data(cleaned_df, player=player_name)
        fig.show()

        print("Interactive plot object created.")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Total execution time: {elapsed_time:.2f} seconds")

        # Ask whether to analyze another player
        if ask_yes_no("Do you want to analyze another player? (y/n): ", default=False):
            continue
        else:
            break


if __name__ == "__main__":
    main()
