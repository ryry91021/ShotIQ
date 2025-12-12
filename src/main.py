from process_data import ShotDataLoader
from clean_data import ShotDataCleaner
from interactive_plots import InteractiveCourtPlotter
from model import ShotOutcomePredictor
from pathlib import Path
import time


DIV = "=" * 60
SUBDIV = "-" * 60


def banner():
    print(DIV)
    print("ShotIQ — NBA Shot Probability Simulator")
    print("Load data -> Clean -> Train ML model -> Predict shots")
    print(DIV)
    print()


def pretty(msg):
    """Helper for status messages."""
    print(f"\n{SUBDIV}\n{msg}\n{SUBDIV}\n")


def main():
    banner()

    BASE_DIR = Path(__file__).parent

    # ======================================================
    # 1. Load Data
    pretty("Loading dataset...")

    loader = ShotDataLoader(base_dir=BASE_DIR, data_subdir="../data")
    parquet_path = BASE_DIR / "../data/shots.parquet"

    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found.\nRun download_nba_data.sh first.")
        return

    df = loader.load_parquet(parquet_path)
    if df is None:
        print("ERROR: Failed to load parquet file. Exiting.")
        return

    print("Data loaded successfully.")

    # ======================================================
    # 2. Clean Data
    pretty("Cleaning dataset...")

    cleaner = ShotDataCleaner()
    cleaned_df = cleaner.clean(df)

    print("Data cleaned successfully.")

    # ======================================================
    # Ask-Yes-No helper
    def ask_yes_no(prompt, default=False):
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
            print("Please enter 'y' or 'n'.")

    hoop_x, hoop_y = 25, 4.75

    # ======================================================
    # Main Player Loop
    while True:
        print()
        print(DIV)
        player_name = input(
            "Enter player name (e.g., LeBron James) or 'q' to quit: "
        ).strip()
        print(DIV)

        if player_name.lower() in {"q", "quit", "exit"}:
            print("Exiting program.")
            break

        if not player_name:
            print("No player name entered. Try again.")
            continue

        # 4. Train ML Model
        pretty(f"Training model for: {player_name}")

        predictor = ShotOutcomePredictor(min_samples=100)

        try:
            accuracy = predictor.train(cleaned_df, player=player_name)
            print("Training complete.")
            print(f"Model accuracy for {player_name}: {accuracy:.3f}")
        except ValueError as e:
            print(f"ERROR: {e}")
            if ask_yes_no("Try a different player? (y/n): ", default=True):
                continue
            else:
                break

        # 5. Visualization
        pretty(f"Generating interactive shot chart for {player_name}...")
        plotter = InteractiveCourtPlotter()
        fig = plotter.plot_shot_data(cleaned_df, player=player_name)
        fig.show()
        print("Interactive visualization displayed.")

        # ======================================================
        # 6. Manual Shot Input Loop
        while True:
            if not ask_yes_no(
                "\nWould you like to enter shot coordinates for prediction? (y/n): ",
                default=False,
            ):
                break

            coord_str = input(
                "Enter coordinates as 'x,y' or 'x y' (or 'b' to go back): "
            ).strip()

            if coord_str.lower() in {"b", "back", "q", "quit", "exit"}:
                break

            coord_str = coord_str.replace(",", " ")
            parts = coord_str.split()
            if len(parts) != 2:
                print("Invalid format. Example: 25 20")
                continue

            try:
                sx = float(parts[0])
                sy = float(parts[1])
            except ValueError:
                print("Coordinates must be numeric.")
                continue

            # Compute distance + infer shot type
            distance = ((sx - hoop_x)**2 + (sy - hoop_y)**2)**0.5
            st = 2 if distance <= 22.0 else 3

            print(f"Distance from hoop: {distance:.1f} ft")
            print(f"Assigned shot type: {st}-pointer")

            # Prediction
            try:
                prob = predictor.predict_probability(shotX=sx, shotY=sy, shot_type=st)

                print("\n" + SUBDIV)
                print(f"Shot Probability for {player_name}")
                print(f"Location: ({sx:.1f}, {sy:.1f})")
                print(f"Distance from hoop: {distance:.1f} ft")
                print(f"Estimated Make Probability: {prob*100:.1f}%")
                print(SUBDIV + "\n")

            except Exception as e:
                print(f"ERROR: Could not compute prediction: {e}")

        # ======================================================
        # Another player?
        if ask_yes_no("Analyze another player? (y/n): ", default=False):
            print("\n" + DIV)
            continue
        else:
            break

    print("\nProgram finished.")


if __name__ == "__main__":
    main()
