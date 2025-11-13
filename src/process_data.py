import os
from pathlib import Path
from typing import List, Optional

import pandas as pd


class ShotDataLoader:
    """
    Loader class for csv shot data
    Converts parquet file to pandas dataframe.
    """

    def __init__(self, base_dir: Path, data_subdir: str = "../data", extension: str = ".parquet",):
        self.base_dir = base_dir
        self.data_dir = (base_dir / data_subdir).resolve()
        self.extension = extension.lower()
        self.dataframes: List[pd.DataFrame] = []





    
    def load_parquet(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Read a single CSV file into a DataFrame, handling timeouts.
        input: file_path - Path to the CSV file
        output: pd.DataFrame if successful, otherwise None
        """
        print(f"Reading {file_path}")

        #Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

        #List files in data directory
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith(self.extension):
                print(f"Found {filename} in {self.data_dir}")


        # Check parquet to csv
        try:
            return pd.read_parquet(file_path, columns=[
                "player", 
                "team", 
                "shotX", 
                "shotY", 
                "made", 
                "distance", 
                "shot_type"
            ])
        
        except TimeoutError as e:
            print(f"  !! TimeoutError on {file_path}: {e}")
            return None

    

    


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent 
    print(BASE_DIR)
    loader = ShotDataLoader(base_dir=BASE_DIR, data_subdir="../data")
    df = loader.load_parquet(BASE_DIR / "../data/shots.parquet")
