from __future__ import annotations

import pandas as pd
from typing import List


class ShotDataCleaner:
    """
    Handles cleaning and normalization of raw shot data.
    """

    def __init__(self, essential_cols: List[str] | None = None) -> None:
        if essential_cols is None:
            essential_cols = ["player", "team", "shotX", "shotY", "distance", "shot_type", "made"]
        self.essential_cols = essential_cols

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning steps to a copy of the input DataFrame.
        input:  df - raw concatenated DataFrame from loader (./process_data.py)
        output: cleaned DataFrame
        """
        df = df.copy()

        # --- 1. Strip/normalize string columns ---
        for col in df.select_dtypes(include="object").columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace("'", "", regex=False)
                .str.replace('"', "", regex=False)
            )

        #Convert numeric columns to appropriate types.
        for col in ["shotX", "shotY", "distance"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert make miss to binary 0/1.
        if "made" in df.columns:
            # if it's already 0/1, this will be no-op; if bool, will become 0/1
            df["made"] = df["made"].astype(int)

        # Convert shot_type to standardized format (2 or 3).
        if "shot_type" in df.columns:
            def _normalize_shot_type(val):
                if pd.isna(val):
                    return None
                s = str(val).lower()
                if "3" in s:
                    return 3
                if "2" in s:
                    return 2
                return s.upper()

            df["shot_type"] = df["shot_type"].map(_normalize_shot_type)

        # Drop row if missing essential columns.
        missing_essentials = [c for c in self.essential_cols if c not in df.columns]
        if missing_essentials:
            raise KeyError(f"Missing essential columns: {missing_essentials}")

        df = df.dropna(subset=self.essential_cols)

        return df
