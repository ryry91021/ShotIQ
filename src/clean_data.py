from __future__ import annotations

import pandas as pd
from typing import List


class ShotDataCleaner:
    """
    Handles cleaning and normalization of raw shot data.
    """

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
            df["made"] = df["made"].astype(int)

        # Convert shot_type to  format 2 or 3.
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

        return df
