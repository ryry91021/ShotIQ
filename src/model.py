import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

class ShotOutcomePredictor:
    """
    Trains a Random Forest Classifier to predict shot success probability.
    Goal: Provide probabilistic outcomes for shot locations[cite: 57].
    """

    def __init__(self, min_samples: int = 100):
        self.model = None
        self.preprocessor = None
        self.trained_player: str = None
        self.min_samples = min_samples


    def prepare_pipeline(self):
        """
        Creates a scikit-learn pipeline for preprocessing and modeling.
        Handles categorical variables (Player, Shot Type) via OneHotEncoding.
        """
        categorical_features = ['shot_type']

        categorical_transformer = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=True
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'  # numeric: shotX, shotY, distance
        )

        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=1500,
                max_depth=10,
                n_jobs=-1,
                random_state=42
            ))
        ])


    def train(self, df: pd.DataFrame, player: str):
        """
        Trains the Random Forest model on the provided dataframe.
        Inputs: df (Cleaned DataFrame), player (str): player name to filter by
        """
        print(f"Preparing data for training for player: {player} ...")

        # Filter to that player's shots only
        df_player = df[df['player'] == player].copy()

        if df_player.empty:
            raise ValueError(f"No data found for player: {player}")

        n_samples = len(df_player)
        if n_samples < self.min_samples:
            print(
                f"Warning: Only {n_samples} shots found for {player}. "
                f"min_samples={self.min_samples}. Model may be unstable."
            )

        self.trained_player = player

        feature_cols = ['shotX', 'shotY', 'distance', 'shot_type']
        target_col = 'made'

        X = df_player[feature_cols]
        y = df_player[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.prepare_pipeline()
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Model Training Complete for {player}. Accuracy: {acc:.4f}")
        return acc


    def predict_probability(self, shotX, shotY, distance, shot_type):
        """
        Returns the probability of a specific shot being made
        for the player this model was trained on (self.trained_player).
        """
        if self.model is None or self.trained_player is None:
            raise RuntimeError("Model has not been trained for any player yet.")

        input_data = pd.DataFrame({
            'shotX': [shotX],
            'shotY': [shotY],
            'distance': [distance],
            'shot_type': [shot_type]
        })

        prob = self.model.predict_proba(input_data)[0, 1]  # P(class=1 'made')
        return prob

