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

    def __init__(self):
        self.model = None
        self.preprocessor = None

    def prepare_pipeline(self):
        """
        Creates a scikit-learn pipeline for preprocessing and modeling.
        Handles categorical variables (Player, Shot Type) via OneHotEncoding.
        """
        categorical_features = ['player', 'shot_type']
        
        # Define transformer for categorical columns
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

        # Bundle preprocessing with the model
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough' # Keep numeric columns (shotX, shotY, distance)
        )

        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42))
        ])

    def train(self, df: pd.DataFrame):
        """
        Trains the Random Forest model on the provided dataframe.
        Inputs: df (Cleaned DataFrame)
        """
        print("Preparing data for training...")
        
        # Features and Target
        # ensure these columns exist from clean_data.py
        feature_cols = ['shotX', 'shotY', 'distance', 'player', 'shot_type']
        target_col = 'made'
        
        X = df[feature_cols]
        y = df[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train
        print("Training Random Forest Classifier...") # [cite: 57]
        self.prepare_pipeline()
        self.model.fit(X_train, y_train)

        # Evaluate
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Model Training Complete. Accuracy: {acc:.4f}")
        return acc

    def predict_probability(self, shotX, shotY, distance, player, shot_type):
        """
        Returns the probability of a specific shot being made.
        """
        input_data = pd.DataFrame({
            'shotX': [shotX],
            'shotY': [shotY],
            'distance': [distance],
            'player': [player],
            'shot_type': [shot_type]
        })
        
        # [:, 1] gets the probability of class '1' (Made Shot)
        prob = self.model.predict_proba(input_data)[0, 1]
        return prob