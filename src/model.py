import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


class ShotOutcomePredictor:
    """
    Per-player Random Forest classifier to predict shot success probability.
    Uses a simple binary-search-style procedure to choose n_estimators
    based on cross-validation for the selected player.
    """

    def __init__(self, min_samples: int = 100):
        self.model = None
        self.preprocessor = None
        self.trained_player: str | None = None
        self.min_samples = min_samples
        self.best_n_estimators: int | None = None

    
    def _build_preprocessor(self):
        """Create the ColumnTransformer used for all models."""
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

    def _cv_score_for_n_estimators(self, X, y, n_estimators: int, cv: int = 3) -> float:
        """
        Evaluate a RandomForest with given n_estimators using cross-validation.
        """
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,          # fixed depth; you can tune later if you want
            n_jobs=-1,
            random_state=42
        )

        pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", clf)
        ])

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )

        return scores.mean()

    def _binary_search_n_estimators(self, X, y, low: int = 50, high: int = 600, iterations: int = 4) -> int:
        """
        Binary/ternary-search-style procedure to pick a good n_estimators.
        - Starts with [low, high]
        - At each step, evaluates low, mid, high
        - Narrows the interval around the best-performing value
        """
        print(f"Searching n_estimators between {low} and {high}...")

        best_n = low
        best_score = -1.0

        for it in range(iterations):
            mid = (low + high) // 2
            candidates = sorted(set([low, mid, high]))  # ensure unique & ordered

            scores = {}
            for n in candidates:
                score = self._cv_score_for_n_estimators(X, y, n)
                scores[n] = score
                print(f"  Iter {it+1}, n_estimators={n}: CV accuracy={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_n = n

            # Pick the best candidate in this iteration
            best_in_iter = max(scores.items(), key=lambda x: x[1])[0]

            # Shrink search window around best_in_iter
            # (simple heuristic; not mathematically perfect but works well)
            window = max(25, (high - low) // 4)
            low = max(10, best_in_iter - window)
            high = best_in_iter + window

            print(f"  -> After iter {it+1}, best so far: n_estimators={best_n} (CV={best_score:.4f}), new range=({low}, {high})")

        print(f"Final chosen n_estimators={best_n} with CV accuracy={best_score:.4f}")
        return best_n


    def train(self, df: pd.DataFrame, player: str):
        """
        Trains the Random Forest model on the provided dataframe for a single player.
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

        # Build preprocessor once (shared across CV and final model)
        self._build_preprocessor()

        # 1) Use binary-search-style tuning to pick n_estimators
        best_n = self._binary_search_n_estimators(X, y, low=50, high=600, iterations=4)
        self.best_n_estimators = best_n

        # 2) Train final model with best_n on a hold-out train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        final_clf = RandomForestClassifier(
            n_estimators=best_n,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )

        self.model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", final_clf)
        ])

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(
            f"Model Training Complete for {player}. "
            f"Accuracy: {acc:.4f} (n_estimators={best_n})"
        )
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
