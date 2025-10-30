from nrg_interview_prep.config import Config
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import json
import pickle


class SurvivalModel:
    def __init__(self, data_path: Optional[str] = None, absolute_test_set_size: Optional[int] = None):
        if data_path is None:
            self.data_path = Config.survival_data_file
        else:
            self.data_path = Path(data_path)
            # Check if the provided path exists
            if not self.data_path.exists():
                raise FileNotFoundError(
                    f"The provided data path {self.data_path} does not exist.")
        self.absolute_test_set_size = absolute_test_set_size

    def _prepDataset(self):
        data_df = pd.read_csv(self.data_path)
        print(
            f"Data loaded from {self.data_path} with {len(data_df)} records.")
        if self.absolute_test_set_size is None:
            test_size = int(0.2 * len(data_df))
        else:
            test_size = self.absolute_test_set_size
        rng = np.random.default_rng(seed=17)

        censored_idx = data_df.index[data_df['E'] == 0].tolist()

        # Ensure atleast one censored data point
        chosen = []
        chosen.append(rng.choice(censored_idx))

        remaining_pool = data_df.index.difference(chosen)
        remaining_choices = rng.choice(
            remaining_pool, size=test_size - 1, replace=False)
        chosen.extend(list(remaining_choices))

        test_df = data_df.loc[chosen].reset_index(drop=True)
        train_df = data_df.drop(index=chosen).reset_index(drop=True)
        return train_df, test_df

    def train(self):
        training_data, test_data = self._prepDataset()
        cph = CoxPHFitter()
        cph.fit(training_data, duration_col='T', event_col='E')
        # Save model summary as json
        model_summary = {
            "summary": cph.summary.to_dict(),
        }

        test_risk = cph.predict_partial_hazard(test_data)
        concordance = concordance_index(
            test_data['T'], -test_risk, test_data['E'])
        print(f"Concordance index on test set: {concordance:.3f}")
        model_summary["test_set_concordance_index"] = concordance

        with open(Config.survival_model_stats_path, 'w') as f:
            json.dump(model_summary, f)
        self.model = cph
        print("\n=== Hazard Ratios (exp(coef)) ===")
        print(cph.hazard_ratios_)

        for var, coef in cph.params_.items():
            if coef > 0:
                direction = "increases"
            elif coef < 0:
                direction = "decreases"
            else:
                direction = "does not affect"
            print(
                f"• A one-unit increase in {var} {direction} the hazard rate (coef={coef:.3f}).")

        # Save model as pickle
        with open(Config.trained_survival_model_path, 'wb') as f:
            pickle.dump(cph, f)
        print(
            f"\nModel trained and saved to {Config.trained_survival_model_path}")

    def predict(self, X):
        # See if saved model exists
        if not Path(Config.trained_survival_model_path).exists():
            raise FileNotFoundError(
                f"Trained model not found at {Config.trained_survival_model_path}. Please train the model first.")
        with open(Config.trained_survival_model_path, 'rb') as f:
            cph = pickle.load(f)
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
        # Ensure X has the same columns as the training data
        training_columns = cph.params_.index.tolist()
        if not all(col in X.columns for col in training_columns):
            raise ValueError(
                f"Input DataFrame must contain the following columns: {training_columns}.")
        # Predict the hazard ratios
        hazard_ratios = cph.predict_partial_hazard(X)
        return hazard_ratios


def train_model(training_data_path: Optional[str] = None, absolute_test_set_size: Optional[int] = 10):
    survival_model = SurvivalModel(
        data_path=training_data_path,
        absolute_test_set_size=absolute_test_set_size
    )
    survival_model.train()
