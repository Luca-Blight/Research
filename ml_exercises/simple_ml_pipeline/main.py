import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression


class DataModeler:
    def __init__(self, sample_df: pd.DataFrame):
        self.train_df = sample_df.copy()
        self.model = None

    def prepare_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare a dataframe so it contains only the columns to model and having suitable types.
        If the argument is None, work on the training data passed in the constructor."""

        if oos_df is None:
            df = self.train_df
        else:
            df = oos_df.copy()

        # Converting 'transaction_date' to datetime type and using month instead of entire date.
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["transaction_date"] = df["transaction_date"].dt.month

        # Drop 'customer_id' and 'outcome' columns
        df = df.drop(columns=["customer_id"], errors="ignore")

        if oos_df is None:
            self.train_df = df
        else:
            return df

    def impute_missing(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:

        """Fill any missing values with the appropriate mean (average) value.
        If the argument is None, work on the training data passed in the constructor.
        Hint: Watch out for data leakage in your solution."""

        if oos_df is None:
            df = self.train_df
            self.mean_values = df.mean()
        else:
            df = oos_df.copy()

        df.fillna(self.mean_values, inplace=True)

        if oos_df is None:
            self.train_df = df
        else:
            return df

    def fit(self) -> None:
        """Fit the model of your choice on the training data paased in the constructor, assuming it has
        been prepared by the functions prepare_data and impute_missing"""

        y = self.train_df["outcome"]
        X = self.train_df.drop("outcome", axis=1)
        self.model = LogisticRegression()
        self.model.fit(X, y)

    def model_summary(self) -> str:
        """Returns summary of your trained model."""
        if self.model:
            # Model type
            model_type = type(self.model).__name__

            # Features and their names
            n_features = self.model.coef_.shape[1]
            feature_names = self.train_df.drop("outcome", axis=1).columns.tolist()

            # Coefficients and intercept
            coefficients = self.model.coef_
            intercept = self.model.intercept_

            # Constructing the summary string
            summary = f"Fit Model Summary:\n"
            summary += f"  - Model Type: {model_type}\n"
            summary += f"  - Number of Features: {n_features}\n"
            summary += f"  - Features: {feature_names}\n"
            summary += f"  - Coefficients: {coefficients}\n"
            summary += f"  - Intercept: {intercept}\n"

            return summary
        else:
            return "Model has not been fitted yet."

    def predict(self, oos_df: pd.DataFrame = None) -> pd.Series:
        """Make a set of predictions with your model. Assume the data has been prepared by the
        functions prepare_data and impute_missing.
        If the argument is None, work on the training data passed in the constructor."""
        if oos_df is None:
            X = self.train_df.drop("outcome", axis=1)
            return self.model.predict(X)
        else:
            return self.model.predict(oos_df)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":

    transact_train_sample = pd.DataFrame(
        {
            "customer_id": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "amount": [1, 3, 12, 6, 0.5, 0.2, np.nan, 5, np.nan, 3],
            "transaction_date": [
                "2022-01-01",
                "2022-08-01",
                None,
                "2022-12-01",
                "2022-02-01",
                None,
                "2022-02-01",
                "2022-01-01",
                "2022-11-01",
                "2022-01-01",
            ],
            "outcome": [False, True, True, True, False, False, True, True, True, False],
        }
    )
    print(f"Training sample:\n{transact_train_sample}\n")
    print(f"Current dtypes:\n{transact_train_sample.dtypes}\n")

    transactions_modeler = DataModeler(transact_train_sample)
    transactions_modeler.prepare_data()
    print(f"Changed columns to:\n{transactions_modeler.train_df.dtypes}\n")

    transactions_modeler.impute_missing()
    print(f"Imputed missing as mean:\n{transactions_modeler.train_df}\n")

    print("Fitting model...")
    transactions_modeler.fit()

    transactions_modeler.model_summary()

    in_sample_predictions = transactions_modeler.predict()
    print(f"Predicted on training sample: {in_sample_predictions}\n")
    print(
        f'Accuracy of Training Set = {sum(in_sample_predictions == transact_train_sample["outcome"]) / len(transact_train_sample) * 100}%'
    )

    transactions_modeler.save("transact_modeler")
    loaded_modeler = DataModeler.load("transact_modeler")

    transact_test_sample = pd.DataFrame(
        {
            "customer_id": [21, 22, 23, 24, 25],
            "amount": [0.5, np.nan, 8, 3, 2],
            "transaction_date": [
                "2022-02-01",
                "2022-11-01",
                "2022-06-01",
                None,
                "2022-02-01",
            ],
        }
    )
    adjusted_test_sample = transactions_modeler.prepare_data(transact_test_sample)
    print(f"Changed columns to:\n{adjusted_test_sample.dtypes}\n")

    filled_test_sample = transactions_modeler.impute_missing(adjusted_test_sample)
    print(f"Imputed missing as mean:\n{filled_test_sample}\n")

    oos_predictions = transactions_modeler.predict(filled_test_sample)
    print(f"Predictions on test data: {oos_predictions}\n ")
    expected_outcomes = [False, True, True, False, False]
    print(
        f"Accuracy of Test Set = {sum(oos_predictions == expected_outcomes) / len(expected_outcomes) * 100}%"
    )


class DataModeler:
    # ... (existing code)

    def model_summary(self) -> str:
        if self.model:
            # Getting model type
            model_type = type(self.model).__name__

            # Getting number of features and their names
            n_features = self.model.coef_.shape[1]
            feature_names = self.train_df.drop("outcome", axis=1).columns.tolist()

            # Getting coefficients and intercept
            coefficients = self.model.coef_
            intercept = self.model.intercept_

            # Constructing the summary string
            summary = f"Model Summary:\n"
            summary += f"  - Model Type: {model_type}\n"
            summary += f"  - Number of Features: {n_features}\n"
            summary += f"  - Features: {feature_names}\n"
            summary += f"  - Coefficients: {coefficients}\n"
            summary += f"  - Intercept: {intercept}\n"

            return summary
        else:
            return "Model has not been fitted yet."
