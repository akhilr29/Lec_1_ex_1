"""Helper functions for the airline satisfaction notebook."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# <project root>/data, resolved from this file so it works from any directory
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

SERVICE_KEYWORDS = [
    "service",
    "comfort",
    "food",
    "drink",
    "cleanliness",
    "entertainment",
    "boarding",
    "wifi",
    "check-in",
    "checkin",
    "online booking",
    "leg room",
]
DELAY_KEYWORDS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
CATEGORICAL_PLOT_COLS = ["Gender", "Class", "Type of Travel", "Customer Type"]
NUMERIC_PLOT_COLS = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


# ----------------------------------------------------------------------------
# Loading and basic inspection
# ----------------------------------------------------------------------------
def load_data(data_dir=DATA_DIR):
    """Load the sample submission, test and train CSVs.

    Returns
    -------
    tuple of pd.DataFrame
        ``(df_sample, df_test, df_train)``
    """
    data_dir = Path(data_dir)
    df_sample = pd.read_csv(data_dir / "sample_submission.csv")
    df_test = pd.read_csv(data_dir / "test.csv")
    df_train = pd.read_csv(data_dir / "train.csv")
    print("Dataset loaded successfully")
    return df_sample, df_test, df_train


def print_shapes(**dfs):
    """Print the row and column counts of each named dataframe."""
    for name, df in dfs.items():
        print(f"{name} Data Shape:")
        print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}\n")


def print_column_lists(**dfs):
    """Print the column names of each named dataframe."""
    for name, df in dfs.items():
        print(f"{name}\n", df.columns.tolist(), "\n")


def print_target_distribution(df, target="satisfaction"):
    """Print the counts and percentages of each target class."""
    print(df[target].value_counts())
    print(df[target].value_counts(normalize=True).mul(100).round(2))


def print_missing_values(**dfs):
    """Print the columns with missing values for each named dataframe."""
    for name, df in dfs.items():
        missing = df.isnull().sum()
        print(f"Missing values in {name} Data")
        print(missing[missing > 0])
        print("-" * 50)


# ----------------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------------
def plot_target_distribution(df, target="satisfaction"):
    """Count plot of the target variable."""
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=target, ax=ax)
    ax.set_title("Airline satisfaction Distribution")
    return fig


def plot_categorical_satisfaction(df, cat_cols, target="satisfaction"):
    """2x2 grid of satisfaction rate by each categorical column."""
    cat_cols = CATEGORICAL_PLOT_COLS
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    for ax, col in zip(axes.flatten(), cat_cols):
        sns.barplot(data=df, x=col, y=target, errorbar=None, ax=ax)
        ax.set_title(f'Satisfaction Rate By "{col}"')
        ax.set_ylabel("Satisfaction Rate")
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def find_matching_columns(df, keywords):
    """Return columns whose lower-cased name contains any keyword."""
    return [
        col
        for col in df.columns
        if any(kw.lower() in col.lower() for kw in keywords)
    ]


def compare_service_ratings(df, service_cols, target="satisfaction"):
    """Mean rating of each service column, split by target class."""
    return df.groupby(target)[service_cols].mean().T


def plot_service_ratings(comparison):
    """Bar chart of the output of :func:`compare_service_ratings`."""
    ax = comparison.plot(kind="bar", figsize=(12, 7))
    ax.set_title("Average service Ratings by Satisfaction")
    ax.set_ylabel("Avg Rating")
    ax.set_xlabel("Service Feature")
    ax.tick_params(axis="x", rotation=60)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.figure.tight_layout()
    return ax.figure


def plot_delay_boxplots(df, delay_cols, target="satisfaction"):
    """One box plot per delay column, split by target class."""
    figs = []
    for col in delay_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x=target, y=col, showfliers=False, ax=ax)
        ax.set_title(f'"{col}" by Satisfaction')
        fig.tight_layout()
        figs.append(fig)
    return figs


def get_numeric_columns(df, exclude=("id",)):
    """Numeric column names, minus any in ``exclude``."""
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]


def summarise_numeric(df, num_cols):
    """Transposed describe() table without the count row."""
    return df[num_cols].describe().drop("count").T.round(2)


def plot_numeric_distributions(df, cols=None, bins=40):
    """One histogram per column."""
    cols = cols or NUMERIC_PLOT_COLS
    figs = []
    for col in cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=df, x=col, bins=bins, kde=False, ax=ax)
        ax.set_title(f'Distribution of "{col}"')
        fig.tight_layout()
        figs.append(fig)
    return figs


def target_correlations(df, num_cols, target="satisfaction"):
    """Correlation of every numeric column with the target.

    Returns
    -------
    tuple
        ``(corr_data, target_corr)``: the numeric frame including the target
        as an int, and the sorted correlations with the target.
    """
    corr_data = df[num_cols].copy()
    corr_data[target] = df[target].astype(int)
    correlation = corr_data.corr(numeric_only=True)
    target_corr = correlation[target].drop(target).sort_values(ascending=False)
    return corr_data, target_corr


def plot_correlation_heatmap(corr_data, target_corr, n=15, target="satisfaction"):
    """Heatmap of the ``n`` features most correlated with the target."""
    top_features = target_corr.abs().nlargest(n).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_data[top_features + [target]].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: Selected Features")
    fig.tight_layout()
    return fig


def eda_summary(df_train, df_test, target="satisfaction"):
    """Headline numbers about the data, returned as a dict."""
    satisfied = df_train[target].mean() * 100
    return {
        "training_rows": df_train.shape[0],
        "training_columns": df_train.shape[1],
        "test_rows": df_test.shape[0],
        "missing_train_values": int(df_train.isnull().sum().sum()),
        "duplicated_train_rows": int(df_train.duplicated().sum()),
        "satisfied_percentage": round(satisfied, 2),
        "not_satisfied_percentage": round(100 - satisfied, 2),
    }


# ----------------------------------------------------------------------------
# Preprocessing and modelling
# ----------------------------------------------------------------------------
def prepare_features(df_train, df_test, target="satisfaction", id_col="id"):
    """Split train into X / y and test into features plus ids.

    Returns
    -------
    tuple
        ``(X, y, X_test, test_ids, cat_features, num_features)``
    """
    X = df_train.drop(columns=[target, id_col])
    y = df_train[target].astype(int)
    test_ids = df_test[id_col].copy()
    X_test = df_test.drop(columns=id_col)

    cat_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    num_features = X.select_dtypes(include=["number"]).columns.tolist()

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"No. of categorical columns: {len(cat_features)}")
    print(f"No. of numerical columns: {len(num_features)}")
    print("-" * 70)
    print(f"categorical columns:\n{cat_features}")
    print("-" * 70)
    print(f"numerical columns:\n{num_features}")
    return X, y, X_test, test_ids, cat_features, num_features


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified train / validation split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor(num_features, cat_features):
    """Impute + scale numerics, impute + one-hot encode categoricals."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    category_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, num_features),
            ("cat", category_pipeline, cat_features),
        ]
    )


def build_logistic_model(preprocessor):
    """Logistic regression pipeline."""
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=5000)),
        ]
    )


def build_random_forest(preprocessor):
    """Random forest pipeline with the notebook's hyper-parameters."""
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=14,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def build_gradient_boosting(preprocessor):
    """Gradient boosting pipeline with the notebook's hyper-parameters."""
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.08,
                    max_depth=3,
                    min_samples_leaf=20,
                    subsample=0.8,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_logistic(model, X_train, y_train, X_test, y_test):
    """Fit a model and print accuracy, F1 scores and the full report."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"F1 binary: {f1_score(y_test, y_pred)}")
    print(f"F1 macro: {f1_score(y_test, y_pred, average='macro')}")
    print(f"F1 weighted: {f1_score(y_test, y_pred, average='weighted')}")
    print("-" * 80)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 70)
    print(classification_report(y_test, y_pred))
    return y_pred


def fit_and_evaluate(model, name, X_train, y_train, X_test, y_test):
    """Fit, time and score a probabilistic model.

    Returns
    -------
    dict
        ``accuracy``, ``f1``, ``auc`` and ``training_time`` (seconds).
    """
    print(f"{name} training started...")
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start
    print("Training completed")

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    results = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba),
        "training_time": training_time,
    }

    print(f"\n======= {name} Result ============")
    print("Accuracy: ", round(results["accuracy"], 5))
    print("F1 Score: ", round(results["f1"], 5))
    print("ROC-AUC: ", round(results["auc"], 5))
    print("Training Time: ", round(training_time, 1), "seconds")
    print("===========================================")
    return results


def pick_best_model(rf_results, gb_results):
    """Name of the model with the higher ROC-AUC."""
    if rf_results["auc"] > gb_results["auc"]:
        return "Random Forest"
    return "Gradient Boosting"
