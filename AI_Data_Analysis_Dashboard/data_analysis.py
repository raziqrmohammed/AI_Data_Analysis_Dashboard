import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest


# =====================================
# GENERATE AUTOMATIC INSIGHTS
# =====================================

def generate_insights(df):

    insights = []

    if df.empty:
        return ["Dataset is empty."]

    insights.append(
        f"Dataset contains {len(df)} rows "
        f"and {len(df.columns)} columns."
    )

    # Missing values analysis
    missing = df.isnull().sum()

    if missing.sum() > 0:

        column = missing.idxmax()

        insights.append(
            f"Column '{column}' contains "
            f"{missing[column]} missing values."
        )

    else:

        insights.append(
            "No missing values found."
        )

    # Numerical analysis
    numeric = df.select_dtypes(
        include=np.number
    )

    for col in numeric.columns:

        values = numeric[col].dropna()

        if not values.empty:

            insights.append(
                f"{col}: Average = {values.mean():.2f}, "
                f"Maximum = {values.max():.2f}, "
                f"Minimum = {values.min():.2f}"
            )

    # Categorical analysis
    categorical = df.select_dtypes(
        exclude=np.number
    )

    for col in categorical.columns:

        values = categorical[col].dropna()

        if not values.empty:

            top_value = values.mode().iloc[0]

            insights.append(
                f"Most common value in {col}: {top_value}"
            )

    # Correlation analysis
    if len(numeric.columns) >= 2:

        correlation = numeric.corr().abs()

        np.fill_diagonal(correlation.values, 0)

        if correlation.notna().any().any():

            pair = correlation.stack().idxmax()

            value = correlation.loc[pair[0], pair[1]]

            insights.append(
                f"Strongest numerical relationship: "
                f"{pair[0]} and {pair[1]}, "
                f"correlation = {value:.2f}."
            )

    return insights[:15]


# =====================================
# ANOMALY DETECTION
# =====================================

def detect_anomalies(df):

    numeric = df.select_dtypes(
        include=np.number
    ).copy()

    numeric = numeric.replace(
        [np.inf, -np.inf], np.nan
    )

    numeric = numeric.dropna(
        axis=1,
        how="all"
    )

    if len(numeric) < 10 or numeric.empty:
        return pd.DataFrame()

    numeric = numeric.fillna(
        numeric.median()
    )

    numeric = numeric.loc[
        :, numeric.nunique() > 1
    ]

    if numeric.empty:
        return pd.DataFrame()

    model = IsolationForest(
        contamination="auto",
        random_state=42
    )

    predictions = model.fit_predict(numeric)

    result = df.copy()

    result["Anomaly"] = predictions == -1

    return result[
        result["Anomaly"]
    ]