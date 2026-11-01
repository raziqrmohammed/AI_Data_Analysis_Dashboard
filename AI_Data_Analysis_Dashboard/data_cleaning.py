import pandas as pd


def clean_data(df, missing_method, remove_duplicates):

    data = df.copy()
    messages = []

    # Remove duplicate records
    if remove_duplicates:

        before = len(data)
        data = data.drop_duplicates()
        removed = before - len(data)

        messages.append(
            f"Removed {removed} duplicate records."
        )

    # Handle missing values
    if missing_method == "Mean / Mode":

        for col in data.columns:

            if data[col].isnull().any():

                if pd.api.types.is_numeric_dtype(data[col]):

                    value = data[col].mean()

                    if pd.notna(value):
                        data[col] = data[col].fillna(value)

                else:

                    mode = data[col].mode(dropna=True)

                    if not mode.empty:
                        data[col] = data[col].fillna(
                            mode.iloc[0]
                        )

        messages.append(
            "Missing values handled using mean and mode."
        )

    elif missing_method == "Drop Rows":

        before = len(data)
        data = data.dropna()

        messages.append(
            f"Removed {before - len(data)} rows with missing values."
        )

    elif missing_method == "Fill with 0 / Unknown":

        for col in data.columns:

            if pd.api.types.is_numeric_dtype(data[col]):

                data[col] = data[col].fillna(0)

            else:

                data[col] = data[col].fillna("Unknown")

        messages.append(
            "Missing values filled with 0 or Unknown."
        )

    else:

        messages.append(
            "Missing values were left unchanged."
        )

    return data, messages