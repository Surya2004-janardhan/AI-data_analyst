import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    """Processes the DataFrame to clean and prepare it for analysis."""
    # 🔹 Handle Missing Values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # 🔹 Remove Duplicates
    df.drop_duplicates(inplace=True)

    # 🔹 Convert Data Types
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    # 🔹 Encode Categorical Variables
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # 🔹 Remove Outliers using IQR
    for col in df.select_dtypes(include=np.number).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = np.where((df[col] < lower) | (df[col] > upper), np.nan, df[col])
        df[col].fillna(df[col].median(), inplace=True)

    # 🔹 Normalize Numeric Data
    scaler = StandardScaler()
    df[df.select_dtypes(include=np.number).columns] = scaler.fit_transform(df.select_dtypes(include=np.number))

    print("Preprocessing completed successfully.")
    return df
