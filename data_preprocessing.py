# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# def preprocess_data(df):
#     """Processes the DataFrame to clean and prepare it for analysis."""
#     # 🔹 Handle Missing Values
#     for col in df.columns:
#         if df[col].dtype == "object":
#             df[col].fillna(df[col].mode()[0], inplace=True)
#         else:
#             df[col].fillna(df[col].median(), inplace=True)

#     # 🔹 Remove Duplicates
#     df.drop_duplicates(inplace=True)

#     # 🔹 Convert Data Types
#     for col in df.columns:
#         if "date" in col.lower():
#             df[col] = pd.to_datetime(df[col], errors="coerce")
#         elif df[col].dtype == "object":
#             try:
#                 df[col] = pd.to_numeric(df[col])
#             except:
#                 pass

#     # 🔹 Encode Categorical Variables
#     for col in df.select_dtypes(include="object").columns:
#         df[col] = LabelEncoder().fit_transform(df[col])

#     # 🔹 Remove Outliers using IQR
#     for col in df.select_dtypes(include=np.number).columns:
#         Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
#         df[col] = np.where((df[col] < lower) | (df[col] > upper), np.nan, df[col])
#         df[col].fillna(df[col].median(), inplace=True)

#     # 🔹 Normalize Numeric Data
#     scaler = StandardScaler()
#     df[df.select_dtypes(include=np.number).columns] = scaler.fit_transform(df.select_dtypes(include=np.number))

#     print("Preprocessing completed successfully.")
#     return df
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, PowerTransformer

def preprocess_data(df):
    """
    Preprocesses any CSV file dynamically by applying only the necessary steps.

    Args:
        file_path (str or file-like object): Path to CSV file or uploaded file.

    Returns:
        pd.DataFrame: Processed DataFrame or error message.
    """
    try:
        # 🔹 Load CSV File
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input is not a valid DataFrame!")

        # 🔹 Handle Missing Values
        for col in df.columns:
            if df[col].dtype == "object":  # Categorical columns
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            else:  # Numerical columns
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)

        # 🔹 Remove Duplicates
        if df.duplicated().sum() > 0:
            df.drop_duplicates(inplace=True)

        # 🔹 Convert Data Types
        for col in df.columns:
            if "date" in col.lower():  # Convert date columns
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif df[col].dtype == "object":  # Convert object to numeric if possible
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass

        # 🔹 Encode Categorical Variables (Only if required)
        categorical_cols = df.select_dtypes(include="object").columns
        if len(categorical_cols) > 0:
            encoder = LabelEncoder()
            for col in categorical_cols:
                df[col] = encoder.fit_transform(df[col])

        # 🔹 Handle Outliers using IQR
        for col in df.select_dtypes(include=np.number).columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            if ((df[col] < lower) | (df[col] > upper)).sum() > 0:  # Apply only if outliers exist
                df[col] = np.clip(df[col], lower, upper)  # Winsorization

        # 🔹 Normalize or Standardize Data (Only if needed)
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            scaler = MinMaxScaler() if df[numeric_cols].skew().abs().mean() < 1 else StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # 🔹 Log Transformation (Only for highly skewed data)
        skewed_cols = df[numeric_cols].skew()[df[numeric_cols].skew().abs() > 1].index
        if len(skewed_cols) > 0:
            transformer = PowerTransformer(method='yeo-johnson')
            df[skewed_cols] = transformer.fit_transform(df[skewed_cols])

        # 🔹 Remove Highly Correlated Features (Optional)
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
        if len(to_drop) > 0:
            df.drop(columns=to_drop, inplace=True)

        print("✅ Preprocessing completed successfully.")
        return df

    except Exception as e:
        return f"❌ Error in processing file: {e}"
