import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss

def build_complex_nn(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def analyze_deep_learning_patterns(df_cleaned):
    results = {}
    
    num_cols = df_cleaned.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        results['Error'] = "No numerical data found for deep learning analysis."
        return results
    
    cat_cols = df_cleaned.select_dtypes(include='object').columns
    for col in cat_cols:
        df_cleaned[col] = LabelEncoder().fit_transform(df_cleaned[col])
    
    print("🔍 Checking for NaN values before processing...")
    print(df_cleaned.isnull().sum())
    print(f"Total NaN values: {df_cleaned.isnull().sum().sum()}")
    
    df_cleaned.dropna(subset=[num_cols[-1]], inplace=True)  # Ensure target column has no NaNs
    df_cleaned.fillna(df_cleaned.median(), inplace=True)  # Fill missing values
    
    target_col = num_cols[-1]
    X = df_cleaned.drop(columns=[target_col]).values
    y = df_cleaned[target_col].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    y = y.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = build_complex_nn(X_train.shape[1])
    
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1, validation_data=(X_test, y_test))
    y_pred = model.predict(X_test)
    y_pred = np.nan_to_num(y_pred)  # Replace NaNs in predictions with zero
    
    print("🔍 Checking for NaN values in predictions...")
    print(f"Total NaN values in y_pred: {np.isnan(y_pred).sum()}")
    print(f"Total NaN values in y_test: {np.isnan(y_test).sum()}")
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    # rmlse = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(np.clip(y_pred, 1e-7, None))))

    # rmlse = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(np.clip(y_pred, 1e-7, None))))
    r2 = r2_score(y_test, y_pred)
    # auc_roc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) == 2 else None
    # logloss = log_loss(y_test, y_pred) if len(np.unique(y_test)) == 2 else None
    
    if len(np.unique(y_test)) == 2:  # Ensure binary classification
        y_test_bin = (y_test > np.median(y_test)).astype(int)  # Convert to binary labels
        y_pred_bin = (y_pred > np.median(y_pred)).astype(int)  # Convert predictions to binary

        precision = precision_score(y_test_bin, y_pred_bin, zero_division=0)
        recall = recall_score(y_test_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
        conf_matrix = confusion_matrix(y_test_bin, y_pred_bin)
    else:
        precision = recall = f1 = conf_matrix = None  # Not applicable for regression

    
    results['Deep Learning Model Performance'] = {
        'Mean Squared Error (MSE)': mse,
        'Mean Absolute Error (MAE)': mae,
        'Root Mean Squared Error (RMSE)': rmse,
        'R² Score': r2,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': conf_matrix,
        'Observations': "Based on the deep learning model's evaluation, additional performance measures such as precision, recall, and AUC-ROC score provide more insight into predictive capability. Further analysis with domain-specific evaluation criteria is recommended for deeper insights."
    }
    
    results['Dataset Summary'] = {
        'Number of Samples': df_cleaned.shape[0],
        'Number of Features': df_cleaned.shape[1],
        'Feature Names': list(df_cleaned.columns),
        'Target Column': target_col
    }
    
    return results