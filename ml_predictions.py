import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.inspection import permutation_importance

def analyze_ml_patterns(df_cleaned):
    results = {}
    
    num_cols = df_cleaned.select_dtypes(include=np.number).columns
    cat_cols = df_cleaned.select_dtypes(include='object').columns
    
    if len(num_cols) == 0:
        results['Error'] = "No numerical data found for ML analysis."
        return results
    
    for col in cat_cols:
        df_cleaned[col] = LabelEncoder().fit_transform(df_cleaned[col])
    
    target_col = num_cols[-1]
    X = df_cleaned.drop(columns=[target_col])
    y = df_cleaned[target_col]
    
    # Determine if the target is classification or regression
    if y.nunique() <= 10 and y.dtype in [np.int64, np.int32, np.uint8]:
        task_type = "classification"
    else:
        task_type = "regression"
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {}
    if task_type == "regression":
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100),
            'XGBoost': XGBRegressor(n_estimators=100),
            'LightGBM': LGBMRegressor(n_estimators=100)
        }
    elif task_type == "classification":
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100),
            'XGBoost': XGBClassifier(n_estimators=100),
            'LightGBM': LGBMClassifier(n_estimators=100)
        }
    
    model_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if task_type == "regression":
            model_results[f'{name} MAE'] = mean_absolute_error(y_test, y_pred)
        else:
            model_results[f'{name} Accuracy'] = accuracy_score(y_test, y_pred)
    
    results.update(model_results)
    
    best_model_name = min(model_results, key=model_results.get)
    best_model = models[best_model_name.split()[0]]
    
    perm_importance = permutation_importance(best_model, X_test, y_test, scoring='accuracy' if task_type == "classification" else 'neg_mean_absolute_error')
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': perm_importance.importances_mean}).sort_values(by='Importance', ascending=False)
    results['Feature Importance'] = feature_importance.to_dict()
    
    kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=100)
    df_cleaned['Cluster'] = kmeans.fit_predict(df_cleaned[num_cols])
    results['Clusters'] = df_cleaned['Cluster'].value_counts().to_dict()
    
    results['Dataset Summary'] = {
        'Number of Samples': df_cleaned.shape[0],
        'Number of Features': df_cleaned.shape[1],
        'Feature Names': list(df_cleaned.columns),
        'Target Column': target_col,
        'Task Type': task_type
    }
    
    return results