def generate_initial_pipeline(task_config: dict) -> str:
    """Generate the initial pipeline template."""
    template = """import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

MODEL_TYPE = "RandomForestClassifier"
FE_STEPS_DESCRIPTION = [
    "数値特徴量の欠損値を中央値で補完",
    "カテゴリ特徴量の欠損値を最頻値で補完",
    "カテゴリ変数をOne-Hotエンコーディング",
    "数値特徴量の標準化"
]

def preprocess_data(df, task_config=None):
    \"\"\"データの前処理を行う関数\"\"\"
    # Print debug information
    print("Starting preprocessing with data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    
    df_processed = df.copy()
    
    # 目的変数を除外して特徴量を処理
    target_col = task_config.get('target_column', 'Survived') if task_config else 'Survived'
    features = [col for col in df_processed.columns if col != target_col]
    
    # 特徴量の型に基づいて分類
    numerical_features = df_processed[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df_processed[features].select_dtypes(include=['object']).columns.tolist()
    
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)
    
    # 前処理パイプラインの定義
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    # 前処理パイプラインの結合
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    try:
        # 前処理の実行
        X = df_processed[features]
        transformed_array = preprocessor.fit_transform(X)
        
        # カラム名の生成
        transformed_feature_names = []
        
        # 数値特徴量の名前をそのまま使用
        transformed_feature_names.extend(numerical_features)
        
        # カテゴリカル特徴量の新しい列名を生成
        if categorical_features:
            for i, feature in enumerate(categorical_features):
                encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                categories = encoder.categories_[i][1:]  # 'first' カテゴリを除外
                transformed_feature_names.extend([f"{feature}_{cat}" for cat in categories])
        
        # 変換されたデータをDataFrameに変換
        X_transformed = pd.DataFrame(
            transformed_array,
            columns=transformed_feature_names,
            index=df_processed.index
        )
        
        # 目的変数が含まれている場合は追加
        if target_col in df_processed.columns:
            X_transformed[target_col] = df_processed[target_col].astype(int)
        
        return X_transformed
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

def create_model_for_optuna(X_train, y_train, X_val, y_val, task_config):
    \"\"\"Optunaを使用してモデルのハイパーパラメータを最適化\"\"\"
    import optuna
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return accuracy_score(y_val, y_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    return study.best_params

def train_final_model(X_train, y_train, best_params):
    \"\"\"最適化されたパラメータでモデルを学習\"\"\"
    model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val, y_pred=None, y_pred_proba=None, task_config=None):
    \"\"\"モデルの評価を行う\"\"\"
    if y_pred is None:
        y_pred = model.predict(X_val)
    if y_pred_proba is None and hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba)
    }
    
    return metrics
"""
    return template