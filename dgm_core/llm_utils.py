"""
LLM (大規模言語モデル) との連携に必要なユーティリティ関数を提供するモジュール。
Gemini APIを使用します。
"""

import os
import json
import re
import time
import logging
from typing import Dict, Any, Optional, List, Union
import google.generativeai as genai
from .config import Config

logger = logging.getLogger(__name__)

# Google Generative AIを初期化
try:
    # 環境変数からAPIキーを取得
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    genai.configure(api_key=api_key)
    
    # セーフティ設定（必要に応じて調整）
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        }
    ]
    
except Exception as e:
    print(f"Error initializing Gemini API: {e}")
    raise

def setup_llm():
    """Initialize the LLM configuration"""
    try:
        genai.configure(api_key=Config.GEMINI_API_KEY)
        
        # Create generation config with temperature
        generation_config = {
            "temperature": Config.LLM_TEMPERATURE,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Initialize the model with generation config
        model = genai.GenerativeModel(model_name=Config.LLM_MODEL)
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to setup LLM: {e}")
        raise

def generate_improvement_suggestion(
    task_description: str,
    current_pipeline_code: str,
    performance_metrics: Dict[str, float],
    task_config: Dict[str, Any],
    global_config: Dict[str, Any]
) -> str:
    """LLMを使用してパイプラインの改善案を生成します。
    
    Args:
        task_description (str): タスクの説明
        current_pipeline_code (str): 現在のパイプラインコード
        performance_metrics (Dict[str, float]): 現在のパフォーマンスメトリクス
        task_config (Dict[str, Any]): タスク固有の設定
        global_config (Dict[str, Any]): グローバルな設定
        
    Returns:
        str: LLMからの応答（JSON形式の文字列）
    """
    model = setup_llm()
    
    # プロンプトを構築
    prompt = f"""あなたは機械学習パイプラインの専門家です。以下の情報に基づいて、与えられたパイプラインを改善するための具体的な提案を行ってください。

# タスクの説明
{task_description}

# 現在のパフォーマンスメトリクス
```json
{json.dumps(performance_metrics, indent=2)}
```

# 現在のパイプラインコード
```python
{current_pipeline_code}
```

# 改善の方向性（例）
- 特徴量エンジニアリングの改善（新しい特徴量の追加、不要な特徴量の削除、特徴量の変換など）
- モデルの変更（異なるアルゴリズムの使用、アンサンブル手法の導入など）
- ハイパーパラメータの最適化
- データの前処理・正規化の改善
- クラス不均衡への対応
- 交差検証戦略の改善

# 出力形式（必ずJSON形式で出力してください）
```json
{{
  "improvement_description": "改善内容の簡潔な説明（1-2文）",
  "improved_code": "改善されたPythonコード（完全な実行可能なコード）",
  "expected_improvement": "期待される改善内容とその根拠（1-2文）"
}}
```

# 注意点
- 既存の関数シグネチャ（preprocess_data, create_model_for_optuna, train_final_model, evaluate_model）は維持してください。
- コードは完全で実行可能な状態で出力してください。
- パフォーマンスの向上が見込める具体的な改善に焦点を当ててください。
"""

    for attempt in range(Config.MAX_RETRIES):
        try:
            # Generate content with safety settings
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            response = model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config={"temperature": Config.LLM_TEMPERATURE}
            )
            
            if response and response.text:
                return response.text
            raise ValueError("Empty response from LLM")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < Config.MAX_RETRIES - 1:
                time.sleep(Config.RETRY_DELAY)
            else:
                raise

def parse_llm_suggestion(response: str) -> Dict[str, str]:
    """LLMからの応答をパースしてコードと説明を取り出す"""
    try:
        # JSONとして解析を試みる
        data = json.loads(response)
        
        # 必要なキーが存在するか確認
        required_keys = ['improvement_description', 'improved_code', 'expected_improvement']
        if not all(key in data for key in required_keys):
            raise ValueError("Missing required keys in LLM response")
            
        # コードブロックのクリーンアップ
        if '```python' in data['improved_code']:
            code = data['improved_code'].split('```python')[1].split('```')[0].strip()
        else:
            code = data['improved_code'].strip()
            
        return {
            'description': data['improvement_description'],
            'code': code,
            'expected_improvement': data['expected_improvement']
        }
        
    except json.JSONDecodeError:
        # JSONとして解析できない場合のフォールバック処理
        code_start = response.find('```python')
        code_end = response.find('```', code_start + 8)
        
        if code_start != -1 and code_end != -1:
            code = response[code_start + 8:code_end].strip()
            return {
                'description': "パースエラー - コードのみ抽出",
                'code': code,
                'expected_improvement': "不明"
            }
        else:
            raise ValueError("Could not parse LLM response")
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return None


def validate_pipeline_code(code: str) -> Dict[str, Any]:
    """生成されたパイプラインコードを検証します。
    
    Args:
        code (str): 検証するPythonコード
        
    Returns:
        Dict[str, Any]: 検証結果とエラーメッセージ
    """
    required_functions = [
        'preprocess_data',
        'create_model_for_optuna',
        'train_final_model',
        'evaluate_model'
    ]
    
    result = {
        'is_valid': True,
        'missing_functions': [],
        'syntax_errors': []
    }
    
    # 構文チェック
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        result['is_valid'] = False
        result['syntax_errors'].append({
            'line': e.lineno,
            'message': str(e),
            'text': e.text
        })
    
    # 必要な関数が存在するかチェック
    local_vars = {}
    try:
        exec(code, globals(), local_vars)
        
        for func_name in required_functions:
            if func_name not in local_vars or not callable(local_vars[func_name]):
                result['missing_functions'].append(func_name)
                result['is_valid'] = False
    except Exception as e:
        result['is_valid'] = False
        result['syntax_errors'].append({
            'line': 'N/A',
            'message': f"Error executing code: {str(e)}",
            'text': ''
        })
    
    return result


def generate_initial_pipeline(task_config: Dict[str, Any]) -> str:
    """タスク設定に基づいて初期パイプラインコードを生成します。
    
    Args:
        task_config (Dict[str, Any]): タスク設定
        
    Returns:
        str: 生成されたパイプラインコード
    """
    # シンプルな初期パイプラインを返す
    return 
# 必要なライブラリをインポート
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import optuna

# 前処理ステップの説明
FE_STEPS_DESCRIPTION = """
# 1. 基本的な欠損値処理
# 2. カテゴリカル変数のエンコーディング
# 3. 不要な列の削除
"""

# 使用するモデルの種類
MODEL_TYPE = "RandomForest"

def preprocess_data(df: pd.DataFrame, task_config: dict) -> pd.DataFrame:
    """データの前処理を行う関数
    
    Args:
        df (pd.DataFrame): 入力データ
        task_config (dict): タスク設定
        
    Returns:
        pd.DataFrame: 前処理済みデータ
    """
    df_processed = df.copy()
    
    # ここに前処理のロジックを実装
    # 例: 欠損値処理
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            # カテゴリカル変数の最頻値で埋める
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            # 数値変数の中央値で埋める
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # カテゴリカル変数のエンコーディング
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    return df_processed

def create_model_for_optuna(X_train: pd.DataFrame, y_train: pd.Series, 
                          X_val: pd.DataFrame, y_val: pd.Series, 
                          task_config: dict) -> dict:
    """Optunaを使用してハイパーパラメータを最適化する関数
    
    Args:
        X_train (pd.DataFrame): 学習用特徴量
        y_train (pd.Series): 学習用ターゲット
        X_val (pd.DataFrame): 検証用特徴量
        y_val (pd.Series): 検証用ターゲット
        task_config (dict): タスク設定
        
    Returns:
        dict: 最適なハイパーパラメータ
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10, step=1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5, step=1),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': task_config.get('random_seed', 42)
        }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # 検証データで評価
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        
        return score
    
    # Optunaスタディを作成して最適化を実行
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=task_config.get('n_trials', 20))
    
    return study.best_params

def train_final_model(X: pd.DataFrame, y: pd.Series, best_params: dict) -> object:
    """最適なパラメータで最終モデルをトレーニングする関数
    
    Args:
        X (pd.DataFrame): 学習用特徴量
        y (pd.Series): 学習用ターゲット
        best_params (dict): 最適なハイパーパラメータ
        
    Returns:
        object: トレーニング済みモデル
    """
    model = RandomForestClassifier(**best_params)
    model.fit(X, y)
    return model

def evaluate_model(model: object, X: pd.DataFrame, y_true: pd.Series, 
                  y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray], 
                  task_config: dict) -> dict:
    """モデルを評価する関数
    
    Args:
        model: 評価対象のモデル
        X (pd.DataFrame): 特徴量
        y_true (pd.Series): 真のラベル
        y_pred (np.ndarray): 予測ラベル
        y_pred_proba (Optional[np.ndarray]): クラス確率(利用可能な場合)
        task_config (dict): タスク設定
        
    Returns:
        dict: 評価メトリクス
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    # 確率予測が利用可能な場合、ROC-AUCも計算
    if y_pred_proba is not None and hasattr(model, 'predict_proba'):
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except Exception as e:
            print(f"Warning: Could not calculate ROC-AUC: {e}")
    
    return metrics


def preprocess_data(df: pd.DataFrame, task_config: dict) -> pd.DataFrame:
    """データの前処理を行う関数
    
    Args:
        df (pd.DataFrame): 入力データ
        task_config (dict): タスク設定
        
    Returns:
        pd.DataFrame: 前処理済みデータ
    """
    df_processed = df.copy()
    
    # ここに前処理のロジックを実装
    # 例: 欠損値処理
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            # カテゴリカル変数の最頻値で埋める
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            # 数値変数の中央値で埋める
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # カテゴリカル変数のエンコーディング
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    return df_processed

def create_model_for_optuna(X_train: pd.DataFrame, y_train: pd.Series, 
                          X_val: pd.DataFrame, y_val: pd.Series, 
                          task_config: dict) -> dict:
    """Optunaを使用してハイパーパラメータを最適化する関数
    
    Args:
        X_train (pd.DataFrame): 学習用特徴量
        y_train (pd.Series): 学習用ターゲット
        X_val (pd.DataFrame): 検証用特徴量
        y_val (pd.Series): 検証用ターゲット
        task_config (dict): タスク設定
        
    Returns:
        dict: 最適なハイパーパラメータ
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10, step=1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5, step=1),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': task_config.get('random_seed', 42)
        }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # 検証データで評価
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        
        return score
    
    # Optunaスタディを作成して最適化を実行
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=task_config.get('n_trials', 20))
    
    return study.best_params

def train_final_model(X: pd.DataFrame, y: pd.Series, best_params: dict) -> object:
    """最適なパラメータで最終モデルをトレーニングする関数
    
    Args:
        X (pd.DataFrame): 学習用特徴量
        y (pd.Series): 学習用ターゲット
        best_params (dict): 最適なハイパーパラメータ
        
    Returns:
        object: トレーニング済みモデル
    """
    model = RandomForestClassifier(**best_params)
    model.fit(X, y)
    return model

def evaluate_model(model: object, X: pd.DataFrame, y_true: pd.Series, 
                  y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray], 
                  task_config: dict) -> dict:
    """モデルを評価する関数
    
    Args:
        model: 評価対象のモデル
        X (pd.DataFrame): 特徴量
        y_true (pd.Series): 真のラベル
        y_pred (np.ndarray): 予測ラベル
        y_pred_proba (Optional[np.ndarray]): クラス確率(利用可能な場合)
        task_config (dict): タスク設定
        
    Returns:
        dict: 評価メトリクス
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    # 確率予測が利用可能な場合、ROC-AUCも計算
    if y_pred_proba is not None and hasattr(model, 'predict_proba'):
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except Exception as e:
            print(f"Warning: Could not calculate ROC-AUC: {e}")
    
    return metrics

# 使用例
if __name__ == "__main__":
    # テスト用の設定
    test_config = {
        "evaluation_metric": "accuracy",
        "random_seed": 42,
        "n_trials": 10
    }