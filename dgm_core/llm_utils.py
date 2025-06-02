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

def setup_llm(global_config: dict = None) -> 'google.generativeai.GenerativeModel':
    """LLMの設定を初期化する

    Args:
        global_config (dict, optional): グローバル設定. Defaults to None.

    Returns:
        google.generativeai.GenerativeModel: 初期化されたモデル
    """
    try:
        # APIキーの設定
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        
        # グローバル設定からLLM設定を取得（デフォルトはgemini-1.5-flash）
        llm_config = global_config.get('llm', {}) if global_config else {}
        model_name = llm_config.get('model_name', 'gemini-1.5-flash')
        
        # セーフティ設定の定義
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]
        
        # モデルの初期化
        model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings
        )
        
        # レートリミット対策の設定をログに出力
        logger.info(f"Initialized {model_name} with safety settings")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to setup LLM: {e}")
        raise

def parse_llm_suggestion(suggestion: str) -> Optional[Dict[str, str]]:
    """LLMの応答をパースする"""
    if not suggestion:
        return None

    try:
        # 前後の空白を削除
        cleaned_suggestion = suggestion.strip()
        
        # まずそのままJSONとして解析を試みる
        try:
            data = json.loads(cleaned_suggestion)
            if all(key in data for key in ['improvement_description', 'improved_code', 'expected_improvement']):
                # コードのクリーンアップ
                code = data['improved_code']
                if '```python' in code:
                    code = code.split('```python')[1].split('```')[0].strip()
                data['improved_code'] = code
                return data
        except json.JSONDecodeError:
            pass

        # マークダウンブロックからJSONを探す
        json_match = re.search(r'```json\s*(.*?)\s*```', suggestion, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1).strip())
                if all(key in data for key in ['improvement_description', 'improved_code', 'expected_improvement']):
                    # コードのクリーンアップ
                    code = data['improved_code']
                    if '```python' in code:
                        code = code.split('```python')[1].split('```')[0].strip()
                    data['improved_code'] = code
                    return data
            except:
                pass

        logger.warning("Could not find valid JSON in response")
        return None

    except Exception as e:
        logger.error(f"Error parsing suggestion: {str(e)}")
        return None

def generate_improvement_suggestion(
    task_description: str,
    current_pipeline_code: str,
    performance_metrics: Dict[str, float],
    task_config: Dict[str, Any],
    global_config: Dict[str, Any]
) -> str:
    """LLMを使用してパイプラインの改善案を生成する"""
    try:
        model = setup_llm(global_config)
        
        # プロンプトをより制御しやすい形式に変更
        prompt = f"""Below is an ML pipeline that needs improvement. Follow these steps exactly:

1. First, analyze the current pipeline:
{current_pipeline_code}

2. Current metrics:
{json.dumps(performance_metrics, indent=2)}

3. Task details:
{task_description}

4. Generate improvements focusing on:
- Feature engineering
- Model selection
- Hyperparameter tuning
- Preprocessing steps

5. Return your response in this exact format:
{{
    "improvement_description": "<list key improvements>",
    "improved_code": "<full_code>",
    "expected_improvement": "<specific metrics improvements>"
}}

Important:
- Return ONLY valid JSON
- NO markdown formatting in code
- NO explanation outside JSON
- Keep ALL function signatures identical
"""

        # より保守的な生成設定
        generation_config = {
            "temperature": 0.1,
            "candidate_count": 1,
            "max_output_tokens": 1024,  # 出力を制限
            "top_p": 0.8,
            "top_k": 10
        }

        # 再試行ロジックの改善
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                if response and response.text:
                    # 応答のクリーンアップを強化
                    cleaned_text = response.text.strip()
                    # JSON部分の抽出を試みる
                    json_match = re.search(r'\{[\s\S]*\}', cleaned_text)
                    if json_match:
                        json_str = json_match.group(0)
                        # 検証
                        json.loads(json_str)
                        return json_str
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # 指数バックオフ
                continue

        return ""

    except Exception as e:
        logger.error(f"Error generating improvement: {str(e)}")
        return ""

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