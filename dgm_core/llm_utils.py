"""
LLM (大規模言語モデル) との連携に必要なユーティリティ関数を提供するモジュール。
複数のLLMプロバイダー（Gemini, Groq Qwen-32Bなど）をサポートします。
"""

import json
import re
import time
import random
import traceback
import logging
import os
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from abc import ABC, abstractmethod

import yaml
from dotenv import load_dotenv

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# 設定をインポート
try:
    from .config import Config
except ImportError:
    # テスト時などで相対インポートが失敗する場合のフォールバック
    from dgm_core.config import Config

logger = logging.getLogger(__name__)

# LLMプロバイダーを定義する列挙型
class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    # 他のプロバイダーを追加可能

# プロバイダーごとのデフォルトモデル
DEFAULT_MODELS = {
    LLMProvider.GEMINI: "gemini-1.5-flash",
    LLMProvider.GROQ: "meta-llama/llama-4-scout-17b-16e-instruct"
}

# プロバイダーごとのAPIキー環境変数名
API_KEY_ENV_VARS = {
    LLMProvider.GEMINI: "GEMINI_API_KEY",
    LLMProvider.GROQ: "GROQ_API_KEY"
}

class LLMClient(ABC):
    """LLMクライアントの抽象基底クラス"""
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """テキストを生成する"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """チャット形式でテキストを生成する"""
        pass

class GeminiClient(LLMClient):
    """Google Gemini APIクライアント"""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        import google.generativeai as genai
        self.genai = genai
        
        # APIキーの設定
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        genai.configure(api_key=self.api_key)
        
        # モデル名の設定
        self.model_name = model_name or DEFAULT_MODELS[LLMProvider.GEMINI]
        
        # セーフティ設定
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
        ]
        
        # モデルの初期化
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings=self.safety_settings
        )
        
        logger.info(f"Initialized Gemini model: {self.model_name}")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """テキストを生成する"""
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": kwargs.get("temperature", 0.7),
                "max_output_tokens": kwargs.get("max_tokens", 2048),
                "top_p": kwargs.get("top_p", 0.95),
            }
        )
        return response.text
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """チャット形式でテキストを生成する"""
        chat = self.model.start_chat(history=[])
        response = chat.send_message(
            messages[-1]["content"],
            generation_config={
                "temperature": kwargs.get("temperature", 0.7),
                "max_output_tokens": kwargs.get("max_tokens", 2048),
                "top_p": kwargs.get("top_p", 0.95),
            }
        )
        return response.text

class GroqClient(LLMClient):
    """Groq APIクライアント（Qwen-32Bなど）"""
    
    def __init__(self, api_key: str = None, model_name: str = None, **kwargs):
        from groq import Groq
        
        # APIキーの設定
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        # モデル名の設定
        self.model_name = model_name or DEFAULT_MODELS[LLMProvider.GROQ]
        
        # デフォルトパラメータの設定
        self.default_params = {
            'temperature': 0.7,
            'max_tokens': 2048,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'stop': None,
            'stream': False
        }
        
        # デフォルトパラメータを更新
        self.default_params.update({k: v for k, v in kwargs.items() if k in self.default_params})
        
        # クライアントの初期化
        self.client = Groq(api_key=self.api_key)
        
        logger.info(f"Initialized Groq model: {self.model_name} with params: {self.default_params}")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """テキストを生成する"""
        messages = [
            {"role": "system", "content": "あなたは親切で、正確で、役立つアシスタントです。"},
            {"role": "user", "content": prompt}
        ]
        
        # パラメータをマージ（メソッド呼び出し時の引数を優先）
        params = self.default_params.copy()
        params.update({k: v for k, v in kwargs.items() if k in self.default_params})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **{k: v for k, v in params.items() if v is not None}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in generate_text: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """チャット形式でテキストを生成する"""
        # メッセージを適切な形式に変換
        formatted_messages = []
        for msg in messages:
            role = "assistant" if msg.get("role") == "model" else msg.get("role", "user")
            formatted_messages.append({"role": role, "content": msg.get("content", "")})
        
        # パラメータをマージ（メソッド呼び出し時の引数を優先）
        params = self.default_params.copy()
        params.update({k: v for k, v in kwargs.items() if k in self.default_params})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                **{k: v for k, v in params.items() if v is not None}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def get_llm_client(
    provider: Union[str, LLMProvider] = LLMProvider.GROQ,
    model_name: str = None,
    api_key: str = None,
    **kwargs
) -> LLMClient:
    """指定されたプロバイダーのLLMクライアントを取得する
    
    Args:
        provider (Union[str, LLMProvider]): LLMプロバイダー
        model_name (str, optional): 使用するモデル名. Defaults to None.
        api_key (str, optional): APIキー. Defaults to None.
        **kwargs: その他の引数（プロバイダー固有）
        
    Returns:
        LLMClient: 初期化されたLLMクライアント
    """
    # プロバイダーを正規化
    if isinstance(provider, str):
        try:
            provider = LLMProvider(provider.lower())
        except ValueError:
            raise ValueError(f"サポートされていないLLMプロバイダーです: {provider}")
    
    # 環境変数からAPIキーを取得（指定されていない場合）
    if not api_key:
        api_key = os.getenv(API_KEY_ENV_VARS[provider])
        if not api_key:
            raise ValueError(f"{API_KEY_ENV_VARS[provider]} 環境変数が設定されていません。")
    
    # デフォルトのモデル名を設定
    if not model_name:
        model_name = DEFAULT_MODELS[provider]
    
    # プロバイダーに応じたクライアントを初期化
    if provider == LLMProvider.GEMINI:
        return GeminiClient(api_key=api_key, model_name=model_name, **kwargs)
    elif provider == LLMProvider.GROQ:
        return GroqClient(api_key=api_key, model_name=model_name, **kwargs)
    else:
        raise ValueError(f"サポートされていないLLMプロバイダーです: {provider}")

def setup_llm(global_config: dict = None) -> LLMClient:
    """LLMクライアントを初期化する

    Args:
        global_config (dict, optional): グローバル設定. Defaults to None.

    Returns:
        LLMClient: 初期化されたLLMクライアント
    """
    # デフォルト設定
    default_config = {
        'llm_provider': 'groq',  # デフォルトはGroq
        'llm_model': DEFAULT_MODELS[LLMProvider.GROQ],
        'temperature': 0.7,
        'max_tokens': 2048,
        'top_p': 1.0,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
    }
    
    # グローバル設定をマージ
    if global_config:
        llm_config = {**default_config, **global_config.get('llm', {})}
    else:
        llm_config = default_config
    
    # LLMクライアントを初期化
    try:
        client = get_llm_client(
            provider=llm_config['llm_provider'],
            model_name=llm_config['llm_model'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens'],
            top_p=llm_config['top_p'],
            frequency_penalty=llm_config['frequency_penalty'],
            presence_penalty=llm_config['presence_penalty']
        )
        
        logger.info(f"Initialized {llm_config['llm_provider']} with model {llm_config['llm_model']}")
        return client
        
    except Exception as e:
        logger.error(f"LLMの初期化中にエラーが発生しました: {e}")
        raise
        
    except Exception as e:
        logger.error(f"Failed to setup LLM: {e}")
        raise

# ヘルパー関数: parse_llm_suggestion の外部または同じモジュールレベルに定義
import re
import json



# def _extract_suggestion_from_json_like_prefix(text_block: str) -> str:
#     ...（旧実装は削除）...

def extract_suggestion_from_llm_output(llm_output: str) -> str:
    """
    LLMの出力から最初のPythonコードブロック（```python ... ```）を抽出して返す。
    Args:
        llm_output (str): LLMの出力テキスト
    Returns:
        str: 抽出されたPythonコード、または見つからない場合は空文字列
    Returns:
        dict | None: 抽出・パースされたJSONオブジェクト。見つからない場合はNone。
    """
    import re, json
    inside_json_block = False
    json_lines = []
    for line in llm_output.split('\n'):
        striped_line = line.strip()
        if striped_line.startswith("```json"):
            inside_json_block = True
            continue
        if inside_json_block and striped_line.startswith("```"):
            inside_json_block = False
            break
        if inside_json_block:
            json_lines.append(line)
    if not json_lines:
        fallback_pattern = r"\{.*?\}"
        matches = re.findall(fallback_pattern, llm_output, re.DOTALL)
        for candidate in matches:
            candidate = candidate.strip()
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    candidate_clean = re.sub(r"[\x00-\x1F\x7F]", "", candidate)
                    try:
                        return json.loads(candidate_clean)
                    except json.JSONDecodeError:
                        continue
        return None
    json_string = "\n".join(json_lines).strip()
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
        try:
            return json.loads(json_string_clean)
        except json.JSONDecodeError:
            return None

def generate_improvement_suggestion(
    task_description: str,
    current_pipeline_code: str,
    performance_metrics: Dict[str, float],
    task_config: Dict[str, Any],
    global_config: Dict[str, Any]
) -> Dict[str, str]:
    """LLMを使用してパイプラインの改善案を生成する
    
    Args:
        task_description (str): タスクの説明
        current_pipeline_code (str): 現在のパイプラインコード
        performance_metrics (Dict[str, float]): 現在のパフォーマンスメトリクス
        task_config (Dict[str, Any]): タスク設定
        global_config (Dict[str, Any]): グローバル設定
        
    Returns:
        Dict[str, str]: 改善案の情報
    """
    try:
        # LLMを初期化
        llm = setup_llm(global_config)
        
        # グローバル設定からLLM設定を取得
        llm_config = global_config.get('llm', {}) if global_config else {}
        provider = llm_config.get('llm_provider', 'groq')
        model_name = llm_config.get('llm_model', DEFAULT_MODELS.get(LLMProvider(provider), 'qwen-qwq-32b'))
        
        # パフォーマンスメトリクスを読みやすい形式に変換
        metrics_str = '\n'.join([f'- {k}: {v:.4f}' for k, v in performance_metrics.items()])
        
        # より具体的なプロンプトテンプレート
        prompt = """
        以下の機械学習パイプラインの改善を依頼します。
        
        # タスクの説明
        {task_description}
        
        # 現在のパフォーマンスメトリクス
        {metrics}
        
        # 現在のパイプラインコード
        ```python
        {pipeline_code}
        ```
        
        # 改善の方向性（優先度順）
        1. パフォーマンスの向上（精度、再現率、F1スコアなどの改善）
        2. コードの効率化（不要な処理の削除、ベクトル化の適用など）
        3. 新しい特徴量の追加（ドメイン知識に基づく特徴量の追加）
        4. ハイパーパラメータの最適化（グリッドサーチやランダムサーチの適用）
        
        # 注意事項
        - コードは完全に実行可能な形式で提供してください
        - 変更点についての説明を具体的に記述してください
        - パフォーマンスの向上が期待できる根拠を説明してください
        - 応答は必ず以下のJSON形式で返してください
        - コードブロックはマークダウンのコードブロック形式で囲んでください
        
        # 応答フォーマット（以下のJSON形式で返してください）
        {{
            "suggestion": "具体的な改善点とその根拠をここに記述",
            "code": "改善された完全なPythonコードをここに記述.改行は\nで表現してください。絶対に改行は\nで表現してください。"
        }}
        
        あなたの回答は自動的に解析されるため、文字列の回答が正確に正しい形式であることを確認してください。

        上記のJSON形式で、suggestionとcodeの2つのキーを含むようにしてください。
        
        """.format(
            task_description=task_description,
            metrics=metrics_str,
            pipeline_code=current_pipeline_code
        )
        
        # リトライとレート制限の設定            
        max_retries = 3
        base_delay = 10  # ベースの遅延（秒）
        max_delay = 60   # 最大遅延（秒）
        
        # リクエスト間の遅延（秒）
        time.sleep(5)  # ベースラインの遅延を追加
        
        for attempt in range(max_retries):
            try:
                # LLMにリクエストを送信
                messages = [
                    {"role": "system", "content": "あなたは熟練したデータサイエンティストです。正確で効率的なコードを生成してください。"},
                    {"role": "user", "content": prompt}
                ]
                
                logger.info(f"Sending request to {provider} (attempt {attempt + 1}/{max_retries})")
                
                # プロバイダーに応じたリクエスト送信
                start_time = time.time()
                try:
                    if provider == 'gemini':
                        # Geminiの場合はテキスト生成を使用
                        response = llm.generate_text(prompt)
                    else:
                        # Groqの場合はチャット形式を使用
                        response = llm.chat(messages)
                    
                    # レスポンスの処理時間を記録
                    elapsed = time.time() - start_time
                    logger.info(f"Received response from {provider} in {elapsed:.2f} seconds")
                    
                    # レスポンスが有効な場合はループを抜ける
                    if response:
                        if isinstance(response, str) and len(response) < 20:
                            logger.warning(f"Suspiciously short response: {response}")
                            raise ValueError("Response too short")
                        break
                        
                except Exception as api_error:
                    elapsed = time.time() - start_time
                    logger.error(f"API error after {elapsed:.2f} seconds: {str(api_error)}")
                    raise
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                # 最後の試行でない場合のみリトライ
                if attempt < max_retries - 1:
                    # 指数バックオフ + ジッターを追加
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 5), max_delay)
                    logger.info(f"Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error("Max retries reached. Giving up.")
                    raise
        
        # レート制限を避けるための追加の遅延
        time.sleep(2)
        
        # 生のレスポンスを表示
        print("\n" + "="*80)
        print("LLM Raw Response:")
        print("-"*40)
        print(response)
        print("="*80 + "\n")
        
        # 応答をパース
        code = extract_suggestion_from_llm_output(response)
        
        return {
            'suggestion': '',
            'code': code if code else current_pipeline_code,
            'model': model_name,
            'provider': provider
        }
        
    except Exception as e:
        logger.error(f"改善案の生成中にエラーが発生しました: {e}")
        logger.error(traceback.format_exc())
        return {
            'suggestion': f"エラーが発生しました: {str(e)}",
            'code': current_pipeline_code,
            'model': 'error',
            'provider': 'error'
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