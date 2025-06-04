import os
import json
from typing import Dict, Any, Optional, List, Union
from openai import OpenAI
from dotenv import load_dotenv
import yaml

class GroqClient:
    """Groq APIを使用してQwen-32Bモデルと通信するためのクライアントクラス"""
    
    def __init__(self, config_path: str = "config/initial_strategy.yaml"):
        """
        GroqClientの初期化
        
        Args:
            config_path (str): 設定ファイルのパス
        """
        # 設定ファイルを読み込む
        self.config = self._load_config(config_path)
        
        # 環境変数を読み込む
        load_dotenv()
        
        # APIキーを取得
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEYが環境変数に設定されていません。.envファイルを確認してください。")
        
        # OpenAIクライアントを初期化
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key,
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"設定ファイルの読み込み中にエラーが発生しました: {e}")
            return {}
    
    def generate_pipeline(
        self, 
        task_description: str, 
        target_variable: str, 
        feature_columns: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        機械学習パイプラインを生成する
        
        Args:
            task_description (str): タスクの説明
            target_variable (str): ターゲット変数名
            feature_columns (List[str]): 特徴量のリスト
            **kwargs: その他のパラメータ
            
        Returns:
            Dict[str, Any]: 生成されたパイプラインの情報
        """
        # プロンプトテンプレートを取得
        prompt_template = self.config.get(
            'initial_prompt_template', 
            'タスク: {task_description}\nターゲット変数: {target_variable}\n特徴量: {feature_columns}'
        )
        
        # プロンプトをフォーマット
        prompt = prompt_template.format(
            task_description=task_description,
            target_variable=target_variable,
            feature_columns=", ".join(feature_columns),
            pipeline_code_placeholder="# ここにパイプラインコードを生成してください"
        )
        
        # モデルにリクエストを送信
        response = self._query_llm(prompt)
        
        # 応答からコードを抽出
        code_blocks = self._extract_code_blocks(response)
        pipeline_code = code_blocks[0] if code_blocks else ""
        
        return {
            'pipeline_code': pipeline_code,
            'response': response,
            'model': self.config.get('llm_model', 'qwen-qwq-32b'),
            'config': self.config
        }
    
    def improve_pipeline(
        self, 
        task_description: str, 
        target_variable: str, 
        current_pipeline_code: str,
        current_performance: Dict[str, float],
        **kwargs
    ) -> Dict[str, Any]:
        """
        既存のパイプラインを改善する
        
        Args:
            task_description (str): タスクの説明
            target_variable (str): ターゲット変数名
            current_pipeline_code (str): 現在のパイプラインコード
            current_performance (Dict[str, float]): 現在のパフォーマンスメトリクス
            **kwargs: その他のパラメータ
            
        Returns:
            Dict[str, Any]: 改善されたパイプラインの情報
        """
        # プロンプトテンプレートを取得
        prompt_template = self.config.get(
            'improvement_prompt_template',
            'タスク: {task_description}\nターゲット変数: {target_variable}\n現在のパフォーマンス: {current_performance}\n\n現在のコード:\n```python\n{current_pipeline_code}\n```\n\n改善案を提案してください。'
        )
        
        # パフォーマンスを文字列に変換
        perf_str = ", ".join([f"{k}: {v:.4f}" for k, v in current_performance.items()])
        
        # プロンプトをフォーマット
        prompt = prompt_template.format(
            task_description=task_description,
            target_variable=target_variable,
            current_performance=perf_str,
            current_pipeline_code=current_pipeline_code,
            improved_pipeline_code_placeholder="# ここに改善されたコードを生成してください"
        )
        
        # モデルにリクエストを送信
        response = self._query_llm(prompt)
        
        # 応答からコードを抽出
        code_blocks = self._extract_code_blocks(response)
        improved_code = code_blocks[0] if code_blocks else current_pipeline_code
        
        return {
            'improved_pipeline_code': improved_code,
            'response': response,
            'model': self.config.get('llm_model', 'qwen-qwq-32b'),
            'config': self.config
        }
    
    def _query_llm(self, prompt: str) -> str:
        """LLMにクエリを送信する内部メソッド"""
        try:
            # チャット補完リクエストを送信
            completion = self.client.chat.completions.create(
                model=self.config.get('llm_model', 'qwen-qwq-32b'),
                messages=[
                    {
                        "role": "system",
                        "content": self.config.get('task_specific_hints', 'あなたは熟練したデータサイエンティストです。正確で効率的なコードを生成してください。')
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=float(self.config.get('temperature', 0.7)),
                max_tokens=int(self.config.get('max_tokens', 2048)),
                top_p=float(self.config.get('top_p', 1.0)),
                frequency_penalty=float(self.config.get('frequency_penalty', 0.0)),
                presence_penalty=float(self.config.get('presence_penalty', 0.0)),
                stream=False
            )
            
            # 応答を返す
            return completion.choices[0].message.content
            
        except Exception as e:
            return f"エラーが発生しました: {str(e)}"
    
    @staticmethod
    def _extract_code_blocks(text: str) -> List[str]:
        """マークダウンからコードブロックを抽出する"""
        import re
        pattern = r'```(?:python\n)?(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]

# ユーティリティ関数
def query_groq(prompt: str, model: str = "qwen-qwq-32b") -> str:
    """
    Groq APIを使用してQwen-32Bモデルにクエリを送信する簡易関数
    
    Args:
        prompt (str): ユーザーのプロンプト
        model (str): 使用するモデル（デフォルトはqwen-qwq-32b）
        
    Returns:
        str: モデルの応答
    """
    client = GroqClient()
    return client._query_llm(prompt)

if __name__ == "__main__":
    # テスト用のコード
    import argparse
    
    parser = argparse.ArgumentParser(description='Groq Qwen-32B クライアント')
    parser.add_argument('--prompt', type=str, help='プロンプトを指定')
    parser.add_argument('--task', type=str, help='タスクの説明')
    parser.add_argument('--target', type=str, help='ターゲット変数')
    parser.add_argument('--features', type=str, help='カンマ区切りの特徴量リスト')
    
        client = GroqClient()
        
        # 1. シンプルなチャットのテスト
        print("\n--- シンプルなチャットのテスト ---")
        messages = [
            {"role": "system", "content": "あなたは親切なAIアシスタントです。簡潔に答えてください。"},
            {"role": "user", "content": "こんにちは、あなたは誰ですか？"}
        ]
        response = client.chat(messages)
        print(f"応答: {response}")
        
        # 2. パイプライン生成のテスト
        print("\n--- パイプライン生成のテスト ---")
        task_description = "タイタニック号の生存予測モデルを作成してください。"
        target_variable = "Survived"
        feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        
        pipeline_result = client.generate_pipeline(
            task_description=task_description,
            target_variable=target_variable,
            feature_columns=feature_columns
        )
        
        print("\n生成されたパイプライン:")
        print(pipeline_result.get('pipeline_code', 'パイプラインの生成に失敗しました。'))
        
        # 3. パイプライン改善のテスト
        print("\n--- パイプライン改善のテスト ---")
        current_pipeline = """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        def train_model(X, y):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            return model, X_test, y_test
        """
        
        current_performance = {"accuracy": 0.78, "precision": 0.75, "recall": 0.70}
        
        improvement_result = client.improve_pipeline(
            task_description=task_description,
            target_variable=target_variable,
            current_pipeline_code=current_pipeline,
            current_performance=current_performance
        )
        
        print("\n改善案:")
        print(improvement_result.get('suggestion', '改善案の生成に失敗しました。'))
        print("\n改善されたパイプライン:")
        print(improvement_result.get('pipeline_code', 'パイプラインの改善に失敗しました。'))
        
        print("\n=== すべてのテストが完了しました ===")
        
    except Exception as e:
        print(f"\n=== エラーが発生しました ===")
        print(f"エラータイプ: {type(e).__name__}")
        print(f"エラーメッセージ: {str(e)}")
        print("\nトラブルシューティングのヒント:")
        print("1. インターネット接続を確認してください")
        print("2. GROQ_API_KEYが正しく設定されているか確認してください")
        print("3. クレジットが十分にあるか確認してください")
        print("4. レート制限に達していないか確認してください")

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # .envファイルから環境変数を読み込む
    load_dotenv()
    
    # テストを実行
    test_groq_client()
