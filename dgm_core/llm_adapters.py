"""
DGMフレームワークと様々なLLMプロバイダーを統合するためのアダプターモジュール
"""
from typing import Dict, Any, Optional, List, Union
import os
import yaml
from abc import ABC, abstractmethod

class LLMAdapter(ABC):
    """LLMアダプターの抽象基底クラス"""
    
    def __init__(self, config_path: str = "config/initial_strategy.yaml"):
        """
        初期化
        
        Args:
            config_path (str): 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"設定ファイルの読み込み中にエラーが発生しました: {e}")
            return {}
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass


class GroqQwenAdapter(LLMAdapter):
    """GroqのQwen-32Bモデルを使用するアダプター"""
    
    def __init__(self, config_path: str = "config/initial_strategy.yaml"):
        """
        初期化
        
        Args:
            config_path (str): 設定ファイルのパス
        """
        super().__init__(config_path)
        from groq_client import GroqClient
        self.client = GroqClient(config_path=config_path)
    
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
        return self.client.generate_pipeline(
            task_description=task_description,
            target_variable=target_variable,
            feature_columns=feature_columns,
            **kwargs
        )
    
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
        return self.client.improve_pipeline(
            task_description=task_description,
            target_variable=target_variable,
            current_pipeline_code=current_pipeline_code,
            current_performance=current_performance,
            **kwargs
        )


def get_llm_adapter(provider: str = "groq", **kwargs) -> LLMAdapter:
    """
    LLMプロバイダーに応じたアダプターを取得する
    
    Args:
        provider (str): LLMプロバイダー名 (例: "groq", "openai" など)
        **kwargs: アダプターの初期化に渡す追加パラメータ
        
    Returns:
        LLMAdapter: 指定されたプロバイダーに対応するアダプターインスタンス
    """
    provider = provider.lower()
    
    if provider == "groq":
        return GroqQwenAdapter(**kwargs)
    # 他のプロバイダーを追加する場合はここに実装
    else:
        raise ValueError(f"サポートされていないLLMプロバイダーです: {provider}")


# 使用例
if __name__ == "__main__":
    # Groqアダプターの使用例
    adapter = get_llm_adapter("groq")
    
    # パイプライン生成の例
    result = adapter.generate_pipeline(
        task_description="タイタニック号の生存予測",
        target_variable="Survived",
        feature_columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    )
    
    print("生成されたパイプラインコード:")
    print(result['pipeline_code'])
    
    # パイプライン改善の例
    improved_result = adapter.improve_pipeline(
        task_description="タイタニック号の生存予測",
        target_variable="Survived",
        current_pipeline_code=result['pipeline_code'],
        current_performance={"accuracy": 0.78, "f1_score": 0.75}
    )
    
    print("\n改善されたパイプラインコード:")
    print(improved_result['improved_pipeline_code'])
