"""
機械学習パイプラインを表現するエージェントクラスを定義するモジュール。
各エージェントは、特定の機械学習パイプラインの実装とその評価結果を保持します。
"""

import os
import uuid
import tempfile
import importlib.util
import traceback
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np


class MachineLearningPipelineAgent:
    """機械学習パイプラインを表現するエージェントクラス。
    
    各エージェントは、特定の機械学習パイプラインの実装とその評価結果を保持します。
    エージェントは、パイプラインコードを実行し、その性能を評価する機能を提供します。
    """
    
    def __init__(self, pipeline_code: str, agent_id: str = None, generation: int = 0, parent_id: str = None):
        """エージェントを初期化します。
        
        Args:
            pipeline_code (str): パイプラインのPythonコード（文字列）
            agent_id (str, optional): エージェントの一意識別子. Defaults to None (自動生成).
            generation (int, optional): 世代数. Defaults to 0.
            parent_id (str, optional): 親エージェントのID. Defaults to None.
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.pipeline_code = pipeline_code  # Pythonコード文字列
        self.generation = generation
        self.parent_id = parent_id
        self.performance_metrics: Dict[str, float] = {}
        self.model = None
        self.feature_importance = None
        self.execution_metadata = {}
    
    def _execute_pipeline_code(self, df_train_full: pd.DataFrame, target_column: str, 
                             task_config: dict, global_config: dict) -> Tuple[float, dict]:
        """与えられたpipeline_codeを実行してモデルを学習し、検証セットで評価する。
        
        Args:
            df_train_full (pd.DataFrame): 学習データ（検証用に分割される）
            target_column (str): 目的変数のカラム名
            task_config (dict): タスク固有の設定
            global_config (dict): グローバルな設定
            
        Returns:
            Tuple[float, dict]: (主要メトリクスのスコア, 評価メトリクスの辞書)
        """
        print(f"\n[評価] エージェント {self.agent_id[:8]}... の評価を開始")
        
        # 一時ファイルにコードを書き出して実行
        temp_file_path = None
        temp_module_name = f"pipeline_{self.agent_id.replace('-', '_')}"
        
        # タイマー開始
        import time
        start_time = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8') as f:
                f.write(self.pipeline_code)
                temp_file_path = f.name
            
            # 動的にモジュールとして読み込む
            spec = importlib.util.spec_from_file_location(temp_module_name, temp_file_path)
            pipeline_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pipeline_module)
            
            # データの前処理
            print("  1/4 データの前処理を実行中...")
            if hasattr(pipeline_module, 'preprocess_data'):
                df_processed = pipeline_module.preprocess_data(df_train_full, task_config)
                print(f"    前処理完了: {df_processed.shape[0]} 行 × {df_processed.shape[1]} 列")
            else:
                df_processed = df_train_full.copy()
                print("    カスタム前処理はスキップされました")
            
            # 特徴量とターゲットを分離
            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column]
            
            # 学習データと検証データに分割
            print("  2/4 データを学習用と検証用に分割中...")
            split_ratio = task_config.get('validation_split', 0.2)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=split_ratio,
                random_state=task_config.get('random_seed', 42),
                stratify=y if task_config.get('stratify', True) else None
            )
            print(f"    学習データ: {X_train.shape[0]} サンプル, 検証データ: {X_val.shape[0]} サンプル")
            
            # 目的変数を適切な型に変換
            y_train = y_train.astype(int)
            y_val = y_val.astype(int)

            print("    学習データの形状:", X_train.shape)
            print("    検証データの形状:", X_val.shape)
            print("    目的変数のクラス:", np.unique(y_train))

            # モデルの学習（Optunaによるチューニング含む）
            print("  3/4 モデルの学習を開始...")
            if hasattr(pipeline_module, 'create_model_for_optuna'):
                print("    Optunaを使用したハイパーパラメータチューニングを実行中...")
                best_params = pipeline_module.create_model_for_optuna(X_train, y_train, X_val, y_val, task_config)
                print("    最適なパラメータが見つかりました。最終モデルをトレーニング中...")
                self.model = pipeline_module.train_final_model(X_train, y_train, best_params)
            else:
                # シンプルなモデルトレーニング
                print("    デフォルトパラメータでモデルをトレーニング中...")
                self.model = pipeline_module.train_model(X_train, y_train, task_config)
            
            # モデル情報を表示
            model_name = self.model.__class__.__name__
            print(f"    モデル: {model_name} のトレーニングが完了しました")
            
            # モデルの評価
            print("  4/4 モデルの評価を実行中...")
            if hasattr(pipeline_module, 'evaluate_model'):
                y_pred = self.model.predict(X_val)
                y_pred_proba = self.model.predict_proba(X_val)[:, 1] if hasattr(self.model, 'predict_proba') else None
                
                metrics = pipeline_module.evaluate_model(
                    self.model, X_val, y_val, y_pred, y_pred_proba, task_config
                )
            else:
                # デフォルトの評価
                y_pred = self.model.predict(X_val)
                metrics = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'f1_score': f1_score(y_val, y_pred, average='weighted')
                }
                
                if hasattr(self.model, 'predict_proba'):
                    y_pred_proba = self.model.predict_proba(X_val)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba)
            
            # 特徴量の重要度を取得（可能な場合）
            if hasattr(self.model, 'feature_importances_') and hasattr(X_train, 'columns'):
                self.feature_importance = dict(zip(X_train.columns, self.model.feature_importances_))
                print("    特徴量の重要度を計算しました")
            
            # メトリクスを保存
            self.performance_metrics = metrics
            
            # 主要メトリクスを取得
            primary_metric = task_config.get('evaluation_metric', 'accuracy')
            primary_score = metrics.get(primary_metric, 0.0)
            
            # 評価結果を表示
            elapsed_time = time.time() - start_time
            print("\n  ✓ 評価が完了しました！")
            print(f"  所要時間: {elapsed_time:.1f}秒")
            print("  評価メトリクス:")
            for metric, value in metrics.items():
                print(f"    - {metric}: {value:.4f}")
            
            return primary_score, metrics
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"\n  ✗ エラーが発生しました (経過時間: {elapsed_time:.1f}秒)")
            print(f"  エラータイプ: {type(e).__name__}")
            print(f"  エラーメッセージ: {str(e)}")
            traceback.print_exc()
            # エラー時は最低スコアを返す
            return 0.0, {task_config.get('evaluation_metric', 'accuracy'): 0.0}
        finally:
            # 一時ファイルを削除
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
    
    def run_and_evaluate(self, df_train_full: pd.DataFrame, target_column: str, 
                    task_config: dict, global_config: dict) -> float:
        """パイプラインを実行し評価する"""
        try:
            # パイプラインコードを実行してモデルを学習し、検証セットで評価する
            score, metrics = self._execute_pipeline_code(df_train_full, target_column, task_config, global_config)
            
            # _execute_pipeline_code で計算されたメトリクスを使用
            self.performance_metrics = metrics
            
            print(f"\nFinal metrics for agent {self.agent_id}:")
            for metric, value in self.performance_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            return score
        
        except Exception as e:
            print(f"Error executing pipeline for agent {self.agent_id}: {e}")
            traceback.print_exc()
            return 0.0
    
    def get_summary(self) -> dict:
        """エージェントのサマリー情報を返す
        
        Returns:
            dict: エージェントのメタデータとパフォーマンスメトリクス
        """
        return {
            'agent_id': self.agent_id,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance,
            'execution_metadata': self.execution_metadata
        }
    
    def save_to_file(self, filepath: str) -> None:
        """エージェントのパイプラインコードをファイルに保存する
        
        Args:
            filepath (str): 保存先のファイルパス
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Agent ID: {self.agent_id}\n")
            f.write(f"# Generation: {self.generation}\n")
            f.write(f"# Parent ID: {self.parent_id or 'None'}\n\n")
            f.write(self.pipeline_code)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'MachineLearningPipelineAgent':
        """ファイルからエージェントをロードする
        
        Args:
            filepath (str): 読み込むファイルのパス
            
        Returns:
            MachineLearningPipelineAgent: ロードされたエージェントインスタンス
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # メタデータをパース
        agent_id = None
        generation = 0
        parent_id = None
        code_lines = []
        
        for line in lines:
            if line.startswith('# Agent ID: '):
                agent_id = line.split(': ')[1].strip()
            elif line.startswith('# Generation: '):
                generation = int(line.split(': ')[1].strip())
            elif line.startswith('# Parent ID: '):
                parent_id = line.split(': ')[1].strip()
                if parent_id == 'None':
                    parent_id = None
            else:
                code_lines.append(line)
        
        pipeline_code = ''.join(code_lines)
        return cls(pipeline_code, agent_id, generation, parent_id)
