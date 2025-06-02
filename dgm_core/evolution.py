"""
パイプラインの進化的な改善を行うモジュール。
LLMを活用してパイプラインを改善する機能を提供します。
"""

import os
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from .agent import MachineLearningPipelineAgent
from .llm_utils import generate_improvement_suggestion, parse_llm_suggestion

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DGMEvolution:
    """パイプラインの進化的な改善を行うクラス。
    
    LLMを活用してパイプラインを改善し、アーカイブと連携して
    パイプラインの進化を管理します。
    """
    
    def __init__(self, task_config: dict, global_config: dict, archive):
        """DGMEvolutionを初期化します。
        
        Args:
            task_config (dict): タスク固有の設定
            global_config (dict): グローバルな設定
            archive: PipelineArchiveインスタンス
        """
        self.task_config = task_config
        self.global_config = global_config
        self.archive = archive
        self.current_generation = 0
        self.best_agent = None
        self.best_score = -float('inf')
        self.history = []
    
    def initialize_population(self, initial_pipeline_code: str, population_size: int = 3) -> List[MachineLearningPipelineAgent]:
        """初期集団を生成します。
        
        Args:
            initial_pipeline_code (str): 初期パイプラインコード
            population_size (int, optional): 初期集団のサイズ. Defaults to 3.
            
        Returns:
            List[MachineLearningPipelineAgent]: 初期集団のエージェントリスト
        """
        population = []
        
        # ベースラインのパイプラインを追加
        base_agent = MachineLearningPipelineAgent(
            pipeline_code=initial_pipeline_code,
            generation=0,
            parent_id=None
        )
        population.append(base_agent)
        
        # ランダムなバリエーションを追加（シード値やパラメータを変えるなど）
        for i in range(1, population_size):
            # ここではシード値を変えたバリエーションを作成
            # 実際の実装では、より洗練されたバリエーション生成ロジックを使用
            variant_code = initial_pipeline_code.replace(
                "random_state=42", 
                f"random_state={42 + i}"
            )
            
            variant_agent = MachineLearningPipelineAgent(
                pipeline_code=variant_code,
                generation=0,
                parent_id=base_agent.agent_id
            )
            population.append(variant_agent)
        
        return population
    
    def select_parent_agent(self) -> MachineLearningPipelineAgent:
        """親エージェントを選択します。
        
        トーナメント選択やルーレット選択などの方法で親を選択します。
        
        Returns:
            MachineLearningPipelineAgent: 選択された親エージェント
        """
        # アーカイブからエージェントを取得
        if not self.archive.metadata['agents']:
            raise ValueError("No agents in archive to select from")
        
        # トーナメント選択を実装
        tournament_size = min(5, len(self.archive.metadata['agents']))
        candidates = random.sample(self.archive.metadata['agents'], tournament_size)
        
        # 主要メトリクスで最良のエージェントを選択
        primary_metric = self.task_config.get('evaluation_metric', 'accuracy')
        best_agent_info = max(candidates, key=lambda x: x['performance'].get(primary_metric, -float('inf')))
        
        # エージェントをロードして返す
        return self.archive.load_agent(best_agent_info['id'])
    
    def self_improve_agent(self, parent_agent: MachineLearningPipelineAgent, current_generation: int) -> Optional[MachineLearningPipelineAgent]:
        """親エージェントを自己改善して子エージェントを生成します。
        
        Args:
            parent_agent (MachineLearningPipelineAgent): 親エージェント
            current_generation (int): 現在の世代数
            
        Returns:
            Optional[MachineLearningPipelineAgent]: 生成された子エージェント、失敗時はNone
        """
        try:
            # 親エージェントの情報を収集
            parent_summary = parent_agent.get_summary()
            parent_performance = parent_summary.get('performance_metrics', {})
            
            # タスクの説明を準備
            task_description = self.task_config.get('description', '')
            
            # LLMに改善案を提案させる
            suggestion = generate_improvement_suggestion(
                task_description=task_description,
                current_pipeline_code=parent_agent.pipeline_code,
                performance_metrics=parent_performance,
                task_config=self.task_config,
                global_config=self.global_config
            )
            
            # LLMの応答をパース
            parsed_suggestion = parse_llm_suggestion(suggestion)
            
            if not parsed_suggestion or 'improved_code' not in parsed_suggestion:
                logger.warning("Failed to parse LLM suggestion or no improvement code provided")
                return None
            
            # 新しいエージェントを作成
            child_agent = MachineLearningPipelineAgent(
                pipeline_code=parsed_suggestion['improved_code'],
                generation=current_generation,
                parent_id=parent_agent.agent_id
            )
            
            # 改善の説明をログに記録
            improvement_desc = parsed_suggestion.get('improvement_description', 'No description provided')
            logger.info(f"Generated improvement for generation {current_generation}: {improvement_desc}")
            
            return child_agent
            
        except Exception as e:
            logger.error(f"Error in self_improve_agent: {e}", exc_info=True)
            return None
    
    def evaluate_agent(self, agent: MachineLearningPipelineAgent, df_train_full: pd.DataFrame, 
                       target_column: str) -> Tuple[float, Dict[str, float]]:
        """エージェントを評価し、スコアを返します。
        
        Args:
            agent (MachineLearningPipelineAgent): 評価するエージェント
            df_train_full (pd.DataFrame): 学習データ
            target_column (str): 目的変数のカラム名
            
        Returns:
            Tuple[float, Dict[str, float]]: (主要メトリクスのスコア, 全メトリクスの辞書)
        """
        try:
            # エージェントを実行して評価
            score = agent.run_and_evaluate(
                df_train_full=df_train_full,
                target_column=target_column,
                task_config=self.task_config,
                global_config=self.global_config
            )
            
            # ベストスコアを更新
            if score > self.best_score:
                self.best_score = score
                self.best_agent = agent
                logger.info(f"New best score: {score:.4f} (Agent: {agent.agent_id})")
            
            # 履歴に記録
            self.history.append({
                'generation': agent.generation,
                'agent_id': agent.agent_id,
                'score': score,
                'parent_id': agent.parent_id,
                'performance_metrics': agent.performance_metrics
            })
            
            # アーカイブに追加
            self.archive.add_agent(agent, agent.performance_metrics)
            
            return score, agent.performance_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating agent {agent.agent_id}: {e}", exc_info=True)
            return 0.0, {}
    
    def run_evolution(self, initial_pipeline_code: str, df_train_full: pd.DataFrame, 
                     target_column: str, num_generations: int = 10, 
                     population_size: int = 3) -> MachineLearningPipelineAgent:
        """進化的な改善を実行します。
        
        Args:
            initial_pipeline_code (str): 初期パイプラインコード
            df_train_full (pd.DataFrame): 学習データ
            target_column (str): 目的変数のカラム名
            num_generations (int, optional): 進化の世代数. Defaults to 10.
            population_size (int, optional): 各世代の個体数. Defaults to 3.
            
        Returns:
            MachineLearningPipelineAgent: 最良のエージェント
        """
        print("\n" + "="*70)
        print(f"  進化プロセスを開始します (最大{num_generations}世代, 1世代あたり{population_size}個体)")
        print("="*70 + "\n")
        
        logger.info("Starting DGM evolution...")
        
        try:
            # 初期集団を生成
            print("\n[初期化] 初期集団を生成しています...")
            population = self.initialize_population(initial_pipeline_code, population_size)
            
            # 初期集団を評価
            print(f"\n[評価] 初期集団を評価しています ({len(population)}個体)...")
            for i, agent in enumerate(population, 1):
                score, metrics = self.evaluate_agent(agent, df_train_full, target_column)
                primary_metric = self.task_config.get('evaluation_metric', 'accuracy')
                print(f"  個体 {i}/{len(population)} - {agent.agent_id[:8]}...: {primary_metric} = {score:.4f}")
            
            # 世代ごとの進化
            print("\n" + "="*70)
            print("  進化を開始します")
            print("="*70 + "\n")
            
            for gen in range(1, num_generations + 1):
                self.current_generation = gen
                print(f"\n[世代 {gen}/{num_generations}] 進化中...")
                
                new_population = []
                
                # 各親エージェントに対して子を生成
                for i in range(population_size):
                    print(f"\n  個体 {i+1}/{population_size} を生成中...")
                    
                    # 親を選択
                    parent = self.select_parent_agent()
                    print(f"    親: {parent.agent_id[:8]}... (スコア: {parent.performance_metrics.get(self.task_config.get('evaluation_metric', 'accuracy'), 0):.4f})")
                    
                    # 子を生成
                    print("    LLMを使用して改善案を生成中...")
                    child = self.self_improve_agent(parent, gen)
                    
                    if child:
                        print(f"    新しい個体を生成: {child.agent_id[:8]}...")
                        score, metrics = self.evaluate_agent(child, df_train_full, target_column)
                        print(f"    評価完了: {self.task_config.get('evaluation_metric', 'accuracy')} = {score:.4f}")
                        new_population.append(child)
                    else:
                        print("    個体の生成に失敗しました")
                
                # 新しい世代で集団を更新
                if new_population:
                    population = new_population
                
                # 世代のサマリーを表示
                best_in_gen = max(population, key=lambda x: x.performance_metrics.get(self.task_config.get('evaluation_metric', 'accuracy'), 0))
                best_score = best_in_gen.performance_metrics.get(self.task_config.get('evaluation_metric', 'accuracy'), 0)
                
                print("\n" + "-"*50)
                print(f"  [世代 {gen}/{num_generations}] サマリー")
                print("  " + "-"*46)
                print(f"  最良スコア: {best_score:.4f} (個体: {best_in_gen.agent_id[:8]}...)")
                print("  メトリクス:")
                for metric, value in best_in_gen.performance_metrics.items():
                    print(f"    - {metric}: {value:.4f}")
                print("-"*50 + "\n")
            
            # 最良のエージェントを見つける
            print("\n[完了] 進化が完了しました。最良の個体を検索中...")
            best_agent_info = max(
                self.archive.metadata['agents'], 
                key=lambda x: x['performance'].get(self.task_config.get('evaluation_metric', 'accuracy'), -float('inf'))
            )
            
            best_agent = self.archive.load_agent(best_agent_info['id'])
            best_score = best_agent_info['performance'].get(self.task_config.get('evaluation_metric', 'accuracy'), 0)
            
            print("\n" + "="*70)
            print(f"  進化が完了しました！最良の個体: {best_agent.agent_id}")
            print("="*70)
            print(f"  最終スコア ({self.task_config.get('evaluation_metric', 'accuracy')}): {best_score:.4f}")
            print("  メトリクス:")
            for metric, value in best_agent.performance_metrics.items():
                print(f"    - {metric}: {value:.4f}")
            print("="*70 + "\n")
            
            return best_agent
            
        except Exception as e:
            print("\n" + "!"*70)
            print(f"  エラーが発生しました: {str(e)}")
            print("!"*70 + "\n")
            logger.exception("Error in run_evolution")
            
            # エラーが発生した場合でも、これまでで最良のエージェントを返す
            if self.best_agent:
                print("\nエラーが発生しましたが、これまでで最良の個体を返します。")
                return self.best_agent
            else:
                print("\nエラーが発生し、有効な個体が見つかりませんでした。")
                return None
    
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """進化の履歴を返します。
        
        Returns:
            List[Dict[str, Any]]: 進化の履歴
        """
        return self.history
    
    def get_best_agent(self) -> Optional[MachineLearningPipelineAgent]:
        """これまでで最良のエージェントを返します。
        
        Returns:
            Optional[MachineLearningPipelineAgent]: 最良のエージェント、見つからない場合はNone
        """
        return self.best_agent
    
    def save_evolution_history(self, filepath: str) -> None:
        """進化の履歴をファイルに保存します。
        
        Args:
            filepath (str): 保存先のファイルパス
        """
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def _generate_improved_pipeline(self, parent_agent, generation: int) -> Optional[MachineLearningPipelineAgent]:
        """既存のパイプラインを改善した新しいパイプラインを生成"""
        try:
            # LLMに改善案を生成させる
            suggestion = generate_improvement_suggestion(
                task_description=self.task_config.get('description', ''),
                current_pipeline_code=parent_agent.pipeline_code,
                performance_metrics=parent_agent.performance_metrics,
                task_config=self.task_config,
                global_config=self.global_config
            )
            
            # 改善案をパース
            parsed = parse_llm_suggestion(suggestion)
            
            if not parsed['code']:
                self.logger.error("No valid code generated")
                return None
                
            # 新しいエージェントを生成
            new_agent = MachineLearningPipelineAgent(
                pipeline_code=parsed['code'],
                generation=generation,
                parent_id=parent_agent.agent_id
            )
            
            self.logger.info(f"Generated improvement for generation {generation}: {parsed['description']}")
            return new_agent
            
        except Exception as e:
            self.logger.error(f"Error generating improvement: {str(e)}")
            return None
