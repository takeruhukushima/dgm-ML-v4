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
from .archive import PipelineArchive
from .templates import generate_initial_pipeline

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DGMEvolution:
    """パイプラインの進化的な改善を行うクラス。
    
    LLMを活用してパイプラインを改善し、アーカイブと連携して
    パイプラインの進化を管理します。
    """
    
    def __init__(self, task_config: dict, global_config: dict, archive: PipelineArchive):
        """DGMEvolutionを初期化します。
        
        Args:
            task_config (dict): タスク固有の設定
            global_config (dict): グローバルな設定
            archive: PipelineArchiveインスタンス
        """
        self.task_config = task_config
        self.global_config = global_config
        self.archive = archive
        self.evolution_config = global_config.get('evolution', {})
        self.current_generation = 0
        self.population: List[MachineLearningPipelineAgent] = []
        self.best_agent: Optional[MachineLearningPipelineAgent] = None
    
    def run(self, df_train: pd.DataFrame) -> Optional[MachineLearningPipelineAgent]:
        """進化プロセスを実行する
        
        Args:
            df_train (pd.DataFrame): 訓練データ
            
        Returns:
            Optional[MachineLearningPipelineAgent]: 最良のエージェント
        """
        logger.info("Starting evolution process...")
        
        # 初期集団の生成
        self._initialize_population()
        
        # 進化のメインループ
        for generation in range(self.evolution_config.get('generations', 10)):
            self.current_generation = generation
            logger.info(f"\n[世代 {generation + 1}/{self.evolution_config.get('generations', 10)}] 進化中...")
            
            # 各エージェントの評価
            for i, agent in enumerate(self.population):
                logger.info(f"\n  個体 {i + 1}/{len(self.population)} を評価中...")
                # タスク設定とグローバル設定の両方を渡す
                score = agent.run_and_evaluate(
                    df_train, 
                    self.task_config,
                    self.global_config
                )
                
                # 最良エージェントの更新
                if not self.best_agent or score > self.best_agent.get_best_score():
                    self.best_agent = agent
                    logger.info(f"New best score: {score:.4f} (Agent: {agent.agent_id})")
            
            # 次世代の生成
            if generation < self.evolution_config.get('generations', 10) - 1:
                self._generate_next_generation()
        
        return self.best_agent
    
    def _initialize_population(self):
        """初期集団を生成"""
        population_size = self.evolution_config.get('population_size', 3)
        
        # 初期パイプラインコードの生成
        initial_pipeline = generate_initial_pipeline(self.task_config)
        
        # 初期集団の生成
        self.population = [
            MachineLearningPipelineAgent(
                pipeline_code=initial_pipeline,
                generation=0
            )
            for _ in range(population_size)
        ]
        
        logger.info(f"Initialized population with {len(self.population)} agents")
    
    def _generate_next_generation(self):
        """次世代の個体を生成"""
        new_population = []
        
        # エリート戦略: 最良個体を次世代に残す
        if self.best_agent:
            new_population.append(self.best_agent)
        
        # 残りの個体を生成
        while len(new_population) < self.evolution_config.get('population_size', 3):
            # ランダムに親を選択
            parent = self._select_parent()
            
            # 親から新しい個体を生成
            child = self._generate_improved_agent(parent)
            if child:
                new_population.append(child)
        
        self.population = new_population
    
    def _select_parent(self) -> MachineLearningPipelineAgent:
        """親個体を選択"""
        # 現状はランダムに選択
        import random
        return random.choice(self.population)
    
    def _generate_improved_agent(self, parent: MachineLearningPipelineAgent) -> Optional[MachineLearningPipelineAgent]:
        """親個体から改善された子個体を生成"""
        try:
            # LLMを使用して改善案を生成
            new_agent = parent.self_improve(self.task_config, self.global_config)
            return new_agent
        except Exception as e:
            logger.error(f"Error generating improved agent: {e}")
            return None
