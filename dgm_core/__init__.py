"""
DGM (Darwin Gödel Machine) コアモジュール

このモジュールは、自己改善型機械学習パイプラインを実現するためのコアコンポーネントを提供します。
"""

from .agent import MachineLearningPipelineAgent
from .archive import PipelineArchive
from .evolution import DGMEvolution
from .templates import generate_initial_pipeline

__all__ = [
    'MachineLearningPipelineAgent',
    'PipelineArchive', 
    'DGMEvolution',
    'generate_initial_pipeline'
]
