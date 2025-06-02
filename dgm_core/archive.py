"""
生成されたパイプラインを管理するアーカイブクラスを定義するモジュール。
パイプラインの保存、読み込み、検索、管理機能を提供します。
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from .agent import MachineLearningPipelineAgent


class PipelineArchive:
    """生成されたパイプラインを管理するアーカイブクラス。
    
    パイプラインの保存、読み込み、検索、管理機能を提供します。
    パフォーマンスや多様性に基づいてパイプラインを選択する機能も提供します。
    """
    
    def __init__(self, archive_dir: str = "pipelines/archive", size_limit: int = 100):
        """アーカイブを初期化します。
        
        Args:
            archive_dir (str, optional): パイプラインを保存するディレクトリ. Defaults to "pipelines/archive".
            size_limit (int, optional): アーカイブの最大サイズ. Defaults to 100.
        """
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.size_limit = size_limit
        self.metadata_file = self.archive_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """メタデータをファイルから読み込みます。
        
        Returns:
            Dict[str, Any]: メタデータの辞書
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'agents': [],
            'next_agent_id': 1,
            'performance_history': {},
            'diversity_metrics': {}
        }
    
    def _save_metadata(self) -> None:
        """メタデータをファイルに保存します。"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def add_agent(self, agent: 'MachineLearningPipelineAgent', performance: Dict[str, float]) -> bool:
        """新しいエージェントをアーカイブに追加します。
        
        Args:
            agent (MachineLearningPipelineAgent): 追加するエージェント
            performance (Dict[str, float]): エージェントのパフォーマンスメトリクス
            
        Returns:
            bool: 追加に成功したかどうか
        """
        # 既存のエージェントと比較して、十分に優れているか確認
        if not self._is_worthy_of_archive(agent, performance):
            return False
        
        # エージェントIDがなければ割り当てる
        if not agent.agent_id:
            agent.agent_id = f"agent_{self.metadata['next_agent_id']}"
            self.metadata['next_agent_id'] += 1
        
        # エージェントをファイルに保存
        agent_file = self.archive_dir / f"{agent.agent_id}.py"
        agent.save_to_file(agent_file)
        
        # メタデータを更新
        agent_info = {
            'id': agent.agent_id,
            'generation': agent.generation,
            'parent_id': agent.parent_id,
            'performance': performance,
            'filepath': str(agent_file.relative_to(self.archive_dir))
        }
        
        # 既存のエージェント情報を更新（もしあれば）
        existing_idx = next((i for i, a in enumerate(self.metadata['agents']) 
                           if a['id'] == agent.agent_id), -1)
        
        if existing_idx >= 0:
            self.metadata['agents'][existing_idx] = agent_info
        else:
            # アーカイブが一杯の場合は、最もスコアの低いエージェントを削除
            if len(self.metadata['agents']) >= self.size_limit:
                self._remove_worst_agent()
            self.metadata['agents'].append(agent_info)
        
        # パフォーマンス履歴を更新
        self._update_performance_history(agent.agent_id, performance)
        
        # 多様性メトリクスを更新
        self._update_diversity_metrics(agent)
        
        # メタデータを保存
        self._save_metadata()
        return True
    
    def _is_worthy_of_archive(self, agent: 'MachineLearningPipelineAgent', 
                            performance: Dict[str, float]) -> bool:
        """エージェントがアーカイブに追加する価値があるかどうかを判定します。
        
        Args:
            agent (MachineLearningPipelineAgent): 評価するエージェント
            performance (Dict[str, float]): エージェントのパフォーマンスメトリクス
            
        Returns:
            bool: アーカイブに追加する価値がある場合はTrue
        """
        if not self.metadata['agents']:
            return True  # アーカイブが空の場合は常に追加
        
        # 主要メトリクスを取得（ここでは最初のメトリクスを使用）
        if not performance:
            return False
            
        primary_metric = next(iter(performance))
        primary_score = performance[primary_metric]
        
        # 既存のエージェントのスコアを取得
        existing_scores = [a['performance'].get(primary_metric, -float('inf')) 
                          for a in self.metadata['agents']]
        
        # 最低でも上位N%に入るか、最高スコアを上回る場合に追加
        threshold = np.percentile(existing_scores, 50)  # 中央値以上のスコアが必要
        return primary_score >= threshold or primary_score >= max(existing_scores)
    
    def _remove_worst_agent(self) -> None:
        """最もスコアの低いエージェントをアーカイブから削除します。"""
        if not self.metadata['agents']:
            return
        
        # 主要メトリクスでソート（最初のメトリクスを使用）
        sorted_agents = sorted(self.metadata['agents'], 
                              key=lambda x: list(x['performance'].values())[0] if x['performance'] else -float('inf'))
        
        # 最もスコアの低いエージェントを削除
        worst_agent = sorted_agents[0]
        agent_file = self.archive_dir / worst_agent['filepath']
        
        # ファイルを削除
        if agent_file.exists():
            try:
                os.unlink(agent_file)
            except:
                pass
        
        # メタデータから削除
        self.metadata['agents'] = [a for a in self.metadata['agents'] 
                                 if a['id'] != worst_agent['id']]
        
        # パフォーマンス履歴からも削除
        if worst_agent['id'] in self.metadata['performance_history']:
            del self.metadata['performance_history'][worst_agent['id']]
    
    def _update_performance_history(self, agent_id: str, performance: Dict[str, float]) -> None:
        """パフォーマンス履歴を更新します。
        
        Args:
            agent_id (str): エージェントID
            performance (Dict[str, float]): パフォーマンスメトリクス
        """
        if 'performance_history' not in self.metadata:
            self.metadata['performance_history'] = {}
        
        if agent_id not in self.metadata['performance_history']:
            self.metadata['performance_history'][agent_id] = []
        
        self.metadata['performance_history'][agent_id].append({
            'generation': len(self.metadata['performance_history'][agent_id]) + 1,
            'metrics': performance,
            'timestamp': str(datetime.now())
        })
    
    def _update_diversity_metrics(self, agent: 'MachineLearningPipelineAgent') -> None:
        """多様性メトリクスを更新します。
        
        Args:
            agent (MachineLearningPipelineAgent): 評価するエージェント
        """
        # ここではシンプルに特徴量の使用状況を記録
        # 実際の実装では、より洗練された多様性メトリクスを使用する
        if 'diversity_metrics' not in self.metadata:
            self.metadata['diversity_metrics'] = {}
        
        # 特徴量の使用状況を解析（シンプルな実装）
        features_used = set()
        for line in agent.pipeline_code.split('\n'):
            if 'drop' in line and 'columns' in line:
                # ドロップされた特徴量を除外するロジックを追加
                pass
            elif 'X[' in line or 'X.' in line:
                # 特徴量が使用されている行を解析
                # 実際の実装ではより正確な構文解析が必要
                features_used.update([f for f in agent.feature_importance.keys() if f in line])
        
        self.metadata['diversity_metrics'][agent.agent_id] = {
            'features_used': list(features_used),
            'num_features_used': len(features_used)
        }
    
    def get_best_agents(self, n: int = 1, metric: str = None) -> List[Dict[str, Any]]:
        """指定されたメトリクスで上位のエージェントを取得します。
        
        Args:
            n (int, optional): 取得するエージェントの数. Defaults to 1.
            metric (str, optional): ソートに使用するメトリクス. 指定しない場合は最初のメトリクスを使用.
            
        Returns:
            List[Dict[str, Any]]: エージェント情報のリスト
        """
        if not self.metadata['agents']:
            return []
        
        # メトリクスが指定されていない場合は最初のメトリクスを使用
        if not metric and self.metadata['agents']:
            metric = next(iter(self.metadata['agents'][0]['performance']))
        
        # メトリクスでソート
        sorted_agents = sorted(
            self.metadata['agents'],
            key=lambda x: x['performance'].get(metric, -float('inf')),
            reverse=True
        )
        
        return sorted_agents[:n]
    
    def get_diverse_agents(self, n: int = 1, strategy: str = 'feature_diversity') -> List[Dict[str, Any]]:
        """多様なエージェントを取得します。
        
        Args:
            n (int, optional): 取得するエージェントの数. Defaults to 1.
            strategy (str, optional): 多様性戦略. Defaults to 'feature_diversity'.
            
        Returns:
            List[Dict[str, Any]]: エージェント情報のリスト
        """
        if not self.metadata['agents']:
            return []
        
        if strategy == 'random':
            # ランダムに選択
            import random
            return random.sample(self.metadata['agents'], min(n, len(self.metadata['agents'])))
        
        elif strategy == 'feature_diversity':
            # 特徴量の多様性に基づいて選択
            # 実装を簡略化しています。実際には、特徴量の使用状況に基づいてクラスタリングなどを行うと良いでしょう。
            agents = self.metadata['agents']
            if len(agents) <= n:
                return agents
                
            # 特徴量の使用状況に基づいて多様性を計算
            feature_counts = {}
            for agent in agents:
                features = set(self.metadata['diversity_metrics'].get(agent['id'], {}).get('features_used', []))
                for f in features:
                    feature_counts[f] = feature_counts.get(f, 0) + 1
            
            # レアな特徴量を使用しているエージェントを優先
            def diversity_score(agent_id):
                features = set(self.metadata['diversity_metrics'].get(agent_id, {}).get('features_used', []))
                return sum(1.0 / (feature_counts.get(f, 1)) for f in features)
            
            # 多様性スコアでソート
            sorted_agents = sorted(
                agents,
                key=lambda x: diversity_score(x['id']),
                reverse=True
            )
            
            return sorted_agents[:n]
        
        else:
            # デフォルトではパフォーマンスでソート
            return self.get_best_agents(n)
    
    def load_agent(self, agent_id: str) -> Optional['MachineLearningPipelineAgent']:
        """指定されたIDのエージェントをファイルから読み込みます。
        
        Args:
            agent_id (str): 読み込むエージェントのID
            
        Returns:
            Optional[MachineLearningPipelineAgent]: 読み込まれたエージェント、見つからない場合はNone
        """
        agent_info = next((a for a in self.metadata['agents'] if a['id'] == agent_id), None)
        if not agent_info:
            return None
        
        agent_file = self.archive_dir / agent_info['filepath']
        if not agent_file.exists():
            return None
        
        try:
            agent = MachineLearningPipelineAgent.load_from_file(agent_file)
            return agent
        except Exception as e:
            print(f"Error loading agent {agent_id}: {e}")
            return None
    
    def get_agent_performance_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """エージェントのパフォーマンス履歴を取得します。
        
        Args:
            agent_id (str): エージェントID
            
        Returns:
            List[Dict[str, Any]]: パフォーマンス履歴のリスト
        """
        return self.metadata.get('performance_history', {}).get(agent_id, [])
    
    def get_archive_summary(self) -> Dict[str, Any]:
        """アーカイブのサマリー情報を返します。
        
        Returns:
            Dict[str, Any]: アーカイブのサマリー情報
        """
        if not self.metadata['agents']:
            return {
                'num_agents': 0,
                'best_performance': {},
                'diversity_metrics': {}
            }
        
        # 主要メトリクスを取得
        metrics = list(self.metadata['agents'][0]['performance'].keys())
        best_performance = {}
        
        for metric in metrics:
            best_agent = max(self.metadata['agents'], 
                           key=lambda x: x['performance'].get(metric, -float('inf')))
            best_performance[metric] = best_agent['performance'][metric]
        
        # 多様性メトリクスを計算
        all_features = set()
        for agent_id, metrics in self.metadata.get('diversity_metrics', {}).items():
            all_features.update(metrics.get('features_used', []))
        
        return {
            'num_agents': len(self.metadata['agents']),
            'best_performance': best_performance,
            'diversity_metrics': {
                'unique_features': len(all_features),
                'avg_features_per_agent': np.mean([
                    m.get('num_features_used', 0) 
                    for m in self.metadata.get('diversity_metrics', {}).values()
                ]) if self.metadata.get('diversity_metrics') else 0
            }
        }
    
    def save_predictions(self, agent_id: str, predictions: pd.Series, timestamp: Optional[str] = None) -> None:
        """予測結果を保存

        Args:
            agent_id (str): エージェントID
            predictions (pd.Series): 予測結果
            timestamp (str, optional): タイムスタンプ
        """
        predictions_dir = self.archive_dir / 'predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        # ファイル名の生成
        timestamp = timestamp or pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{agent_id}_{timestamp}_predictions.csv"
        
        # 予測結果の保存
        predictions.to_csv(predictions_dir / filename)
        
        # メタデータの更新
        if not hasattr(self, 'predictions_metadata'):
            self.predictions_metadata = {}
            
        self.predictions_metadata[agent_id] = {
            'latest_prediction_file': filename,
            'timestamp': timestamp,
            'num_predictions': len(predictions)
        }
        
        # メタデータをJSONに保存
        self._save_metadata()
