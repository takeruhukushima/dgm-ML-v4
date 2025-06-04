#!/usr/bin/env python3
"""
DGM (Darwin Gödel Machine) のメイン実行スクリプト。
コマンドラインからDGMを実行するためのエントリーポイントです。
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import yaml
from datetime import datetime

# Add the project root to sys.path to allow importing dgm_core
# This is necessary because the script is in a subdirectory (scripts/)
# and needs to access modules in the parent directory.
_current_file_path = os.path.abspath(__file__)
_scripts_dir = os.path.dirname(_current_file_path)
_project_root = os.path.abspath(os.path.join(_scripts_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dgm_core import (
    MachineLearningPipelineAgent,
    PipelineArchive,
    DGMEvolution,
    generate_initial_pipeline
)


def setup_logging(config: dict, run_dir: Path) -> logging.Logger:
    """ロギングの設定"""
    log_config = config.get('logging', {})
    log_file = run_dir / 'dgm_run.log'
    
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('DGM')


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data(data_path: str, target_column: Optional[str] = None) -> pd.DataFrame:
    """データを読み込みます。
    
    Args:
        data_path (str): データファイルのパス
        target_column (Optional[str], optional): 目的変数のカラム名. Defaults to None.
        
    Returns:
        pd.DataFrame: 読み込まれたデータ
    """
    # ファイルの拡張子に応じて読み込み方法を変更
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # 目的変数が指定されている場合は、欠損値がある行を削除
    if target_column and target_column in df.columns:
        df = df.dropna(subset=[target_column])
    
    return df


def setup_output_dirs(output_dir: str) -> Path:
    """出力ディレクトリをセットアップします。
    
    Args:
        output_dir (str): ベース出力ディレクトリのパス
        
    Returns:
        Path: 実行用のタイムスタンプ付きディレクトリのパス
    """
    # ベースディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 実行用のタイムスタンプ付きディレクトリを作成
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリを作成
    (run_dir / 'pipelines').mkdir(exist_ok=True)
    (run_dir / 'results').mkdir(exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)
    
    return run_dir


def main():
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='DGM ML Pipeline Evolution')
    parser.add_argument('--task', type=str, required=True, help='Task name')
    parser.add_argument('--config', type=str, required=True, help='Task config path')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='出力ディレクトリ (デフォルト: output)')
    
    args = parser.parse_args()
    
    # 設定ファイルの読み込み
    task_config = load_config(args.config)
    global_config_path = Path(__file__).parent.parent / 'config' / 'global_config.yaml'
    global_config = load_config(global_config_path)
    
    # 出力ディレクトリの設定
    run_dir = setup_output_dirs(args.output_dir)
    
    # ロガーの設定
    logger = setup_logging(global_config, run_dir)
    
    try:
        logger.info("Starting DGM run for task: %s", args.task)
        
        # データの読み込み
        data_path = task_config['task']['data_path']
        target_column = task_config['task']['target_column']
        logger.info(f"Loading data from {data_path}...")
        df_train = load_data(data_path, target_column)
        logger.info(f"Loaded data shape: {df_train.shape}")
        
        # アーカイブの初期化
        archive = PipelineArchive(run_dir / 'pipelines' / 'archive')
        
        # DGM進化の実行
        dgm = DGMEvolution(
            task_config=task_config['task'],
            global_config=global_config,
            archive=archive
        )
        
        try:
            logger.info("""
            ========================================
            DGM Evolution を初期化しています...
            タスク: %s
            ========================================
            """, args.task)
            
            dgm_evolution = DGMEvolution(
                task_config=task_config['task'],
                global_config=global_config,
                archive=archive
            )
            
            logger.info("初期化が完了しました。データの準備を開始します...")
            
            # テストデータの読み込み
            test_data = None
            test_data_path = task_config.get('data', {}).get('test_path')
            if test_data_path:
                test_data = pd.read_csv(test_data_path)
                logger.info(f"Test data loaded: {test_data.shape}")
            
            best_agent = dgm.run(df_train)
            
            # 結果の保存
            if best_agent:
                best_pipeline_path = run_dir / 'pipelines' / 'best_pipeline.py'
                best_agent.save_to_file(str(best_pipeline_path))
                
                # ベストエージェントの情報を表示
                if best_agent and best_agent.performance_metrics:
                    best_metrics = best_agent.performance_metrics
                    
                    print("\n" + "="*50)
                    print("  最良パイプラインのパフォーマンスメトリクス")
                    print("="*50)
                    
                    # メトリクスが存在する場合のみ表示処理を実行
                    if best_metrics:
                        max_metric_length = max(len(metric) for metric in best_metrics.keys())
                        for metric, value in best_metrics.items():
                            print(f"  {metric.ljust(max_metric_length)}: {value:.4f}")
                        
                        print(f"\n  最良パイプラインは以下に保存されました:")
                        print(f"  {best_pipeline_path}")
                        print("="*50 + "\n")
                    else:
                        print("  メトリクスが計算されていません。")
                        print("="*50 + "\n")
                    
                    # 結果をJSONファイルに保存
                    results = {
                        'best_agent_id': best_agent.agent_id,
                        'best_metrics': best_metrics or {},  # 空の場合は空辞書を使用
                        'generation': best_agent.generation,
                        'parent_id': best_agent.parent_id,
                        'pipeline_file': str(best_pipeline_path.relative_to(run_dir))
                    }
                    
                    results_file = run_dir / 'results' / 'final_results.json'
                    with open(results_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2)
                        
                    logger.info(f"Final results saved to: {results_file}")
                else:
                    logger.warning("No successful pipeline was generated or no metrics were calculated.")
            
            # テストデータがあれば予測を実行し、latest_predictionsをセット
            if test_data is not None and best_agent is not None:
                best_agent.latest_predictions = best_agent.predict(test_data)

            # 予測結果の保存（latest_predictions属性があれば）
            if hasattr(best_agent, 'latest_predictions') and best_agent.latest_predictions is not None:
                prediction_path = run_dir / 'results' / 'prediction.csv'
                best_agent.latest_predictions.to_csv(prediction_path, index=True)
                logger.info(f"Prediction results saved to: {prediction_path}")
            
            logger.info("DGM evolution completed successfully!")
            return 0
            
        except Exception as e:
            logger.error(f"Error during DGM evolution: {e}", exc_info=True)
            return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())