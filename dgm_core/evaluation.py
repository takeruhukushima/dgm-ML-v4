"""
パイプラインの評価に使用するユーティリティ関数を提供するモジュール。
"""

from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)

def evaluate_pipeline(pipeline, X: pd.DataFrame, y: pd.Series, 
                     task_type: str = 'classification',
                     cv: int = 5, 
                     metrics: List[str] = None) -> Dict[str, Union[float, List[float]]]:
    """パイプラインを評価します。
    
    Args:
        pipeline: 評価するパイプライン（scikit-learn互換のfit/predictメソッドを持つオブジェクト）
        X (pd.DataFrame): 特徴量
        y (pd.Series): ターゲット
        task_type (str, optional): タスクの種類（'classification' または 'regression'）. Defaults to 'classification'.
        cv (int, optional): クロスバリデーションの分割数. Defaults to 5.
        metrics (List[str], optional): 計算するメトリクスのリスト. 指定しない場合はタスクに適したデフォルト値を使用.
        
    Returns:
        Dict[str, Union[float, List[float]]]: 評価メトリクスの辞書
    """
    if metrics is None:
        if task_type == 'classification':
            metrics = ['accuracy', 'f1_weighted', 'roc_auc']
        else:  # regression
            metrics = ['mse', 'mae', 'r2']
    
    # クロスバリデーションで評価
    cv_scores = {}
    
    for metric in metrics:
        if metric in ['accuracy', 'roc_auc'] and task_type == 'classification':
            scoring = metric
        elif metric == 'f1_weighted' and task_type == 'classification':
            scoring = 'f1_weighted'
        elif metric == 'mse' and task_type == 'regression':
            scoring = 'neg_mean_squared_error'
        elif metric == 'mae' and task_type == 'regression':
            scoring = 'neg_mean_absolute_error'
        elif metric == 'r2' and task_type == 'regression':
            scoring = 'r2'
        else:
            continue
        
        # クロスバリデーションを実行
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) if task_type == 'classification' else cv
        
        try:
            scores = cross_val_score(
                pipeline, X, y, 
                cv=cv, 
                scoring=scoring,
                n_jobs=-1
            )
            
            # スコアを正の値に変換（回帰タスクの場合）
            if scoring.startswith('neg_'):
                scores = -scores
                
            cv_scores[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores.tolist()
            }
            
        except Exception as e:
            print(f"Warning: Could not calculate {metric}: {e}")
            cv_scores[metric] = {
                'mean': np.nan,
                'std': np.nan,
                'scores': []
            }
    
    # ホールドアウトセットでも評価（オプション）
    # ここでは実装を簡略化するため、クロスバリデーションの結果のみを返す
    
    return cv_scores

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: np.ndarray = None,
                     task_type: str = 'classification') -> Dict[str, float]:
    """予測結果からメトリクスを計算します。
    
    Args:
        y_true (np.ndarray): 真のラベル
        y_pred (np.ndarray): 予測ラベル
        y_pred_proba (np.ndarray, optional): クラス確率（分類タスクの場合）. Defaults to None.
        task_type (str, optional): タスクの種類（'classification' または 'regression'）. Defaults to 'classification'.
        
    Returns:
        Dict[str, float]: 計算されたメトリクスの辞書
    """
    metrics = {}
    
    if task_type == 'classification':
        # 分類タスクのメトリクス
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # マルチクラス分類の場合はaverageパラメータが必要
        average = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'
        
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # ROC-AUC（確率予測が利用可能な場合）
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) > 2:  # マルチクラス
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovr', average='weighted'
                    )
                    metrics['roc_auc_ovo'] = roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovo', average='weighted'
                    )
                else:  # バイナリ分類
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except Exception as e:
                print(f"Warning: Could not calculate ROC-AUC: {e}")
    
    else:  # 回帰タスクのメトリクス
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
    
    return metrics

def compare_pipelines(pipelines: List[Tuple[Any, str]], 
                     X: pd.DataFrame, 
                     y: pd.Series, 
                     task_type: str = 'classification',
                     cv: int = 5) -> Dict[str, Dict[str, Any]]:
    """複数のパイプラインを比較します。
    
    Args:
        pipelines (List[Tuple[Any, str]]): (パイプライン, 名前) のタプルのリスト
        X (pd.DataFrame): 特徴量
        y (pd.Series): ターゲット
        task_type (str, optional): タスクの種類（'classification' または 'regression'）. Defaults to 'classification'.
        cv (int, optional): クロスバリデーションの分割数. Defaults to 5.
        
    Returns:
        Dict[str, Dict[str, Any]]: 各パイプラインの評価結果
    """
    results = {}
    
    for pipeline, name in pipelines:
        try:
            print(f"Evaluating pipeline: {name}")
            scores = evaluate_pipeline(pipeline, X, y, task_type, cv)
            results[name] = {
                'scores': scores,
                'status': 'completed'
            }
        except Exception as e:
            print(f"Error evaluating pipeline {name}: {e}")
            results[name] = {
                'error': str(e),
                'status': 'failed'
            }
    
    return results

def get_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """特徴量の重要度を取得します。
    
    Args:
        model: 特徴量重要度を取得するモデル
        feature_names (List[str]): 特徴量名のリスト
        
    Returns:
        Dict[str, float]: 特徴量名をキー、重要度を値とする辞書
    """
    importance_dict = {}
    
    # モデルが特徴量重要度をサポートしているか確認
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_dict = dict(zip(feature_names, importances))
    elif hasattr(model, 'coef_'):
        # 線形モデルの場合（係数の絶対値で重要度を計算）
        coef = model.coef_
        if len(coef.shape) > 1:  # マルチクラス分類
            importances = np.mean(np.abs(coef), axis=0)
        else:  # バイナリ分類または回帰
            importances = np.abs(coef)
        importance_dict = dict(zip(feature_names, importances))
    else:
        print("Warning: Model does not support feature importance")
    
    return importance_dict

def analyze_errors(y_true: np.ndarray, y_pred: np.ndarray, 
                  X: pd.DataFrame = None,
                  task_type: str = 'classification') -> Dict[str, Any]:
    """予測誤差を分析します。
    
    Args:
        y_true (np.ndarray): 真のラベル
        y_pred (np.ndarray): 予測ラベル
        X (pd.DataFrame, optional): 特徴量. Defaults to None.
        task_type (str, optional): タスクの種類（'classification' または 'regression'）. Defaults to 'classification'.
        
    Returns:
        Dict[str, Any]: エラー分析の結果
    """
    analysis = {}
    
    if task_type == 'classification':
        # 分類タスクのエラー分析
        incorrect_mask = (y_true != y_pred)
        error_rate = np.mean(incorrect_mask)
        
        analysis['error_rate'] = float(error_rate)
        analysis['correct_count'] = int(len(y_true) - np.sum(incorrect_mask))
        analysis['incorrect_count'] = int(np.sum(incorrect_mask))
        
        # クラスごとのエラー率
        unique_classes = np.unique(y_true)
        class_errors = {}
        
        for cls in unique_classes:
            cls_mask = (y_true == cls)
            cls_error = np.mean(incorrect_mask[cls_mask]) if np.any(cls_mask) else 0.0
            class_errors[str(cls)] = float(cls_error)
        
        analysis['class_errors'] = class_errors
        
        # 特徴量が提供されている場合、エラーと相関の高い特徴量を特定
        if X is not None and isinstance(X, pd.DataFrame):
            try:
                # エラーと相関の高い特徴量を特定
                X_errors = X.copy()
                X_errors['is_error'] = incorrect_mask.astype(int)
                
                # 数値特徴量との相関
                numeric_cols = X_errors.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    error_corr = X_errors[numeric_cols].corrwith(X_errors['is_error']).sort_values(
                        key=abs, ascending=False
                    )
                    analysis['feature_error_correlation'] = error_corr.drop('is_error', errors='ignore').to_dict()
                
                # カテゴリカル特徴量のエラー率
                categorical_cols = X_errors.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    cat_errors = {}
                    for col in categorical_cols:
                        error_rates = X_errors.groupby(col)['is_error'].mean().sort_values(ascending=False)
                        cat_errors[col] = error_rates.to_dict()
                    analysis['categorical_error_rates'] = cat_errors
                    
            except Exception as e:
                print(f"Warning: Error during feature analysis: {e}")
    
    else:  # 回帰タスクのエラー分析
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        
        analysis['mean_absolute_error'] = float(np.mean(abs_errors))
        analysis['median_absolute_error'] = float(np.median(abs_errors))
        analysis['max_error'] = float(np.max(abs_errors))
        analysis['mean_error'] = float(np.mean(errors))  # バイアス
        
        # 誤差の分布
        analysis['error_stats'] = {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            '25%': float(np.percentile(errors, 25)),
            '50%': float(np.median(errors)),
            '75%': float(np.percentile(errors, 75)),
            'max': float(np.max(errors))
        }
        
        # 特徴量が提供されている場合、誤差と相関の高い特徴量を特定
        if X is not None and isinstance(X, pd.DataFrame):
            try:
                X_errors = X.copy()
                X_errors['error'] = errors
                
                # 数値特徴量との相関
                numeric_cols = X_errors.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    error_corr = X_errors[numeric_cols].corrwith(X_errors['error']).sort_values(
                        key=abs, ascending=False
                    )
                    analysis['feature_error_correlation'] = error_corr.drop('error', errors='ignore').to_dict()
                
            except Exception as e:
                print(f"Warning: Error during feature analysis: {e}")
    
    return analysis
