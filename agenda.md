はい、承知いたしました。DGMの概念をタイタニックのような一般的な機械学習タスクに応用するための、新しいリポジトリのディレクトリ構造と主要なファイルの雛形を作成します。

これはあくまで初期の雛形であり、実際の開発ではさらに多くのファイルや詳細なコードが必要になることをご了承ください。

**リポジトリ名:** `dgm_for_ml_tasks` (例)

**ディレクトリ構造案:**

```
dgm_for_ml_tasks/
├── .github/                # GitHub ActionsなどCI/CD用 (オプション)
│   └── workflows/
│       └── main.yml
├── .gitignore              # Gitで無視するファイル
├── README.md               # リポジトリの説明
├── requirements.txt        # 必要なPythonライブラリ
├── config/                 # 設定ファイル用ディレクトリ
│   ├── initial_strategy.yaml # 初期戦略やベースライン設定
│   └── task_titanic.yaml     # タスク特有の設定 (タイタニック用)
├── data/                   # データセット用ディレクトリ (gitignoreで実際のデータは除外推奨)
│   ├── titanic/
│   │   ├── train.csv
│   │   └── test.csv
│   └── (他のタスク用データセット)
├── dgm_core/               # DGMのコアロジック
│   ├── __init__.py
│   ├── agent.py              # 機械学習パイプラインを表現するエージェント
│   ├── archive.py            # 生成されたエージェント(パイプライン)を保存・管理
│   ├── evolution.py          # 進化的操作 (選択、自己修正指示など)
│   ├── llm_utils.py          # LLMとの連携用ユーティリティ
│   └── evaluation.py         # パイプラインの評価用
├── notebooks/              # 実験や分析用のJupyter Notebook (オプション)
│   └── titanic_initial_exploration.ipynb
├── pipelines/              # 生成・進化したパイプラインのコードが保存される場所
│   └── archive/              # アーカイブされたパイプラインコード
├── scripts/                # 補助的なスクリプト (データ準備、結果集計など)
│   └── run_dgm.py            # DGMのメイン実行スクリプト
└── tests/                  # テストコード用ディレクトリ
    ├── __init__.py
    ├── test_agent.py
    └── test_evaluation.py
```

**主要ファイルの雛形:**

---

**`.gitignore`**

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
ENV/
VENV/

# Jupyter Notebook
.ipynb_checkpoints

# Data files (プロジェクトに合わせて調整)
data/*
!data/.gitkeep # dataディレクトリ自体は残す場合

# IDE and OS specific
.vscode/
.idea/
*.DS_Store

# Output files
pipelines/archive/*
output/
logs/

# Credentials
credentials.json
*.env
```

---

**`README.md`**

```markdown
# DGM for General ML Tasks (dgm_for_ml_tasks)

このリポジトリは、Darwin Gödel Machine (DGM) の概念を、タイタニックの生存予測のような一般的な機械学習タスクに応用するための実装です。
LLMと進化的アルゴリズムを組み合わせ、機械学習パイプライン（特徴量エンジニアリング、モデル選択、ハイパーパラメータ最適化など）を自動的に改善・進化させることを目指します。

## 特徴

* **自己改善型パイプライン:** 機械学習パイプライン自体がDGMのエージェントとして扱われ、自己の構成や処理フローを改善します。
* **LLMによる提案:** 大規模言語モデル(LLM)を活用して、パイプラインの改善案（新しい特徴量の追加、モデルの変更など）を生成します。
* **経験的検証:** 提案された変更は、実際のデータセットでの評価を通じてその効果が検証されます。
* **オープンエンドな探索:** 生成された多様なパイプラインはアーカイブに保存され、将来の改善のための「踏み石」となります。

## ディレクトリ構造

(ここに上記のディレクトリ構造を記載)

## セットアップ

1.  リポジトリをクローンします:
    ```bash
    git clone [https://github.com/your-username/dgm_for_ml_tasks.git](https://github.com/your-username/dgm_for_ml_tasks.git)
    cd dgm_for_ml_tasks
    ```
2.  (推奨) 仮想環境を作成し、アクティベートします:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  必要なライブラリをインストールします:
    ```bash
    pip install -r requirements.txt
    ```
4.  LLMのAPIキーを設定します (例: 環境変数 `OPENAI_API_KEY` や `GOOGLE_API_KEY`)。
5.  `data/titanic/` にタイタニックの `train.csv` と `test.csv` を配置します。

## 使用方法

DGMのメイン実行スクリプトを実行します:
```bash
python scripts/run_dgm.py --task titanic --config config/task_titanic.yaml
```

## 今後の展望

* より多様な機械学習タスクへの対応
* 自己修正メカニズムの高度化
* 計算効率の改善

## 貢献

貢献を歓迎します！Issueを作成するか、Pull Requestを送ってください。

## 優先的に実装すべき機能

1. **Pipeline Archiveの強化**
- パイプラインの系統追跡機能
- より効率的な性能比較メカニズム
- パイプラインの多様性を保つための機能

2. **LLMインタラクションの改善**
- より構造化されたプロンプト設計
- エラー時のリトライメカニズム
- コード生成の品質チェック機能

3. **評価システムの強化**
- クロスバリデーションのサポート
- 複数メトリックの同時評価
- 計算コストの考慮

4. **実行環境の整備**
- 依存関係の明確な管理
- 環境変数の適切な処理
- ロギングシステムの強化
```

---

**`requirements.txt`**

```txt
# Core ML Libraries
pandas
numpy
scikit-learn
xgboost
lightgbm
optuna # ハイパーパラメータ最適化用

# LLM Interaction (例: OpenAI, Google Generative AI)
openai
google-generativeai

# Configuration
pyyaml

# Utilities
tqdm # プログレスバー

# (テスト用)
# pytest
```

---

**`config/initial_strategy.yaml`**

```yaml
# DGMの初期戦略やグローバル設定
# このファイルは、特定のタスクに依存しない基本的なDGMの挙動を定義します。

# DGMの実行設定
max_iterations: 50 # DGMの反復回数
archive_size_limit: 100 # アーカイブに保存するパイプラインの最大数 (古いものから削除など)

# LLM設定
llm_provider: "openai" # "openai", "google", "anthropic" など
# llm_model_name: "gpt-4" # 具体的なモデル名 (llm_utils.pyで選択ロジックを持つ方が柔軟かも)
llm_temperature_suggestion: 0.7 # 改善提案生成時の温度
llm_temperature_code_gen: 0.3   # コード生成時の温度

# 進化戦略
parent_selection_strategy: "elite_plus_diversity" # "roulette", "tournament" など
mutation_rate: 0.8 # LLMに変異（改善提案）を促す確率 (残りは既存の良い部分を組み合わせるなど)

# 初期パイプラインのベース (タスク設定ファイルで上書き可能)
initial_pipeline_template: |
  # この下にPythonコードのテンプレートを記述
  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score

  def preprocess_data(df_train, df_test):
      # 初期の前処理ステップ (LLMがここを改善していく)
      df_train_processed = df_train.copy()
      df_test_processed = df_test.copy()

      # 例: 数値列の欠損値を中央値で埋める (Age, Fare)
      for col in ['Age', 'Fare']:
          if col in df_train_processed.columns:
              median_val = df_train_processed[col].median()
              df_train_processed[col] = df_train_processed[col].fillna(median_val)
              if col in df_test_processed.columns:
                  df_test_processed[col] = df_test_processed[col].fillna(median_val)
      
      # 例: Sex列を数値に変換
      if 'Sex' in df_train_processed.columns:
          df_train_processed['Sex'] = df_train_processed['Sex'].map({'male': 0, 'female': 1})
          if 'Sex' in df_test_processed.columns:
              df_test_processed['Sex'] = df_test_processed['Sex'].map({'male': 0, 'female': 1})

      # LLMはここに新しい特徴量エンジニアリングステップを追加/変更できる
      # 例: Embarkedのワンホットエンコーディング (LLMが提案するかもしれない)
      # df_train_processed = pd.get_dummies(df_train_processed, columns=['Embarked'], dummy_na=False)
      # df_test_processed = pd.get_dummies(df_test_processed, columns=['Embarked'], dummy_na=False)

      # 特徴量の選択 (LLMがここを改善していく)
      # シンプルな例として、数値とSexのみを選択
      features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
      final_features_train = [f for f in features if f in df_train_processed.columns]
      final_features_test = [f for f in features if f in df_test_processed.columns]
      
      X_train = df_train_processed[final_features_train]
      X_test = df_test_processed[final_features_test]
      
      # テストデータに訓練データと同じ列構造を保証する (ダミー変数の差異などに対応)
      train_cols = X_train.columns
      X_test = X_test.reindex(columns=train_cols, fill_value=0)

      return X_train, X_test

  def train_model(X_train, y_train, model_params=None):
      # 初期モデル (LLMがここを改善していく)
      # 例: RandomForestClassifier
      default_params = {'n_estimators': 100, 'random_state': 42}
      if model_params:
          default_params.update(model_params)
      
      model = RandomForestClassifier(**default_params)
      model.fit(X_train, y_train)
      return model

  def evaluate_model(model, X_val, y_val):
      predictions = model.predict(X_val)
      return accuracy_score(y_val, predictions)

  # --- メイン実行ロジック ---
  # この部分は run_dgm.py から呼び出され、具体的に実行される
  # df_train, df_test, target_column は外部から与えられる想定
  
  # データ読み込み (run_dgm.py側で実施)
  # df_train = pd.read_csv('path_to_train.csv')
  # df_test = pd.read_csv('path_to_test.csv')
  # target_column = 'Survived'

  # y_train = df_train[target_column]
  # X_train_raw = df_train.drop(columns=[target_column])
  # X_test_raw = df_test.copy() # test.csvにターゲット列はない想定

  # 開発/検証用に訓練データの一部を分割 (LLMは分割戦略も変更できる)
  # X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
  #     X_train_raw, y_train, test_size=0.2, random_state=42, stratify=y_train
  # )

  # X_train_processed, X_val_processed = preprocess_data(X_train_split, X_val_split)
  # model = train_model(X_train_processed, y_train_split)
  # accuracy = evaluate_model(model, X_val_processed, y_val_split)
  # print(f"Validation Accuracy: {accuracy}")

  # 最終的な提出用予測 (この部分は評価後、最良パイプラインで行う)
  # X_train_full_processed, X_test_final_processed = preprocess_data(X_train_raw, X_test_raw)
  # final_model = train_model(X_train_full_processed, y_train)
  # test_predictions = final_model.predict(X_test_final_processed)
  # print("Test predictions generated.")
```

---

**`config/task_titanic.yaml`**

```yaml
# タイタニックタスク特有の設定
# initial_strategy.yaml の設定を上書き、または追加する

task_name: "titanic_survival_prediction"
description: "タイタニック号の乗客データから生存を予測する二値分類タスク"

dataset:
  train_path: "data/titanic/train.csv"
  test_path: "data/titanic/test.csv" # 提出用
  target_column: "Survived"
  id_column: "PassengerId" # 提出ファイル作成用

evaluation_metric: "accuracy" # "f1_score", "roc_auc" など
# モデルの評価に使用する主要メトリック

# タスク特有の初期パイプラインやヒント (オプション)
# initial_pipeline_template を上書きしたり、LLMへの初期プロンプトに含める情報を記述
# task_specific_hints:
#   - "PclassとFareの交互作用項が有効かもしれない"
#   - "NameからTitleを抽出すると良い特徴量になる"

# Optunaによるハイパーパラメータ探索の設定 (オプション)
hyperparameter_tuning:
  enabled: true
  n_trials: 20 # 各パイプライン候補のモデルに対するOptunaの試行回数
  timeout_per_pipeline: 300 # 秒

# このタスクで試行するモデルの候補リスト (LLMがこれらから選択・提案)
candidate_models:
  - "LogisticRegression"
  - "RandomForestClassifier"
  - "GradientBoostingClassifier"
  - "XGBClassifier"
  - "LGBMClassifier"
  # - "SVC" # 計算コストが高い場合がある

# 特徴量エンジニアリングで試行する操作の候補 (LLMがこれらを選択・提案)
candidate_fe_operations:
  - "impute_numerical_median" # 数値列を中央値で補完
  - "impute_numerical_mean"   # 数値列を平均値で補完
  - "impute_categorical_mode" # カテゴリ列を最頻値で補完
  - "encode_categorical_onehot" # ワンホットエンコーディング
  - "encode_categorical_label"  # ラベルエンコーディング
  - "create_polynomial_features" # 多項式特徴量
  - "extract_title_from_name"   # 名前から敬称を抽出
  - "bin_numerical_feature"     # 数値特徴量をビニング
  - "scale_numerical_standard"  # 標準スケーリング
  # - "drop_columns" # 特定の列を削除 (LLMが判断)
```

---
**`dgm_core/agent.py`**

```python
import uuid
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import optuna
import traceback
import importlib.util
import sys
from pathlib import Path
import joblib # モデル永続化用

# 各モデルクラスのインポート (requirements.txt に合わせて)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
# from sklearn.svm import SVC # 必要に応じて

SUPPORTED_MODELS = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "XGBClassifier": XGBClassifier,
    "LGBMClassifier": LGBMClassifier,
    # "SVC": SVC,
}


class MachineLearningPipelineAgent:
    def __init__(self, pipeline_code: str, agent_id: str = None, generation: int = 0, parent_id: str = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.pipeline_code = pipeline_code # Pythonコード文字列
        self.generation = generation
        self.parent_id = parent_id
        self.performance_metrics = {} # {'accuracy': 0.85, 'f1': 0.82}
        self.model = None # 学習済みモデル
        self.feature_engineering_steps = [] # LLMによって抽出されたFEステップの説明
        self.model_type = None # 使用されたモデルのタイプ
        self.hyperparameters = {} # 使用されたハイパーパラメータ

    def _execute_pipeline_code(self, df_train_full: pd.DataFrame, target_column: str, task_config: dict, global_config: dict):
        """
        与えられたpipeline_codeを実行してモデルを学習し、検証セットで評価する。
        Optunaによるハイパーパラメータチューニングもこの中で行う。
        """
        # パイプラインコードを動的に実行するための準備
        # 一時ファイルにコードを書き出し、モジュールとしてインポートする
        temp_module_name = f"pipeline_agent_{self.agent_id.replace('-', '_')}"
        temp_file_path = Path(f"{temp_module_name}.py")

        try:
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(self.pipeline_code)
            
            spec = importlib.util.spec_from_file_location(temp_module_name, temp_file_path)
            pipeline_module = importlib.util.module_from_spec(spec)
            sys.modules[temp_module_name] = pipeline_module # モジュールキャッシュに追加
            spec.loader.exec_module(pipeline_module)

            # --- データ準備 ---
            y_train_full = df_train_full[target_column]
            X_train_full_raw = df_train_full.drop(columns=[target_column])

            # 訓練データと検証データに分割 (層化サンプリングを推奨)
            X_train_raw, X_val_raw, y_train, y_val = train_test_split(
                X_train_full_raw, y_train_full, 
                test_size=0.2, # configから取得できるようにする
                random_state=global_config.get('random_seed', 42),
                stratify=y_train_full # 分類タスクの場合
            )
            
            # --- 特徴量エンジニアリング (パイプラインコード内の関数を呼び出す) ---
            # preprocess_data は (df_train, df_test) を受け取り、(X_train_processed, X_test_processed) を返す想定
            if not hasattr(pipeline_module, 'preprocess_data'):
                raise AttributeError("Pipeline code must define a 'preprocess_data' function.")
            X_train_processed, X_val_processed = pipeline_module.preprocess_data(X_train_raw.copy(), X_val_raw.copy())
            
            self.model_type = getattr(pipeline_module, 'MODEL_TYPE', 'Unknown') # パイプラインコードにMODEL_TYPEを定義させる

            # --- Optunaによるハイパーパラメータ最適化 (有効な場合) ---
            best_params = {}
            if task_config.get('hyperparameter_tuning', {}).get('enabled', False):
                if not hasattr(pipeline_module, 'create_model_for_optuna'):
                    raise AttributeError("Pipeline code must define 'create_model_for_optuna' for hyperparameter tuning.")
                
                def objective(trial):
                    # create_model_for_optuna は (trial, model_type) を受け取り、未学習モデルとパラメータ辞書を返す想定
                    model_instance, params = pipeline_module.create_model_for_optuna(trial, self.model_type)
                    
                    # このモデルで学習・評価
                    model_instance.fit(X_train_processed, y_train)
                    # Optunaは最大化/最小化する指標を返す必要がある
                    # 例: accuracy (最大化)
                    if task_config['evaluation_metric'] == 'accuracy':
                        return accuracy_score(y_val, model_instance.predict(X_val_processed))
                    elif task_config['evaluation_metric'] == 'f1_score':
                        # 二値分類以外の場合は average='weighted' などを考慮
                        return f1_score(y_val, model_instance.predict(X_val_processed), average='binary' if len(y_val.unique()) == 2 else 'weighted')
                    # 他のメトリックもサポート
                    else: # デフォルトはaccuracy
                        return accuracy_score(y_val, model_instance.predict(X_val_processed))

                study_direction = "maximize" # メトリックに応じて変更
                study = optuna.create_study(direction=study_direction)
                study.optimize(objective, 
                               n_trials=task_config['hyperparameter_tuning'].get('n_trials', 20),
                               timeout=task_config['hyperparameter_tuning'].get('timeout_per_pipeline'))
                best_params = study.best_params
                self.hyperparameters = best_params
            
            # --- 最適化されたパラメータ (またはデフォルト) で最終モデル学習 ---
            if not hasattr(pipeline_module, 'train_final_model'):
                 raise AttributeError("Pipeline code must define a 'train_final_model' function.")
            # train_final_model は (X_train, y_train, model_type, best_params) を受け取り、学習済みモデルを返す想定
            self.model = pipeline_module.train_final_model(X_train_processed, y_train, self.model_type, best_params)
            
            # --- 検証セットで評価 ---
            if not hasattr(pipeline_module, 'evaluate_model'):
                 raise AttributeError("Pipeline code must define an 'evaluate_model' function.")
            # evaluate_model は (model, X_val, y_val, metric_name) を受け取り、スコアを返す想定
            
            primary_metric_name = task_config['evaluation_metric']
            self.performance_metrics[primary_metric_name] = pipeline_module.evaluate_model(
                self.model, X_val_processed, y_val, primary_metric_name
            )
            # 他のメトリックも計算 (オプション)
            try:
                self.performance_metrics['accuracy'] = accuracy_score(y_val, self.model.predict(X_val_processed))
                self.performance_metrics['f1_score'] = f1_score(y_val, self.model.predict(X_val_processed), average='binary' if len(y_val.unique()) == 2 else 'weighted', zero_division=0)
                if hasattr(self.model, "predict_proba"):
                     self.performance_metrics['roc_auc'] = roc_auc_score(y_val, self.model.predict_proba(X_val_processed)[:,1])
            except Exception as e:
                print(f"Warning: Could not calculate auxiliary metrics: {e}")

            # 特徴量エンジニアリングステップやモデル情報を抽出 (LLMに解釈させるか、コード内から取得)
            self.feature_engineering_steps = getattr(pipeline_module, 'FE_STEPS_DESCRIPTION', ["No description provided."])
            
            print(f"Agent {self.agent_id} executed. Primary metric ({primary_metric_name}): {self.performance_metrics[primary_metric_name]:.4f}")

        except Exception as e:
            print(f"Error executing pipeline for agent {self.agent_id}: {e}")
            traceback.print_exc()
            self.performance_metrics[task_config.get('evaluation_metric', 'accuracy')] = 0.0 # エラー時は性能0
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink() # 一時ファイルを削除
            if temp_module_name in sys.modules:
                del sys.modules[temp_module_name] # モジュールキャッシュから削除

    def run_and_evaluate(self, df_train_full: pd.DataFrame, target_column: str, task_config: dict, global_config: dict):
        """パイプラインを実行し、性能を評価する"""
        self._execute_pipeline_code(df_train_full, target_column, task_config, global_config)
        return self.performance_metrics.get(task_config['evaluation_metric'], 0.0)

    def predict(self, df_test_raw: pd.DataFrame):
        """学習済みモデルを使ってテストデータで予測する"""
        if not self.model:
            raise ValueError("Model is not trained yet.")

        # パイプラインコードから preprocess_data を再度利用してテストデータを処理
        temp_module_name = f"pipeline_agent_{self.agent_id.replace('-', '_')}_predict"
        temp_file_path = Path(f"{temp_module_name}.py")
        predictions = None
        try:
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(self.pipeline_code)
            
            spec = importlib.util.spec_from_file_location(temp_module_name, temp_file_path)
            pipeline_module = importlib.util.module_from_spec(spec)
            sys.modules[temp_module_name] = pipeline_module
            spec.loader.exec_module(pipeline_module)
            
            # preprocess_data は (df_train, df_test) を受け取る想定だが、
            # 予測時は df_train は使わない形で呼び出すか、Noneを渡せるようにする。
            # ここでは、簡略化のため、df_trainにダミーの空データフレームを渡すことを想定。
            # より良いのは、予測専用の transform 関数をパイプラインに持たせること。
            _, X_test_processed = pipeline_module.preprocess_data(pd.DataFrame(), df_test_raw.copy())
            predictions = self.model.predict(X_test_processed)
        except Exception as e:
            print(f"Error during prediction for agent {self.agent_id}: {e}")
            traceback.print_exc()
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()
            if temp_module_name in sys.modules:
                del sys.modules[temp_module_name]
        return predictions

    def save_model(self, file_path: str):
        if self.model and joblib:
            joblib.dump(self.model, file_path)
            print(f"Model for agent {self.agent_id} saved to {file_path}")
        elif not joblib:
            print("Warning: joblib is not installed. Cannot save model.")
        else:
            print("Warning: No model to save.")

    def load_model(self, file_path: str):
        if joblib:
            try:
                self.model = joblib.load(file_path)
                print(f"Model for agent {self.agent_id} loaded from {file_path}")
            except FileNotFoundError:
                print(f"Error: Model file not found at {file_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("Warning: joblib is not installed. Cannot load model.")


    def get_summary(self):
        return {
            "agent_id": self.agent_id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "performance_metrics": self.performance_metrics,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "feature_engineering_steps_desc": self.feature_engineering_steps
            # "pipeline_code": self.pipeline_code # 必要に応じて含める (長くなる可能性)
        }

```
**注意:** `agent.py` 内の `_execute_pipeline_code` は、パイプラインコード (`self.pipeline_code`) が特定の関数（`preprocess_data`, `create_model_for_optuna`, `train_final_model`, `evaluate_model`）や定数（`MODEL_TYPE`, `FE_STEPS_DESCRIPTION`）を定義していることを期待しています。LLMが生成するパイプラインコードは、この規約に従う必要があります。

---

**`dgm_core/archive.py`**

```python
import json
from pathlib import Path
import shutil

class PipelineArchive:
    def __init__(self, archive_dir: str = "pipelines/archive", size_limit: int = 100):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.size_limit = size_limit
        self.agents_summary = [] # agentのサマリー情報を保持 (パフォーマンスなどでソートするため)
        self._load_existing_summaries()

    def _load_existing_summaries(self):
        """既存のサマリーファイルを読み込む"""
        summary_file = self.archive_dir / "_archive_summary.jsonl"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        self.agents_summary.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line in summary: {line}")
            # パフォーマンスでソート (例: accuracyが高い順)
            # TODO: evaluation_metricに応じてソートキーを変更できるようにする
            self.agents_summary.sort(key=lambda x: x.get('performance_metrics', {}).get('accuracy', 0.0), reverse=True)


    def add_agent(self, agent):
        """エージェントをアーカイブに追加する"""
        agent_summary = agent.get_summary()
        
        # パイプラインコードをファイルに保存
        agent_code_path = self.archive_dir / f"{agent.agent_id}.py"
        with open(agent_code_path, "w", encoding="utf-8") as f:
            f.write(agent.pipeline_code)
        
        # モデルを保存 (オプション)
        agent_model_path = self.archive_dir / f"{agent.agent_id}.joblib"
        agent.save_model(agent_model_path)

        # サマリーリストに追加してソート
        self.agents_summary.append(agent_summary)
        # TODO: evaluation_metricに応じてソートキーを変更
        self.agents_summary.sort(key=lambda x: x.get('performance_metrics', {}).get('accuracy', 0.0), reverse=True)

        # サイズ制限を超えていたら古いエージェントを削除
        if len(self.agents_summary) > self.size_limit:
            removed_agent_summary = self.agents_summary.pop() # 最も性能の低いエージェント (ソート済みなので末尾)
            removed_agent_id = removed_agent_summary['agent_id']
            
            # 対応するファイルも削除
            if (self.archive_dir / f"{removed_agent_id}.py").exists():
                (self.archive_dir / f"{removed_agent_id}.py").unlink()
            if (self.archive_dir / f"{removed_agent_id}.joblib").exists():
                (self.archive_dir / f"{removed_agent_id}.joblib").unlink()
            print(f"Removed agent {removed_agent_id} from archive due to size limit.")
        
        self._save_summaries()
        print(f"Agent {agent.agent_id} added to archive. Performance: {agent_summary.get('performance_metrics')}")

    def _save_summaries(self):
        """現在のサマリーリストをファイルに保存"""
        summary_file = self.archive_dir / "_archive_summary.jsonl"
        with open(summary_file, "w", encoding="utf-8") as f:
            for summary in self.agents_summary:
                f.write(json.dumps(summary) + "\n")

    def get_best_agents(self, n: int = 1, metric: str = 'accuracy'):
        """指定されたメトリックで上位N件のエージェントのサマリーを返す"""
        # ソートキーを動的に設定
        sorted_summaries = sorted(
            self.agents_summary,
            key=lambda x: x.get('performance_metrics', {}).get(metric, 0.0),
            reverse=True
        )
        return sorted_summaries[:n]

    def get_agent_code(self, agent_id: str) -> str | None:
        """指定されたIDのエージェントのパイプラインコードを読み込む"""
        agent_code_path = self.archive_dir / f"{agent_id}.py"
        if agent_code_path.exists():
            with open(agent_code_path, "r", encoding="utf-8") as f:
                return f.read()
        return None
        
    def get_all_summaries(self):
        return self.agents_summary.copy()

    def clear_archive(self):
        """アーカイブディレクトリ内の全ファイル（サブディレクトリ含む）を削除する"""
        for item in self.archive_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        self.agents_summary = []
        print("Archive cleared.")
```

---
**`dgm_core/llm_utils.py`**

```python
# LLMとのやり取りを行う関数群
# 例: OpenAI, Google Generative AI, Anthropicなどに対応できるようにする

import os
from openai import OpenAI
import google.generativeai as genai
# import anthropic # 必要に応じて

# --- 設定 ---
# APIキーは環境変数から読み込むことを推奨
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# if GOOGLE_API_KEY:
#     genai.configure(api_key=GOOGLE_API_KEY)

class LLMProvider:
    def __init__(self, provider_name: str, model_name: str = None, api_key: str = None):
        self.provider_name = provider_name.lower()
        self.model_name = model_name
        self.api_key = api_key

        if self.provider_name == "openai":
            if not self.api_key: self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key: raise ValueError("OpenAI API key not found.")
            self.client = OpenAI(api_key=self.api_key)
            if not self.model_name: self.model_name = "gpt-3.5-turbo" # デフォルト
        elif self.provider_name == "google":
            if not self.api_key: self.api_key = os.getenv("GOOGLE_API_KEY")
            if not self.api_key: raise ValueError("Google API key not found.")
            genai.configure(api_key=self.api_key)
            if not self.model_name: self.model_name = "gemini-1.5-flash-latest" # デフォルト
            self.client = genai.GenerativeModel(self.model_name)
        # elif self.provider_name == "anthropic":
        #     if not self.api_key: self.api_key = os.getenv("ANTHROPIC_API_KEY")
        #     if not self.api_key: raise ValueError("Anthropic API key not found.")
        #     self.client = anthropic.Anthropic(api_key=self.api_key)
        #     if not self.model_name: self.model_name = "claude-3-opus-20240229" # デフォルト
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        try:
            if self.provider_name == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            elif self.provider_name == "google":
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
                response = self.client.generate_content(prompt, generation_config=generation_config)
                # エラーハンドリングとレスポンス形式の確認が必要 (DGM論文のコード参照)
                if response.candidates and response.candidates[0].content.parts:
                    return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                else:
                    # 詳細なエラーや空のレスポンスのログ出力
                    print(f"Warning: LLM (Google) returned no usable content. Response: {response}")
                    return ""
            # elif self.provider_name == "anthropic":
            #     response = self.client.messages.create(
            #         model=self.model_name,
            #         max_tokens=max_tokens,
            #         temperature=temperature,
            #         messages=[{"role": "user", "content": prompt}]
            #     )
            #     return response.content[0].text.strip()
        except Exception as e:
            print(f"Error generating text with {self.provider_name} ({self.model_name}): {e}")
            return f"Error: Could not get response from LLM. Details: {str(e)}"
        return ""


def generate_pipeline_improvement_prompt(
    task_description: str,
    current_pipeline_code: str,
    current_performance: dict,
    historical_context: list, # 過去の試行のサマリーリスト
    available_models: list,
    available_fe_ops: list,
    evaluation_metric: str,
    initial_template: str # ベースとなるテンプレート
    ):
    """
    LLMにパイプライン改善案と新しいコードを生成させるためのプロンプトを作成する。
    DGM論文の create_llm_prompt に相当するものを、このタスク用に調整する。
    """
    history_summary = "\n".join([
        f"- Generation {h['generation']}, Agent ID {h['agent_id'][:8]}, {evaluation_metric}: {h.get('performance_metrics',{}).get(evaluation_metric,0):.4f}, Model: {h.get('model_type', 'N/A')}"
        for h in historical_context[-5:] # 直近5件の履歴など
    ])

    prompt = f"""あなたは最先端の機械学習エキスパートAIです。
目的は、与えられた機械学習タスクの性能({evaluation_metric})を最大化するために、Pythonで書かれた機械学習パイプラインを改善することです。

## タスク概要
{task_description}

## 現在のパイプラインコード
```python
{current_pipeline_code}
```

## 現在のパイプラインの性能
{json.dumps(current_performance, indent=2)}

## これまでの改善履歴 (直近)
{history_summary if history_summary else "まだ履歴はありません。これが最初の試行、またはベースラインからの改善です。"}

## あなたへの指示
上記の情報と履歴を踏まえ、現在のパイプラインを改善するための**具体的な変更点**を提案し、**変更後の完全なPythonパイプラインコード**を生成してください。
パイプラインコードは、以下の規約に従う必要があります。
1.  `preprocess_data(df_train, df_test)`関数を定義してください。これは訓練データとテストデータを受け取り、処理済みの特徴量X_train, X_testを返します。
2.  `create_model_for_optuna(trial, model_type)`関数を定義してください。これはOptunaの`trial`オブジェクトとモデルタイプ文字列を受け取り、未学習のモデルインスタンスと試行されたパラメータの辞書を返します。
3.  `train_final_model(X_train, y_train, model_type, best_params)`関数を定義してください。これは訓練データ、ターゲット、モデルタイプ、最適化されたパラメータを受け取り、学習済みモデルを返します。
4.  `evaluate_model(model, X_val, y_val, metric_name)`関数を定義してください。これは学習済みモデル、検証データ、ターゲット、評価メトリック名を受け取り、そのメトリックのスコアを返します。
5.  スクリプトのグローバルスコープに `MODEL_TYPE = "モデル名"` (例: "RandomForestClassifier") と `FE_STEPS_DESCRIPTION = ["特徴量エンジニアリングの説明1", "説明2"]` をリスト形式で定義してください。

改善提案では、以下の点を考慮してください：
- **特徴量エンジニアリング:** 新しい特徴量の作成、既存特徴量の変換、不要な特徴量の削除、欠損値処理の改善、エンコーディング方法の変更など。利用可能な操作の候補: {', '.join(available_fe_ops)}
- **モデル選択:** 別の種類のモデルの試用。利用可能なモデルの候補: {', '.join(available_models)}
- **ハイパーパラメータ:** Optunaが探索するパラメータ範囲の変更や、`create_model_for_optuna`関数内での提案方法の変更。
- **パイプライン全体の構造:** より効率的、または効果的な処理フローへの変更。

**出力フォーマット:**
提案理由と新しいパイプラインコードを、以下のJSON形式で提供してください。
```json
{{
  "reasoning": "ここに、なぜこの変更を行うのか、どのような改善が期待できるのかについての詳細な説明を記述します。",
  "new_pipeline_code": "# ここに改善後の完全なPythonパイプラインコードを記述します。\n# 上記の規約に従ってください。\nimport pandas as pd\n# ... (必要なインポート)\n\nMODEL_TYPE = \"NewModelClassifier\"\nFE_STEPS_DESCRIPTION = [\"新しい特徴量Xを追加\", \"Y列を対数変換\"]\n\ndef preprocess_data(df_train, df_test):\n  # ...\n  return X_train, X_test\n\ndef create_model_for_optuna(trial, model_type):\n  # ...\n  return model, params\n\ndef train_final_model(X_train, y_train, model_type, best_params):\n  # ...\n  return model\n\ndef evaluate_model(model, X_val, y_val, metric_name):\n  # ...\n  return score\n"
}}
```
**最も重要なのは、過去の繰り返しを避け、新しいアプローチで{evaluation_metric}の向上を目指すことです。**
ベースとなるパイプラインのテンプレートは以下の通りです。これを参考に、または改善の基点としてください。
```python
{initial_template}
```
変更後の完全なコードを提供してください。差分だけではありません。
"""
    return prompt

def parse_llm_suggestion(llm_response_str: str) -> dict | None:
    """LLMからのJSON形式の応答をパースする"""
    try:
        # ```json ... ``` マーカーを探す (DGM論文のコード参照)
        match = None
        if "```json" in llm_response_str:
            start_index = llm_response_str.find("```json") + len("```json")
            end_index = llm_response_str.rfind("```")
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = llm_response_str[start_index:end_index].strip()
            else: # マーカーが見つからない場合、全体を試す
                json_str = llm_response_str.strip()
        else: # マーカーがない場合
            json_str = llm_response_str.strip()
            
        # Python風のTrue/False/NoneをJSON準拠に
        json_str = json_str.replace("True", "true").replace("False", "false").replace("None", "null")
        # コメント行の削除 (単純な行頭 // や #)
        json_str = "\n".join(line for line in json_str.splitlines() if not line.strip().startswith(("//", "#")))
        # 末尾カンマの削除 (より堅牢な正規表現が必要な場合もある)
        json_str = json_str.replace(",\n]", "\n]").replace(",\n}", "\n}")


        parsed = json.loads(json_str)
        if "reasoning" in parsed and "new_pipeline_code" in parsed:
            return parsed
        else:
            print(f"Warning: LLM response missing 'reasoning' or 'new_pipeline_code'. Got keys: {parsed.keys()}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM JSON response: {e}")
        print(f"Problematic JSON string was: \n---\n{json_str}\n---")
        # LLMの応答全体もログ出力するとデバッグに役立つ
        # print(f"Full LLM response was: \n---\n{llm_response_str}\n---")
        return None
    except Exception as e:
        print(f"Unexpected error parsing LLM suggestion: {e}")
        # print(f"Full LLM response was: \n---\n{llm_response_str}\n---")
        return None

```

---
**`dgm_core/evolution.py`**

```python
import random
from .agent import MachineLearningPipelineAgent
from .llm_utils import LLMProvider, generate_pipeline_improvement_prompt, parse_llm_suggestion
from .archive import PipelineArchive

class DGMEvolution:
    def __init__(self, task_config: dict, global_config: dict, archive: PipelineArchive):
        self.task_config = task_config
        self.global_config = global_config
        self.archive = archive
        self.llm_provider = LLMProvider(
            provider_name=global_config['llm']['provider_name'],
            model_name=global_config['llm'].get('model_name') # Providerクラスでデフォルト設定
        )

    def select_parent_agent(self) -> MachineLearningPipelineAgent | None:
        """アーカイブから親となるエージェントを選択する"""
        # 戦略例: エリート選択 + 多様性 (ルーレットホイールなど)
        # ここでは、最も性能の良いものをベースにするか、ランダム性も加味する
        
        # 初回はベーステンプレートを使用
        if not self.archive.get_all_summaries():
            print("Archive is empty. Using initial template as parent.")
            initial_code = self.global_config.get('initial_pipeline_template', "# Empty initial template")
            # 初期エージェントの performance_metrics は空か、ベースライン性能を設定
            return MachineLearningPipelineAgent(pipeline_code=initial_code, generation=-1) # generation -1 は特別扱い

        # 現在の戦略 (configから取得)
        strategy = self.global_config.get('evolution',{}).get('parent_selection_strategy', 'elite')

        if strategy == 'elite':
            # 最も性能の良いエージェントのサマリーを取得
            best_agent_summary_list = self.archive.get_best_agents(n=1, metric=self.task_config['evaluation_metric'])
            if not best_agent_summary_list: return None # まだアーカイブに有効なものがない
            parent_summary = best_agent_summary_list[0]
        elif strategy == 'roulette_wheel': # TODO: 実装
            # 性能に応じた確率で選択
            summaries = self.archive.get_all_summaries()
            if not summaries: return None
            # ここでは単純にランダム選択の例 (改善が必要)
            parent_summary = random.choice(summaries)
            print(f"Selected parent {parent_summary['agent_id']} via random choice (placeholder for roulette).")

        elif strategy == 'elite_plus_diversity': # 例
            summaries = self.archive.get_all_summaries()
            if not summaries: return None
            if random.random() < 0.7: # 70%の確率でエリート
                 parent_summary = self.archive.get_best_agents(n=1, metric=self.task_config['evaluation_metric'])[0]
                 print(f"Selected elite parent {parent_summary['agent_id']}.")
            else: # 30%の確率でランダム（多様性のため）
                 parent_summary = random.choice(summaries)
                 print(f"Selected diverse parent {parent_summary['agent_id']}.")
        else: # デフォルトはエリート
            best_agent_summary_list = self.archive.get_best_agents(n=1, metric=self.task_config['evaluation_metric'])
            if not best_agent_summary_list: return None
            parent_summary = best_agent_summary_list[0]

        parent_code = self.archive.get_agent_code(parent_summary['agent_id'])
        if not parent_code:
            print(f"Error: Could not retrieve code for parent agent {parent_summary['agent_id']}")
            return None
            
        # MachineLearningPipelineAgentとして復元
        parent_agent = MachineLearningPipelineAgent(
            pipeline_code=parent_code,
            agent_id=parent_summary['agent_id'],
            generation=parent_summary['generation']
        )
        parent_agent.performance_metrics = parent_summary.get('performance_metrics', {})
        parent_agent.model_type = parent_summary.get('model_type')
        parent_agent.hyperparameters = parent_summary.get('hyperparameters', {})
        parent_agent.feature_engineering_steps = parent_summary.get('feature_engineering_steps_desc', [])
        return parent_agent

    def self_improve_agent(self, parent_agent: MachineLearningPipelineAgent, current_generation: int) -> MachineLearningPipelineAgent | None:
        """
        親エージェントのコードと性能を基に、LLMに改善案を問い合わせ、
        新しい子エージェントのコードを生成する。
        """
        print(f"\nAttempting to improve agent {parent_agent.agent_id} (Gen: {parent_agent.generation}) for new generation {current_generation}...")

        prompt = generate_pipeline_improvement_prompt(
            task_description=self.task_config['description'],
            current_pipeline_code=parent_agent.pipeline_code,
            current_performance=parent_agent.performance_metrics,
            historical_context=self.archive.get_all_summaries(), # アーカイブのサマリー全体を渡す
            available_models=self.task_config.get('candidate_models', []),
            available_fe_ops=self.task_config.get('candidate_fe_operations', []),
            evaluation_metric=self.task_config['evaluation_metric'],
            initial_template=self.global_config.get('initial_pipeline_template', "")
        )
        
        print("Sending prompt to LLM for improvement suggestion...")
        # print(f"--- PROMPT START ---\n{prompt[:1000]}...\n--- PROMPT END ---") # デバッグ用に一部表示

        llm_response_str = self.llm_provider.generate_text(
            prompt,
            temperature=self.global_config['llm'].get('llm_temperature_suggestion', 0.7)
        )

        if not llm_response_str or "Error: Could not get response from LLM" in llm_response_str :
            print(f"LLM did not provide a valid response. Raw response: {llm_response_str}")
            return None

        suggestion = parse_llm_suggestion(llm_response_str)

        if suggestion and suggestion.get("new_pipeline_code"):
            print(f"LLM suggested improvement. Reasoning: {suggestion.get('reasoning', 'N/A')[:200]}...")
            new_code = suggestion["new_pipeline_code"]
            
            # 新しいエージェントを作成
            child_agent = MachineLearningPipelineAgent(
                pipeline_code=new_code,
                generation=current_generation,
                parent_id=parent_agent.agent_id if parent_agent.generation != -1 else "initial_template"
            )
            # LLMのreasoningも保存しておくと良い
            child_agent.llm_reasoning = suggestion.get('reasoning')
            return child_agent
        else:
            print("LLM response could not be parsed or did not contain new pipeline code.")
            # print(f"Full LLM response for debugging:\n{llm_response_str}") # デバッグ用
            return None

```
---

**`scripts/run_dgm.py`**

```python
import argparse
import yaml
import pandas as pd
from pathlib import Path
import os
import sys
from datetime import datetime

# dgm_coreへのパスを追加 (実行場所によって調整)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dgm_core.agent import MachineLearningPipelineAgent
from dgm_core.archive import PipelineArchive
from dgm_core.evolution import DGMEvolution
# from dgm_core.evaluation import evaluate_pipeline # agent.run_and_evaluate に統合

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run DGM for a specified machine learning task.")
    parser.add_argument("--task", required=True, help="Name of the task (e.g., titanic).")
    parser.add_argument("--config", required=True, help="Path to the task-specific YAML configuration file.")
    parser.add_argument("--global-config", default="config/initial_strategy.yaml", help="Path to the global DGM strategy YAML file.")
    parser.add_argument("--output-dir", default="output", help="Directory to save DGM run outputs.")
    parser.add_argument("--clear-archive-on-start", action="store_true", help="Clear the pipeline archive before starting a new run.")

    args = parser.parse_args()

    # 設定ファイルの読み込み
    try:
        task_config = load_config(args.config)
        global_config = load_config(args.global_config)
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # 出力ディレクトリの準備
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_output_dir = Path(args.output_dir) / args.task / run_timestamp
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"DGM run output will be saved to: {current_run_output_dir}")

    # アーカイブの準備
    # タスクごとにアーカイブを分けるか、共通のアーカイブを使うか検討
    # ここではタスクごとのアーカイブディレクトリを使用
    archive_base_dir = global_config.get("archive", {}).get("base_dir", "pipelines")
    task_archive_dir = Path(archive_base_dir) / args.task / "archive_pool" # agent.pyのデフォルトと合わせる
    
    archive = PipelineArchive(
        archive_dir=str(task_archive_dir), # 文字列として渡す
        size_limit=global_config.get('archive_size_limit', 100)
    )
    if args.clear_archive_on_start:
        print("Clearing existing archive as requested...")
        archive.clear_archive()


    # データセットの読み込み (タスク設定ファイルからパスを取得)
    try:
        df_train_full = pd.read_csv(task_config['dataset']['train_path'])
        # テストセットは最終的な提出用モデルの評価に使う (DGMループ内では使わない)
        # df_test_for_submission = pd.read_csv(task_config['dataset']['test_path'])
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found. {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing dataset path in task configuration. {e}")
        sys.exit(1)
        
    target_column = task_config['dataset']['target_column']

    # DGM進化エンジンの初期化
    dgm_evolution = DGMEvolution(task_config, global_config, archive)

    max_iterations = global_config.get('dgm_run',{}).get('max_iterations', 10) # global_configのキー構造に合わせる

    print(f"Starting DGM for task '{args.task}' with {max_iterations} iterations...")

    for generation in range(max_iterations):
        print(f"\n--- Generation {generation + 1}/{max_iterations} ---")

        # 1. 親エージェントの選択
        parent_agent = dgm_evolution.select_parent_agent()
        if not parent_agent:
            print("Could not select a parent agent. Skipping this generation.")
            # ベーステンプレートから開始するロジックが select_parent_agent にあるはず
            # それでもNoneなら、設定を見直す必要がある
            if generation == 0: # 初回で親が取れないのは致命的
                 print("Error: Could not get initial parent/template. Exiting.")
                 sys.exit(1)
            continue
        
        if parent_agent.generation == -1: # 初期テンプレートの場合
            print(f"Using initial pipeline template for generation {generation + 1}.")
        else:
            print(f"Selected parent agent: {parent_agent.agent_id} (Gen: {parent_agent.generation}, Perf: {parent_agent.performance_metrics})")

        # 2. LLMによる自己改善提案と新しいパイプラインコードの生成
        child_agent = dgm_evolution.self_improve_agent(parent_agent, generation + 1)

        if not child_agent:
            print("LLM could not generate a valid improved agent. Skipping this generation.")
            continue
        
        print(f"Generated new child agent: {child_agent.agent_id} (Parent: {child_agent.parent_id})")

        # 3. 新しいエージェント(パイプライン)の実行と評価
        print(f"Evaluating child agent {child_agent.agent_id}...")
        child_agent.run_and_evaluate(
            df_train_full.copy(), # 毎回新しいコピーを渡す
            target_column,
            task_config,
            global_config
        )
        
        # 4. アーカイブへの追加
        if child_agent.performance_metrics.get(task_config['evaluation_metric'], 0.0) > 0.0: # 最低限の性能があるか
            archive.add_agent(child_agent)
        else:
            print(f"Child agent {child_agent.agent_id} did not achieve positive performance. Not adding to archive.")
        
        # 実行結果のログ保存 (各世代のサマリーなど)
        current_gen_summary_path = current_run_output_dir / f"generation_{generation+1}_summary.json"
        with open(current_gen_summary_path, "w", encoding="utf-8") as f:
            # parent_summary と child_summary を結合するなど
            log_entry = {
                "generation": generation + 1,
                "parent_summary": parent_agent.get_summary() if parent_agent.generation != -1 else "initial_template",
                "child_summary": child_agent.get_summary(),
                "llm_reasoning": getattr(child_agent, 'llm_reasoning', "N/A")
            }
            json.dump(log_entry, f, indent=2)
        
        print(f"Best agent in archive so far (top 1 by {task_config['evaluation_metric']}):")
        best_overall = archive.get_best_agents(n=1, metric=task_config['evaluation_metric'])
        if best_overall:
            print(json.dumps(best_overall[0], indent=2))
        else:
            print("Archive is still empty or no performing agents.")


    print("\n--- DGM Run Finished ---")
    final_best_agents = archive.get_best_agents(n=3, metric=task_config['evaluation_metric'])
    print(f"\nTop 3 agents from the archive based on {task_config['evaluation_metric']}:")
    for i, agent_summary in enumerate(final_best_agents):
        print(f"{i+1}. Agent ID: {agent_summary['agent_id']}, Gen: {agent_summary['generation']}, Performance: {agent_summary.get('performance_metrics')}")
        # 最良のエージェントのコードとモデルを特定の場所に保存するなども検討

    # 最終結果サマリーを保存
    final_summary_path = current_run_output_dir / "final_dgm_summary.json"
    with open(final_summary_path, "w", encoding="utf-8") as f:
        json.dump(archive.get_all_summaries(), f, indent=2)
    print(f"Full archive summary saved to {final_summary_path}")

if __name__ == "__main__":
    main()
```

---

これはあくまで骨子です。各ファイル、特に `dgm_core` 内のクラスや関数は、より詳細なエラーハンドリング、ロギング、多様な戦略のサポート、そしてLLMとのより洗練されたインタラクション（例えば、コードの差分だけを扱ったり、特定の部分の改善を指示したりする）といった機能拡張が必要になります。

特に、`agent.py` で実行されるパイプラインコードと、LLMがそれを生成・改善する際の規約（どの関数名を使うか、どのような入出力を期待するかなど）を明確に定義し、LLMへのプロンプトでそれを正確に伝えることが重要です。

この雛形を元に、具体的な機能を追加していくことで、DGMのコンセプトを一般的な機械学習タスクに応用するプログラムを構築できるでしょう。

curl https://api.groq.com/openai/v1/chat/completions -s \
-H "Content-Type: application/json" \
-H "Authorization: $GEOQ_API_KEY" \
-d '{
"model": "qwen-qwq-32b",
"messages": [{
    "role": "user",
    "content": "Explain the importance of fast language models"
}]
}'
