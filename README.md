# DGM for General ML Tasks (dgm-ml-v4)

このリポジトリは、Darwin Gödel Machine (DGM) の概念を、タイタニックの生存予測のような一般的な機械学習タスクに応用するための実装です。
LLMと進化的アルゴリズムを組み合わせ、機械学習パイプライン（特徴量エンジニアリング、モデル選択、ハイパーパラメータ最適化など）を自動的に改善・進化させることを目指します。

## 特徴

* **自己改善型パイプライン:** 機械学習パイプライン自体がDGMのエージェントとして扱われ、自己の構成や処理フローを改善します。
* **LLMによる提案:** 大規模言語モデル(LLM)を活用して、パイプラインの改善案（新しい特徴量の追加、モデルの変更など）を生成します。
* **経験的検証:** 提案された変更は、実際のデータセットでの評価を通じてその効果が検証されます。
* **オープンエンドな探索:** 生成された多様なパイプラインはアーカイブに保存され、将来の改善のための「踏み石」となります。

## ディレクトリ構造

```
dgm-ml-v4/
├── .github/                # GitHub ActionsなどCI/CD用 (オプション)
│   └── workflows/
│       └── main.yml
├── .gitignore              # Gitで無視するファイル
├── README.md               # リポジトリの説明
├── requirements.txt        # 必要なPythonライブラリ
├── config/                 # 設定ファイル用ディレクトリ
│   ├── global_config.yaml  # 初期戦略やベースライン設定
│   └── task_titanic.yaml  # タスク特有の設定 (タイタニック用)
├── data/                   # データセット用ディレクトリ (gitignoreで実際のデータは除外推奨)
│   └── titanic/
│       ├── train.csv
│       └── test.csv
├── dgm_core/               # DGMのコアロジック
│   ├── __init__.py
│   ├── agent.py           # 機械学習パイプラインを表現するエージェント
│   ├── archive.py         # 生成されたエージェント(パイプライン)を保存・管理
│   ├── evolution.py       # 進化的操作 (選択、自己修正指示など)
│   ├── llm_utils.py      # LLMとの連携用ユーティリティ
│   └── evaluation.py      # パイプラインの評価用
├── notebooks/             # 実験や分析用のJupyter Notebook (オプション)
│   └── titanic_initial_exploration.ipynb
├── pipelines/             # 生成・進化したパイプラインのコードが保存される場所
│   └── archive/           # アーカイブされたパイプラインコード
├── scripts/               # 補助的なスクリプト (データ準備、結果集計など)
│   └── run_dgm.py         # DGMのメイン実行スクリプト
└── tests/                 # テストコード用ディレクトリ
    ├── __init__.py
    ├── test_agent.py
    └── test_evaluation.py
```

## セットアップ

1. リポジトリをクローンします:
   ```bash
   git clone [リポジトリURL]
   cd dgm-ml-v4
   ```

2. (推奨) 仮想環境を作成し、アクティベートします:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate  # Windows
   ```

3. 必要なライブラリをインストールします:
   ```bash
   pip install -r requirements.txt
   ```

4. LLM API（GeminiまたはGroq）設定手順:
    - **Gemini APIの場合**:
      - Gemini APIキーを取得します
      - 環境変数としてAPIキーを設定します:
        ```bash
        # Linux/Mac
        echo 'export GEMINI_API_KEY="your-gemini-api-key"' >> ~/.zshrc  # or ~/.bashrc
        source ~/.zshrc  # or ~/.bashrc
        # Windows (PowerShell)
        [System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'your-gemini-api-key', 'User')
        ```
    - **Groq APIの場合（推奨: Llama-3/4モデル対応）**:
      - [Groq Cloud](https://console.groq.com/) でAPIキーを取得します
      - 環境変数としてAPIキーを設定します:
        ```bash
        # Linux/Mac
        echo 'export GROQ_API_KEY="your-groq-api-key"' >> ~/.zshrc  # or ~/.bashrc
        source ~/.zshrc  # or ~/.bashrc
        # Windows (PowerShell)
        [System.Environment]::SetEnvironmentVariable('GROQ_API_KEY', 'your-groq-api-key', 'User')
        ```
      - configファイル（例: `config/global_config.yaml`）のllmセクションで `provider: groq` とし、`model_name` を `meta-llama/llama-4-scout-17b-16e-instruct` などGroq対応モデル名に設定してください。
      - 例:
        ```yaml
        llm:
          provider: groq
          model_name: meta-llama/llama-4-scout-17b-16e-instruct
          temperature: 0.4
          max_tokens: 2048
        ```
      - 詳細は [Groq公式ドキュメント](https://console.groq.com/docs) を参照してください。
    - **注意:** GroqとGeminiはどちらもサポートされていますが、Groq（Llama系）は日本語指示やコード生成時に高速・高精度な傾向があります。

5. 設定ファイルを確認します:
   - `config/global_config.yaml` の `llm` セクションが正しく設定されていることを確認してください
   - 必要に応じて `model_name` を変更できます（例: `gemini-1.5-pro` または `gemini-1.0-pro`）

5. `data/titanic/` にタイタニックの `train.csv` と `test.csv` を配置します。

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

## ライセンス

MIT
