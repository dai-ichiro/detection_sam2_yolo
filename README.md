# SAM2 & YOLO 動画物体検出・追跡プロジェクト

このプロジェクトは、Metaの **SAM2 (Segment Anything Model 2)** を使用して動画内から学習データを自動生成し、そのデータを用いて **YOLO** モデルを学習させるためのツールを提供します。

## 特徴
- **自動アノテーション**: 動画の最初のフレームで対象を囲むだけで、SAM2が全フレームをトラッキングし、YOLO学習用のラベル（バウンディングボックス）を自動生成します。
- **一気通貫のパイプライン**: データの生成からYOLOの学習までを一つのスクリプトで実行可能です。

## 環境
- Ubuntu 24.04 (WSL2推奨)
- Python 3.12
- NVIDIA GPU (CUDA対応)

## セットアップ

### システム依存関係のインストール
```bash
sudo apt update
sudo apt install -y libgtk-3-dev libjpeg-dev libpng-dev imagemagick
```

### リポジトリのクローン
```bash
git clone https://github.com/dai-ichiro/detection_sam2_yolo
cd detection_sam2_yolo
```

### SAM2チェックポイントのダウンロード
```bash
cd models
. download_models.sh
cd ../
```

### Python環境の構築 (uvを使用)
```bash
uv sync
```

### サンプル動画のダウンロード (任意)
```bash
uv run tools/download_sample_videos.py
```

## 学習とデータ生成 (`train.py`)

`train.py` は、SAM2によるトラッキング（データ生成）とYOLOの学習の両方、あるいはいずれかを実行します。

### 基本的な実行
```bash
uv run train.py
```

### 引数詳細

| 引数 | 型 | デフォルト値 | 説明 |
| :--- | :--- | :--- | :--- |
| `--videos_dir` | `str` | `videos` | 学習用動画が保存されているディレクトリ。フォルダ名がクラス名として使用されます。 |
| `--epochs` | `int` | `4` | YOLO学習の総エポック数。 |
| `--batch` | `int` | `8` | YOLO学習のバッチサイズ。 |
| `--weights` | `str` | `yolo26m.pt` | YOLOの初期重み（または学習済みモデル）のパス。 |
| `--imgsz` | `int` | `640` | YOLO学習時の画像サイズ。 |
| `--sam2_checkpoint` | `str` | `models/sam2.1_hiera_large.pt` | 使用するSAM2のチェックポイントパス。 |
| `--sam2_config_dir` | `str` | `models` | SAM2のHydra設定ディレクトリ。 |
| `--sam2_model_cfg` | `str` | `sam2.1_hiera_l` | 使用するSAM2のモデル設定名。 |
| `--mode` | `str` | `both` | 実行モード。`tracking` (データ生成のみ), `yolo_train` (学習のみ), `both` (両方) から選択。 |

### 実行の流れ
1. `mode`が`tracking`または`both`の場合、ウィンドウが開き、動画の最初のフレームが表示されます。
2. 対象物をマウスで囲み（ROI選択）、`Enter`キーまたは`Space`キーを押します。
3. すべての動画に対して選択が終わると、SAM2による自動トラッキングが始まり、`train_data/`に学習用画像とラベルが保存されます。
4. `mode`が`yolo_train`または`both`の場合、生成されたデータを用いてYOLOの学習が始まります。

## 推論 (`inference.py`)

学習済みモデルを使用して、静止画に対して物体検出を行います。

### 実行例
```bash
uv run inference.py --image sample_images/sample01.jpg --weights runs/detect/train/weights/best.pt
```

### 引数詳細

| 引数 | 型 | 必須 | デフォルト値 | 説明 |
| :--- | :--- | :---: | :--- | :--- |
| `--image` | `str` | Yes | - | 推論を行いたい画像のパス。 |
| `--weights` | `str` | Yes | - | 学習済みYOLOモデル（`.pt`）のパス。 |
| `--output` | `str` | No | `result.jpg` | 結果を保存するパス。 |

## ディレクトリ構成
- `videos/`: 学習用動画を配置（フォルダ分けすることでマルチクラスに対応）。
- `train_data/`: SAM2によって生成された画像とラベルが保存されます。
- `runs/`: YOLOの学習結果（重みやログ）が保存されます。
- `models/`: SAM2のモデルファイルと設定。
