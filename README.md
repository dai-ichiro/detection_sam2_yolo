### 環境
```
Ubuntu 24.04 on WSL2
Python 3.12
```

### 事前準備
```
sudo apt install -y libgtk-3-dev libjpeg-dev libpng-dev
```

### リポジトリのクローン
```
git clone https://github.com/dai-ichiro/detection_sam2_yolo
cd detection_sam2_yolo
```

### モデルのダウンロード（SAM2）
```
cd models
. download_models.sh
cd ../
```

### Python環境の構築
```
uv sync
```

### サンプルデータ（学習用）のダウンロード
```
uv run tools/download_sample_videos.py
```

## 学習

```
uv run train.py
```

## 推論

```
uv run inference.py --image sample_images/sample01.jpg --weights runs/detect/train/weights/best.pt
```
