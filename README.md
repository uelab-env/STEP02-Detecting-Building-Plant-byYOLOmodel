# 航空写真からの建物設備検出システム（YOLOv8 OBB）
# Building Equipment Detection from Aerial Imagery (YOLOv8 OBB)

---
## 前提・準備 📋
- 実行環境：Windows11のUbuntu 24.04.3 LTS
- 仮想環境：Python 3.10.20
- 仮想環境は以下のコマンドを実行して構築可能
```conda env create -f yolo_env.yaml```


## 日本語

### 概要

航空写真（高解像度衛星画像）を用いて、建物屋上に設置された熱源設備を自動検出し、建物リスト CSV に設備種別（`plant` 列）を付与するシステムです。

検出対象の設備クラス：

| クラス | 設備名 |
|--------|--------|
| CT | 冷却塔 (Cooling Tower) |
| ACC | 空冷チラー (Air-Cooled Chiller) |
| MUL | マルチエアコン室外機 (Multi-split AC) |
| PAC | パッケージエアコン (Package AC) |

---

### ディレクトリ構成

```
.
├── detect_object_YOLO_carobock_ver8_1.py  # メイン検出スクリプト
├── merge_images_with_tfw.py               # 検出結果画像の統合ツール
├── models/
│   └── best.pt                            # 学習済みモデル（別途入手）
├── input/
│   ├── image_cut/
│   │   ├── png/                           # 入力画像（分割済み航空写真、1000×750px）
│   │   └── tfw/                           # 各画像の座標ファイル（TFW形式）
│   └── building_list/
│       └── TokyoChuo.csv                  # 建物リスト（緯度・経度列を含む CSV）
├── bld_boundary/
│   └── FG-GML-*.xml                       # 建物境界線 XML（国土地理院 FGD BldA）
├── cache/                                 # 実行後に自動生成（2回目以降の高速化用）
├── detection_results_images/              # 検出結果の可視化画像（実行後に自動生成）
└── output/
    └── TokyoChuo.csv                      # 出力 CSV（plant 列追加済み）
```

---

### セットアップ手順

#### 1. リポジトリのクローン

```bash
git clone <リポジトリURL>
cd <リポジトリ名>
```

#### 2. conda 環境の作成と有効化

```bash
conda create -n yolo_py_env python=3.10 -y ## お気に入りのpythonの仮想環境あれば、それを使用してもらってください
conda activate yolo_py_env
```

#### 3. 必要パッケージのインストール

```bash
pip install ultralytics shapely pyproj pandas Pillow natsort tqdm
```

> **GPU を使用する場合（推奨）**  
> PyTorch の GPU 対応版をインストールしてください。CUDA バージョンは環境に合わせて変更してください。
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

#### 4. 学習済みモデルの配置

`models/best.pt` に学習済みモデルファイルを配置してください（別途入手が必要です）。
これは、西村がデフォルトで配置してます

---

### 入力データの準備

| 配置先 | 内容 |
|--------|------|
| `input/image_cut/png/` | 航空写真の PNG 画像（1000×750px） |
| `input/image_cut/tfw/` | 対応する TFW ファイル（ファイル名は PNG と同一） |
| `input/building_list/TokyoChuo.csv` | 建物リスト CSV（`緯度`・`経度` 列が必要） |
| `bld_boundary/` | 建物境界線 XML（国土地理院「基盤地図情報 建物外周」FGD BldA 形式） |

- 建物境界線データは以下のリンクを参考にダウンロードしてください  
https://uelab.growi.cloud/6a6c68099b9a6f8caadc8172

#### 建物リスト CSV に必要な列

| 列名 | 説明 |
|------|------|
| `緯度` | 建物代表点の緯度（WGS84） |
| `経度` | 建物代表点の経度（WGS84） |

その他の列はそのまま出力 CSV に引き継がれます。

---

### 実行方法

#### STEP 1: 設備検出・判定の実行

```bash
conda activate yolo_py_env
python detect_object_YOLO_carobock_ver8_1.py
```

実行が完了すると、`output/TokyoChuo.csv` に `plant` 列が追加された CSV が出力されます。  
`plant` 列の値：`CT` / `ACC` / `MUL` / `PAC` / 空文字（検出なし）

> **2回目以降の実行について**  
> 初回実行後は `cache/` フォルダに中間結果が保存されるため、2回目以降は検出処理がスキップされ高速に動作します。
>
> **キャッシュファイルの削除が必要なケース：**
>
> | シナリオ | `detections_cache.json` | `building_detections_map_cache.json` |
> |---------|------------------------|-------------------------------------|
> | **建物リストのみ変更**<br>（同じ画像で別の建物リスト） | ✅ 再利用可能 | ❌ 削除が必要 |
> | **画像も変更** | ❌ 削除が必要 | ❌ 削除が必要 |
>
> **削除コマンド例：**
> ```bash
> # 建物リストのみ変更した場合
> rm cache/building_detections_map_cache.json
>
> # 画像も変更した場合（全キャッシュ削除）
> rm cache/*.json
> ```
>
> または、スクリプト冒頭の `USE_DETECTION_CACHE = False` に変更してください。

#### STEP 2: 検出結果画像の統合（任意）

```bash
python merge_images_with_tfw.py
```

STEP 1 で生成された検出結果の可視化画像（`detection_results_images/`）を、1枚の画像に統合します。

---

### 設定パラメータ

スクリプト冒頭の定数を変更することで動作を調整できます。

| 定数 | デフォルト値 | 説明 |
|------|------------|------|
| `EPSG_CODE` | 6677 | **入力画像の座標系（平面直角座標系）** <br> **⚠️ 対象地域に応じて必ず変更してください** <br> 6677 = IX系（東京・神奈川など）<br> 6669 = I系（北海道西部）、6670 = II系（北海道東部）<br> 6671 = III系（東北）、6672 = IV系（関東）<br> 6673 = V系（北陸）、6674 = VI系（中部）<br> 6675 = VII系（近畿）、6676 = VIII系（中国）<br> 6678 = X系（九州北部）、6679 = XI系（九州南部・沖縄）など |
| `YOLO_BATCH_SIZE` | 8 | 一度に処理する画像枚数（メモリに応じて調整） |
| `USE_DETECTION_CACHE` | `True` | `False` にすると検出を最初から再実行 |

---

### 注意事項

- **⚠️ 必ず対象地域の平面直角座標系に合わせて `EPSG_CODE` を変更してください**  
  デフォルトは東京周辺（IX系、EPSG:6677）です。他の地域で実行する場合、正しい座標系コードに変更しないと位置情報が大きくずれます。
- conda 環境を有効化（`conda activate yolo_py_env`）してから実行してください。
- 建物境界 XML は国土地理院「基盤地図情報 建物外周」（FGD BldA）の GML 形式のみ対応しています。
- STEP 1 を実行する前に、`input/image_cut/png/` に PNG 画像が存在することを確認してください。

---

---

## English

### Overview

This system automatically detects heat-source equipment installed on building rooftops from high-resolution aerial/satellite imagery. An equipment-type label (`plant` column) is appended to the input building list CSV.

Detected equipment classes:

| Class | Equipment |
|-------|-----------|
| CT | Cooling Tower |
| ACC | Air-Cooled Chiller |
| MUL | Multi-split Air Conditioner (outdoor unit) |
| PAC | Package Air Conditioner |

---

### Directory Structure

```
.
├── detect_object_YOLO_carobock_ver8_1.py  # Main detection script
├── merge_images_with_tfw.py               # Detection-image merging utility
├── models/
│   └── best.pt                            # Trained model weights (obtain separately)
├── input/
│   ├── image_cut/
│   │   ├── png/                           # Aerial image tiles (1000×750 px PNG)
│   │   └── tfw/                           # Coordinate files for each image (TFW format)
│   └── building_list/
│       └── TokyoChuo.csv                  # Building list CSV (must include lat/lng columns)
├── bld_boundary/
│   └── FG-GML-*.xml                       # Building boundary XMLs (GSI FGD BldA format)
├── cache/                                 # Auto-generated on first run (speeds up subsequent runs)
├── detection_results_images/              # Annotated detection images (auto-generated)
└── output/
    └── TokyoChuo.csv                      # Output CSV with `plant` column added
```

---

### Setup

#### 1. Clone the Repository

```bash
git clone <repository URL>
cd <repository name>
```

#### 2. Create and Activate a conda Environment

```bash
conda create -n yolo_py_env python=3.10 -y
conda activate yolo_py_env
```

#### 3. Install Required Packages

```bash
pip install ultralytics shapely pyproj pandas Pillow natsort tqdm
```

> **For GPU acceleration (recommended)**  
> Install the GPU-enabled version of PyTorch. Adjust the CUDA version to match your environment.
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

#### 4. Place the Model Weights

Place the trained model file at `models/best.pt` (must be obtained separately).

---

### Preparing Input Data

| Location | Contents |
|----------|----------|
| `input/image_cut/png/` | Tiled aerial PNG images (1000×750 px) |
| `input/image_cut/tfw/` | Corresponding TFW files (filenames must match the PNG files) |
| `input/building_list/TokyoChuo.csv` | Building list CSV (must include `緯度` and `経度` columns) |
| `bld_boundary/` | Building boundary XMLs (Japan GSI Fundamental Geospatial Data, FGD BldA GML format) |
- 建物境界線データは以下のリンクを参考にダウンロードしてください  
https://uelab.growi.cloud/6a6c68099b9a6f8caadc8172
#### Required Columns in the Building List CSV

| Column | Description |
|--------|-------------|
| `緯度` | Building centroid latitude (WGS84) |
| `経度` | Building centroid longitude (WGS84) |

All other columns are passed through to the output CSV unchanged.

---

### Running the System

#### STEP 1: Run Equipment Detection

```bash
conda activate yolo_py_env
python detect_object_YOLO_carobock_ver8_1.py
```

When complete, `output/TokyoChuo.csv` will be created with a `plant` column added.  
Possible values: `CT` / `ACC` / `MUL` / `PAC` / empty string (no equipment detected)

> **From the second run onwards**  
> Intermediate results are saved to `cache/` after the first run, so detection is skipped and the script runs much faster.
>
> **When to delete cache files:**
>
> | Scenario | `detections_cache.json` | `building_detections_map_cache.json` |
> |----------|------------------------|--------------------------------------|
> | **Building list changed only**<br>(same images, different building list) | ✅ Can reuse | ❌ Must delete |
> | **Images also changed** | ❌ Must delete | ❌ Must delete |
>
> **Example deletion commands:**
> ```bash
> # If only building list changed
> rm cache/building_detections_map_cache.json
>
> # If images also changed (delete all cache)
> rm cache/*.json
> ```
>
> Or set `USE_DETECTION_CACHE = False` at the top of the script.

#### STEP 2: Merge Detection Images (Optional)

```bash
python merge_images_with_tfw.py
```

Stitches all annotated tile images from `detection_results_images/` into a single large image.

---

### Configuration Parameters

Edit the constants at the top of the script to adjust behavior.

| Constant | Default | Description |
|----------|---------|-------------|
| `EPSG_CODE` | 6677 | **Coordinate system of input images (Japan Plane Rectangular CS)** <br> **⚠️ MUST be changed according to your target region** <br> 6677 = Zone IX (Tokyo, Kanagawa, etc.)<br> 6669 = Zone I (Western Hokkaido), 6670 = Zone II (Eastern Hokkaido)<br> 6671 = Zone III (Tohoku), 6672 = Zone IV (Kanto)<br> 6673 = Zone V (Hokuriku), 6674 = Zone VI (Chubu)<br> 6675 = Zone VII (Kinki), 6676 = Zone VIII (Chugoku)<br> 6678 = Zone X (Northern Kyushu), 6679 = Zone XI (Southern Kyushu, Okinawa), etc. |
| `YOLO_BATCH_SIZE` | 8 | Number of images processed at once (reduce if memory is limited) |
| `USE_DETECTION_CACHE` | `True` | Set to `False` to force reprocessing from scratch |

---

### Notes

- **⚠️ MUST change `EPSG_CODE` to match your target region's plane rectangular coordinate system**  
  Default is Tokyo area (Zone IX, EPSG:6677). Using the wrong coordinate system will cause significant positioning errors.
- Always activate the conda environment (`conda activate yolo_py_env`) before running.
- Building boundary XMLs must be in the Japan Geospatial Information Authority (GSI) Fundamental Geospatial Data (FGD) BldA GML format.
- Ensure PNG images are present in `input/image_cut/png/` before running STEP 1.
