# 航空写真からの建物設備検出システム（YOLOv8 OBB）
# Building Equipment Detection from Aerial Imagery (YOLOv8 OBB)

---
## 前提・準備 📋
- 実行環境：Windows11のUbuntu 24.04.3 LTS
- 仮想環境：Python 3.10.20
- 仮想環境は以下のコマンドを実行して構築可能
```bash
conda env create -f yolo_env.yml
```


## 日本語

### 概要

航空写真（高解像度衛星画像）を用いて、建物屋上に設置された熱源設備を自動検出し、建物リスト（ZENRIN建物ポイントデータ）に設備種別（`plant` 列）を付与するシステムです。全国どの都道府県のデータでも、実行時に対象地域を入力するだけで利用できます。

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
├── tile_geotiff.py                        # GeoTIFF（生の航空写真）をタイル分割するツール
├── area_config.py                         # 対象地域→平面直角座標系(EPSG)の対話的決定
├── merge_images_with_tfw.py               # 検出結果画像の統合ツール
├── models/
│   └── best.pt                            # 学習済みモデル（別途入手）
├── input/
│   ├── create_tile/                       # タイル分割前の生GeoTIFFを置く場所
│   │   └── 1_0001.tif
│   ├── image_cut/
│   │   ├── png/                           # 入力画像（**分割済み航空写真、1000×750px**）
│   │   └── tfw/                           # 各画像の座標ファイル（TFW形式,tile_geotiff.pyで生成）
│   └── building_list/
│       └── <地名>.csv                     # 建物リスト（ZENRIN建物ポイントデータ）
├── bld_boundary/
│   └── FG-GML-*.xml                       # 建物境界線 XML（国土地理院 FGD BldA）
├── cache/                                 # 実行後に自動生成（地域ごとに {地名}_detections_cache.json 等）
├── detection_results_images/              # 検出結果の可視化画像（実行後に自動生成）
└── output/
    └── <地名>.csv                         # 出力 CSV（plant 列追加済み）
```

---

### セットアップ手順

#### 1. リポジトリのクローン

```bash
git clone git@github.com:uelab-env/STEP02-Detecting-Building-Plant-byYOLOmodel.git
cd STEP02-Detecting-Building-Plant-byYOLOmodel
```
[リポジトリURL]　：git@github.com:uelab-env/STEP02-Detecting-Building-Plant-byYOLOmodel.git
[リポジトリ名]　：STEP02-Detecting-Building-Plant-byYOLOmodel

#### 2. 開発環境の構築

conda環境定義ファイル（`yolo_env.yml`）から仮想環境を構築します。必要なパッケージ（`ultralytics`, `torch`, `shapely`, `pyproj`, `pandas`, `Pillow`, `natsort` など）はすべてこのファイルに含まれているため、これ以外のインストール作業は不要です。

```bash
conda env create -f yolo_env.yml
conda activate yolo_env
```

#### 3. GPU を使用する場合

`yolo_env.yml` には CUDA 対応版の `torch` / `torchvision` が既に含まれているため、追加のインストール作業は不要です。NVIDIA GPU が搭載された環境では自動的に使用されます（`nvidia-smi` コマンドでドライバがyml内のCUDAバージョンに対応しているか確認してください）。GPUが無い環境ではCPUで自動的に動作します（追加設定不要）。

#### 4. 学習済みモデルの配置

`models/best.pt` に学習済みモデルファイルを配置してください
これは、西村がデフォルトで配置してます
※学習済みモデルの生成方法に関してのリポジトリは、10月末までに作成

---

### 入力データの準備

| 配置先 | 内容 |
|--------|------|
| `input/create_tile/` | タイル分割前の生GeoTIFF。`tile_geotiff.py` で `png/`・`tfw/` に変換します |
| `input/image_cut/png/` | 航空写真の PNG 画像（**1000×750px**） |
| `input/image_cut/tfw/` | 対応する TFW ファイル（ファイル名は PNG と同一,tile_geotiff.pyで生成） |
| `input/building_list/<地名>.csv` | 建物リスト（ZENRIN建物ポイントデータ）。`緯度`・`経度` 列が必要 |
| `bld_boundary/` | 建物境界線 XML（国土地理院「基盤地図情報 建物外周」FGD BldA 形式） |

- 入力ファイルの配置
1. 業務モデルの[STEP01](https://github.com/uelab-env/Building-Usage-Determination_py)の以下のパスの"地名".csvを入力ファイルとする
```01_Building-Usage-Determination\Building-Usage-Determination_py\"地名"\BuildingUsageDetermination_Chuo\"地名".csv```
2. `input/building_list/` ディレクトリに、対象地域の"地名".csvを配置してください。
3. **コードの編集は不要です。** `detect_object_YOLO_carobock_ver8_1.py` を実行すると、使用するCSVファイルと対象地域（都道府県）を対話的に選択できます（詳細は「実行方法」参照）。

- 建物境界線データは以下のリンクを参考にダウンロードしてください
https://uelab.growi.cloud/6a6c68099b9a6f8caadc8172

#### 建物リスト（ZENRIN建物ポイントデータ）に必要な列

| 列名 | 説明 |
|------|------|
| `緯度` | 建物代表点の緯度（WGS84） |
| `経度` | 建物代表点の経度（WGS84） |

その他の列はそのまま出力 CSV に引き継がれます。

---
## 参考程度に
### GeoTIFFタイル分割について（`tile_geotiff.py`）

航空写真がタイル分割済みのPNG+TFWではなく、位置情報付きの1枚の大きなGeoTIFF（例: `1_0001.tif`）として提供される場合、検出スクリプトが前提とする1000×750pxのPNG+TFW形式へあらかじめ変換する必要があります。

**TFW（ワールドファイル）とは**

画像ファイルに位置情報を付与するための、6行のテキストファイルです。

```
line1: A  x方向のピクセルサイズ（m/px）
line2: D  回転項（通常0）
line3: B  回転項（通常0）
line4: E  y方向のピクセルサイズ（m/px、通常負の値）
line5: C  左上ピクセル中心のX座標
line6: F  左上ピクセル中心のY座標
```

ピクセル座標 `(px, py)` から実座標への変換は `X = A*px + B*py + C`、`Y = D*px + E*py + F` で計算されます（`detect_object_YOLO_carobock_ver8_1.py` の `pixel_to_latlng()` 参照）。

**ModelTransformationTag とは**

GeoTIFF規格でジオリファレンス情報を埋め込むためのタグ（タグ番号34264）です。4×4のアフィン変換行列として、TFWと同等の情報（回転・スケール・平行移動）を、タイル分割せずに1枚のTIFFファイル内に保持できます。`tile_geotiff.py` はこのタグを読み取り、各タイルの左上座標に応じて平行移動成分だけを再計算し、TFWとして書き出します。

**半ピクセル補正について**

GeoTIFFにはさらに `GTRasterTypeGeoKey` というキーがあり、アフィン変換の原点がピクセルの「外角（Area規約、値=1）」を指すか「中心（Point規約、値=2）」を指すかを区別します。TFWは規約上「左上ピクセルの中心」を原点として扱うため、元データがArea規約（多くのGeoTIFFはこちら）の場合、変換時に0.5ピクセル分のオフセット補正が必要です。これを怠ると、全ての検出結果が実座標で数cm系統的にズレます。`tile_geotiff.py` はこの補正を自動的に行います。

---  


**使い方**

```bash
python tile_geotiff.py
```

- タイル分割したいGeoTIFFを `input/create_tile/` に配置してください（複数配置した場合、そのフォルダ内の全tifファイルがまとめて処理されます）。コードの編集は不要です。
- 出力は既定で `input/image_cut/png/`・`input/image_cut/tfw/` に書き込まれ、`{元ファイル名}_r{行}_c{列}.png/.tfw` という名前になります。
- 画像サイズがタイルサイズ（1000×750px）で割り切れない場合でも、パディングを行わずに画像全体をカバーできるよう、必要なオーバーラップ率を自動算出し、タイル間に均等に分配します（画像サイズがどのようなものであっても同じアルゴリズムで対応します）。実行時にログへ算出されたオーバーラップ率（例: 列方向2.91%・行方向2.05%）が表示されます。タイルサイズで割り切れる場合はオーバーラップ率0%になります。
- 元のGeoTIFFにはCRS（座標系）の情報が含まれないため、このスクリプトはピクセル→実座標のアフィン変換のみをタイルへ伝播します。どのEPSGコードを使うかは検出スクリプト実行時に対話的に決定します。

> **⚠️ 地域を切り替える際の注意**
> `input/image_cut/png/`・`tfw/` に別地域のタイルが既に存在する状態で `tile_geotiff.py` を実行すると警告が表示されます。異なる地域のタイルが同じフォルダに混在した状態で検出スクリプトを実行すると、単一のEPSGコードで全タイルが処理されてしまい、一部の建物の位置情報が誤って計算されます。地域を切り替える際は、事前に古いタイルを別フォルダへ退避してください。

---

### 実行方法

#### STEP 1: 設備検出・判定の実行

```bash
conda activate yolo_env
python detect_object_YOLO_carobock_ver8_1.py
```

実行すると、まず対話的に以下を選択します（コードの編集は不要です）。

```
複数の建物リスト（ZENRIN建物ポイントデータ）CSVが見つかりました。使用するファイルを選択してください:
  1. TokyoChuo.csv
  2. Takarazuka.csv
番号を入力 [1-2] > 2

対象地域の平面直角座標系（EPSGコード）を設定します。
入力例: 都道府県名（東京都 / 兵庫県）、系番号（IX / 9 / 系9）、EPSGコード直接指定（6677 / EPSG:6677）
対象地域 > 兵庫県
  → EPSG:6673（系V）を使用します。
  よろしいですか？ [Y/n] > y
```

- 建物リストCSVが1件しかない場合は自動選択されます。
- 都道府県が複数の系にまたがる場合（東京都・北海道・鹿児島県・沖縄県）は、さらにサブ選択のプロンプトが表示されます。
- 都道府県名の代わりに `6673` や `EPSG:6673`、系番号 `V` / `5` / `系5` を直接入力することもできます。

実行が完了すると、`output/<選択したCSVのファイル名>.csv` に `plant` 列が追加された CSV が出力されます。
`plant` 列の値：`CT` / `ACC` / `MUL` / `PAC` / 空文字（検出なし）

> **2回目以降の実行について**
> 初回実行後は `cache/` フォルダに `{選択したCSV名}_detections_cache.json` 等の形で中間結果が保存されるため、同じ地域を再選択して実行すると検出処理がスキップされ高速に動作します。キャッシュファイル名は地域ごとに自動で分離されるため、**異なる地域を選択しても前回のキャッシュが誤って再利用されることはありません**。
>
> **キャッシュファイルの削除が必要なケース：**
>
> | シナリオ | `{地域}_detections_cache.json` | `{地域}_building_detections_map_cache.json` |
> |---------|------------------------|-------------------------------------|
> | **同じ地域で建物リストの内容のみ更新** | ✅ 再利用可能 | ❌ 削除が必要 |
> | **同じ地域で画像（png/tfw）も更新** | ❌ 削除が必要 | ❌ 削除が必要 |
> | **別の地域を選択** | 対応不要（自動的に別ファイル名になる） | 対応不要（自動的に別ファイル名になる） |
>
> **削除コマンド例：**
> ```bash
> # 例: TokyoChuo の建物リストのみ更新した場合
> rm cache/TokyoChuo_building_detections_map_cache.json
>
> # 画像も更新した場合（該当地域の全キャッシュ削除）
> rm cache/TokyoChuo_*.json
> ```
>
> または、スクリプト冒頭の `USE_DETECTION_CACHE = False` に変更してください。

#### STEP 2: 検出結果画像の統合（任意）

```bash
python merge_images_with_tfw.py
```

STEP 1 で生成された検出結果の可視化画像（`detection_results_images/`）を、1枚の画像に統合します。

---

### モデルの実行単位

YOLOモデルによる検出は、`input/image_cut/png/` にある **1000×750pxのタイル1枚単位** で実行されます。オルソモザイク画像全体（例: `1_0001.tif` の14592×25728px）を一括で推論にかけることはありません。これは学習済みモデル自体がこのタイルサイズで学習されていることに加え、メモリ使用量を抑えるためです（生のGeoTIFFしかない場合に `tile_geotiff.py` によるタイル分割が必須である理由でもあります）。

`YOLO_BATCH_SIZE`（既定8）は、この「1000×750pxタイル」を一度に何枚まとめて 処理するかを制御するパラメータです。検出の単位や精度そのものを変えるものではなく、純粋にスループット・メモリ使用量の調整用です。メモリが不足する場合は値を小さく、余裕がある場合は大きくしてください。

---

### 設定パラメータ

スクリプト冒頭の定数を変更することで動作を調整できます。対象地域（建物リストCSV・EPSGコード）は実行時の対話プロンプトで指定するため、コード編集は不要です（`area_config.py` 参照）。

| 定数 | デフォルト値 | 説明 |
|------|------------|------|
| `YOLO_BATCH_SIZE` | 8 | 一度に処理する画像枚数（メモリに応じて調整） |
| `USE_DETECTION_CACHE` | `True` | `False` にすると検出を最初から再実行 |

`tile_geotiff.py` 側の定数（生GeoTIFFを使う場合に調整）:

| 定数 | デフォルト値 | 説明 |
|------|------------|------|
| `SOURCE_TIF_DIR` | `input/create_tile` | タイル分割対象のGeoTIFFを置くフォルダ（中の全`*.tif`を処理） |
| `TILE_WIDTH` / `TILE_HEIGHT` | 1000 / 750 | 出力タイルサイズ（px） |

#### 平面直角座標系（系）と都道府県の対応表

`area_config.py` の `prompt_epsg_code()` は、以下の国土地理院の公式区分に基づいて都道府県名からEPSGコードを解決します（東京都・北海道・鹿児島県・沖縄県は複数系にまたがるためサブ選択が入ります）。

| 系 | EPSG | 対象都道府県 |
|----|------|-------------|
| I | 6669 | 長崎県／鹿児島県の一部離島（奄美群島など） |
| II | 6670 | 福岡県・佐賀県・熊本県・大分県・宮崎県・鹿児島県（本土） |
| III | 6671 | 山口県・島根県・広島県 |
| IV | 6672 | 香川県・愛媛県・徳島県・高知県 |
| V | 6673 | 兵庫県・鳥取県・岡山県 |
| VI | 6674 | 京都府・大阪府・福井県・滋賀県・三重県・奈良県・和歌山県 |
| VII | 6675 | 石川県・富山県・岐阜県・愛知県 |
| VIII | 6676 | 新潟県・長野県・山梨県・静岡県 |
| IX | 6677 | 東京都（本土）・福島県・栃木県・茨城県・埼玉県・千葉県・群馬県・神奈川県 |
| X | 6678 | 青森県・秋田県・山形県・岩手県・宮城県 |
| XI | 6679 | 北海道（道南西部：小樽市・函館市・伊達市など） |
| XII | 6680 | 北海道（道央：札幌市・旭川市・稚内市など） |
| XIII | 6681 | 北海道（道東：北見市・帯広市・釧路市・網走市・根室市など） |
| XIV | 6682 | 東京都小笠原諸島 |
| XV | 6683 | 沖縄県（沖縄本島など、東経126°-130°） |
| XVI | 6684 | 沖縄県（宮古・八重山諸島、東経126°以西） |
| XVII | 6685 | 沖縄県（大東諸島、東経130°以東） |
| XVIII | 6686 | 東京都沖ノ鳥島 |
| XIX | 6687 | 東京都南鳥島 |

都道府県名の代わりに、系番号（`IX` / `9` / `系9`）やEPSGコード（`6677` / `EPSG:6677`）を直接入力することも常に可能です。

---

### 注意事項

- conda 環境を有効化（`conda activate yolo_env`）してから実行してください。
- 建物境界 XML は国土地理院「基盤地図情報 建物外周」（FGD BldA）の GML 形式のみ対応しています。
- STEP 1 を実行する前に、`input/image_cut/png/` に PNG 画像が存在することを確認してください。生のGeoTIFFしかない場合は、先に `tile_geotiff.py` を実行してください。
- **地域を切り替える際は、`input/image_cut/png/`・`tfw/` の中身を切り替え先の地域のタイルだけにしてください。** 複数地域のタイルが混在すると、単一のEPSGコードで全タイルが処理され、位置情報が誤って計算されます。
- キャッシュは地域ごとに自動分離されますが、同一地域の画像を更新した場合は引き続き該当キャッシュの手動削除が必要です。

---

---

## English

### Overview

This system automatically detects heat-source equipment installed on building rooftops from high-resolution aerial/satellite imagery. An equipment-type label (`plant` column) is appended to the input building list CSV. It works with data from any prefecture in Japan — just enter the target region when you run it.

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
├── tile_geotiff.py                        # Tiles a raw GeoTIFF into PNG+TFW tiles
├── area_config.py                         # Interactively resolves target region -> EPSG code
├── merge_images_with_tfw.py               # Detection-image merging utility
├── models/
│   └── best.pt                            # Trained model weights (obtain separately)
├── input/
│   ├── create_tile/                       # place raw GeoTIFFs here before tiling
│   │   └── 1_0001.tif
│   ├── image_cut/
│   │   ├── png/                           # Aerial image tiles (1000×750 px)
│   │   └── tfw/                           # Coordinate files for each image (TFW format)
│   └── building_list/
│       └── <AreaName>.csv                 # Building list (ZENRIN building point dataset)
├── bld_boundary/
│   └── FG-GML-*.xml                       # Building boundary XMLs (GSI FGD BldA format)
├── cache/                                 # Auto-generated on first run ({AreaName}_detections_cache.json, etc., per region)
├── detection_results_images/              # Annotated detection images (auto-generated)
└── output/
    └── <AreaName>.csv                     # Output CSV with `plant` column added
```

---

### Setup

#### 1. Clone the Repository

```bash
git clone <repository URL>
cd <repository name>
```

#### 2. Create the conda Environment

Build the environment from the `yolo_env.yml` definition file. It already includes every required package (`ultralytics`, `torch`, `shapely`, `pyproj`, `pandas`, `Pillow`, `natsort`, etc.), so no further installation step is needed.

```bash
conda env create -f yolo_env.yml
conda activate yolo_env
```

#### 3. GPU Acceleration

`yolo_env.yml` already includes a CUDA-enabled build of `torch`/`torchvision`, so no extra installation is required. It will be used automatically on machines with an NVIDIA GPU (check `nvidia-smi` to confirm your driver supports the CUDA version bundled in the yml). Machines without a GPU fall back to CPU automatically — no configuration needed.

#### 4. Place the Model Weights

Place the trained model file at `models/best.pt` (must be obtained separately).

---

### Preparing Input Data

| Location | Contents |
|----------|----------|
| `input/create_tile/` | (Optional) raw GeoTIFF before tiling. Convert it with `tile_geotiff.py` into `png/`/`tfw/` |
| `input/image_cut/png/` | Tiled aerial PNG images (1000×750 px) |
| `input/image_cut/tfw/` | Corresponding TFW files (filenames must match the PNG files) |
| `input/building_list/<AreaName>.csv` | Building list (ZENRIN building point dataset). Must include `緯度`/`経度` columns |
| `bld_boundary/` | Building boundary XMLs (Japan GSI Fundamental Geospatial Data, FGD BldA GML format) |

- **Prepare the input files**

1. Use the following CSV file generated by **STEP01** as the input file:

   ```
   01_Building-Usage-Determination/Building-Usage-Determination_py/<AreaName>/BuildingUsageDetermination_Chuo/<AreaName>.csv
   ```

2. Place the target building list (ZENRIN building point dataset) CSV file (`<AreaName>.csv`) in the `input/building_list/` directory.

3. **No code editing is required.** Running `detect_object_YOLO_carobock_ver8_1.py` will interactively prompt you to select the CSV file and the target region (prefecture) — see "Running the System" below.

- **Download the building boundary data**

  Please download the building boundary data by referring to the following link:

  https://uelab.growi.cloud/6a6c68099b9a6f8caadc8172

#### Required Columns in the Building List (ZENRIN Building Point Dataset)

| Column | Description |
|--------|-------------|
| `緯度` | Building centroid latitude (WGS84) |
| `経度` | Building centroid longitude (WGS84) |

All other columns are passed through to the output CSV unchanged.

---

### GeoTIFF Tiling (`tile_geotiff.py`)

If your aerial imagery is provided as a single large georeferenced GeoTIFF (e.g. `1_0001.tif`) rather than pre-tiled PNG+TFW pairs, you must convert it into the 1000×750 px PNG+TFW format the detection script expects.

**What a TFW (world file) is**

A 6-line text file that attaches georeferencing to an image:

```
line1: A  pixel size in the x direction (m/px)
line2: D  rotation term (usually 0)
line3: B  rotation term (usually 0)
line4: E  pixel size in the y direction (m/px, usually negative)
line5: C  X coordinate of the top-left pixel's center
line6: F  Y coordinate of the top-left pixel's center
```

Pixel coordinates `(px, py)` map to real-world coordinates via `X = A*px + B*py + C`, `Y = D*px + E*py + F` (see `pixel_to_latlng()` in `detect_object_YOLO_carobock_ver8_1.py`).

**What `ModelTransformationTag` is**

A GeoTIFF tag (tag number 34264) that embeds georeferencing directly inside a TIFF file, as a 4×4 affine transformation matrix carrying the same information (rotation, scale, translation) as a TFW — without needing the image to be tiled. `tile_geotiff.py` reads this tag and, for each output tile, recomputes only the translation component based on the tile's top-left offset, writing the result out as a TFW file.

**About the half-pixel correction**

GeoTIFF also defines a `GTRasterTypeGeoKey`, which distinguishes whether the affine transform's origin refers to a pixel's outer corner ("Area" convention, value=1) or its center ("Point" convention, value=2). TFW always treats the top-left pixel's *center* as its origin, so when the source data uses the Area convention (true for most GeoTIFFs), a 0.5-pixel offset correction is required during conversion. Skipping this would introduce a systematic few-centimeter error into every detected coordinate. `tile_geotiff.py` applies this correction automatically.

**Usage**

```bash
python tile_geotiff.py
```

- Place the GeoTIFF(s) you want to tile in `input/create_tile/` (if more than one file is present, all of them are processed in one run). No code editing required.
- Output is written by default to `input/image_cut/png/` and `input/image_cut/tfw/`, named `{source_basename}_r{row}_c{col}.png/.tfw`.
- When the image dimensions aren't evenly divisible by the tile size, the script automatically computes the overlap ratio needed to cover the full image with no padding and distributes it evenly across all tiles (the same algorithm works regardless of the source image's size). The computed overlap ratio (e.g. "2.91% horizontally, 2.05% vertically") is printed at run time. Dimensions that divide evenly produce 0% overlap.
- The source GeoTIFF carries no CRS information, so this script only propagates the pixel→raw-coordinate affine transform into each tile; which EPSG code to use is resolved interactively at detection time.

> **⚠️ Note when switching regions**
> If `input/image_cut/png/`/`tfw/` already contain tiles from a different region, `tile_geotiff.py` will print a warning. Running the detection script against a folder mixing tiles from multiple regions will process all of them under a single EPSG code, producing incorrect coordinates for whichever region doesn't match. Archive the old tiles elsewhere before switching regions.

---

### Running the System

#### STEP 1: Run Equipment Detection

```bash
conda activate yolo_env
python detect_object_YOLO_carobock_ver8_1.py
```

Running it first prompts you interactively for the following (no code editing required):

```
複数の建物リスト（ZENRIN建物ポイントデータ）CSVが見つかりました。使用するファイルを選択してください:
  1. TokyoChuo.csv
  2. Takarazuka.csv
番号を入力 [1-2] > 2

対象地域の平面直角座標系（EPSGコード）を設定します。
入力例: 都道府県名（東京都 / 兵庫県）、系番号（IX / 9 / 系9）、EPSGコード直接指定（6677 / EPSG:6677）
対象地域 > 兵庫県
  → EPSG:6673（系V）を使用します。
  よろしいですか？ [Y/n] > y
```

- If only one building-list CSV exists, it's selected automatically.
- If the selected prefecture spans multiple zones (Tokyo, Hokkaido, Kagoshima, Okinawa), a further sub-region prompt appears.
- Instead of a prefecture name, you can always enter an EPSG code (`6673` / `EPSG:6673`) or a zone identifier (`V` / `5` / `系5`) directly.

When complete, `output/<selected CSV filename>.csv` will be created with a `plant` column added.
Possible values: `CT` / `ACC` / `MUL` / `PAC` / empty string (no equipment detected)

> **From the second run onwards**
> Intermediate results are saved to `cache/` as `{selected CSV name}_detections_cache.json`, etc., after the first run — re-running with the same region selected skips detection and runs much faster. Cache filenames are automatically namespaced per region, so **selecting a different region will never accidentally reuse another region's stale cache**.
>
> **When to delete cache files:**
>
> | Scenario | `{region}_detections_cache.json` | `{region}_building_detections_map_cache.json` |
> |----------|------------------------|--------------------------------------|
> | **Same region, building list content updated only** | ✅ Can reuse | ❌ Must delete |
> | **Same region, images (png/tfw) also updated** | ❌ Must delete | ❌ Must delete |
> | **A different region is selected** | No action needed (automatically a different file) | No action needed (automatically a different file) |
>
> **Example deletion commands:**
> ```bash
> # e.g. only the TokyoChuo building list was updated
> rm cache/TokyoChuo_building_detections_map_cache.json
>
> # images were also updated (delete all cache for that region)
> rm cache/TokyoChuo_*.json
> ```
>
> Or set `USE_DETECTION_CACHE = False` at the top of the script.

#### STEP 2: Merge Detection Images (Optional)

```bash
python merge_images_with_tfw.py
```

Stitches all annotated tile images from `detection_results_images/` into a single large image.

---

### Model Execution Unit

YOLO inference runs on **individual 1000×750 px tiles** in `input/image_cut/png/` — it never processes an entire orthomosaic (e.g. `1_0001.tif`'s 14592×25728 px) in one pass. This matches the tile size the model was trained on, and keeps memory usage bounded (it's also why tiling with `tile_geotiff.py` is mandatory when starting from a raw GeoTIFF).

`YOLO_BATCH_SIZE` (default 8) controls how many of these 1000×750 px tiles are grouped into a single `model.predict()` call. It's purely a throughput/memory knob — it does not change the detection unit or accuracy. Lower it if you run out of memory, raise it if you have headroom.

---

### Configuration Parameters

Edit the constants at the top of the script to adjust behavior. The target region (building list CSV / EPSG code) is now selected interactively at runtime, so no code editing is needed for that (see `area_config.py`).

| Constant | Default | Description |
|----------|---------|-------------|
| `YOLO_BATCH_SIZE` | 8 | Number of images processed at once (reduce if memory is limited) |
| `USE_DETECTION_CACHE` | `True` | Set to `False` to force reprocessing from scratch |

Constants in `tile_geotiff.py` (adjust when tiling a raw GeoTIFF):

| Constant | Default | Description |
|----------|---------|-------------|
| `SOURCE_TIF_DIR` | `input/create_tile` | Folder holding the GeoTIFF(s) to tile (every `*.tif` inside is processed) |
| `TILE_WIDTH` / `TILE_HEIGHT` | 1000 / 750 | Output tile size (px) |

#### Plane Rectangular CS Zone ↔ Prefecture Table

`area_config.py`'s `prompt_epsg_code()` resolves a prefecture name to an EPSG code using the following official 国土地理院 (GSI) zone assignments (Tokyo, Hokkaido, Kagoshima, and Okinawa span multiple zones and trigger a sub-region prompt).

| Zone | EPSG | Prefectures |
|------|------|-------------|
| I | 6669 | Nagasaki-ken; some remote islands of Kagoshima-ken (Amami islands, etc.) |
| II | 6670 | Fukuoka-ken, Saga-ken, Kumamoto-ken, Oita-ken, Miyazaki-ken, Kagoshima-ken (mainland) |
| III | 6671 | Yamaguchi-ken, Shimane-ken, Hiroshima-ken |
| IV | 6672 | Kagawa-ken, Ehime-ken, Tokushima-ken, Kochi-ken |
| V | 6673 | Hyogo-ken, Tottori-ken, Okayama-ken |
| VI | 6674 | Kyoto-fu, Osaka-fu, Fukui-ken, Shiga-ken, Mie-ken, Nara-ken, Wakayama-ken |
| VII | 6675 | Ishikawa-ken, Toyama-ken, Gifu-ken, Aichi-ken |
| VIII | 6676 | Niigata-ken, Nagano-ken, Yamanashi-ken, Shizuoka-ken |
| IX | 6677 | Tokyo-to (mainland), Fukushima-ken, Tochigi-ken, Ibaraki-ken, Saitama-ken, Chiba-ken, Gunma-ken, Kanagawa-ken |
| X | 6678 | Aomori-ken, Akita-ken, Yamagata-ken, Iwate-ken, Miyagi-ken |
| XI | 6679 | Hokkaido (SW: Otaru, Hakodate, Date, etc.) |
| XII | 6680 | Hokkaido (central: Sapporo, Asahikawa, Wakkanai, etc.) |
| XIII | 6681 | Hokkaido (east: Kitami, Obihiro, Kushiro, Abashiri, Nemuro, etc.) |
| XIV | 6682 | Tokyo-to Ogasawara Islands |
| XV | 6683 | Okinawa-ken (main island, etc., 126°E–130°E) |
| XVI | 6684 | Okinawa-ken (Miyako/Yaeyama Islands, west of 126°E) |
| XVII | 6685 | Okinawa-ken (Daito Islands, east of 130°E) |
| XVIII | 6686 | Tokyo-to Okinotorishima |
| XIX | 6687 | Tokyo-to Minamitorishima |

You can always enter a zone identifier (`IX` / `9` / `系9`) or an EPSG code (`6677` / `EPSG:6677`) directly instead of a prefecture name.

---

### Notes

- Activate the conda environment (`conda activate yolo_env`) before running.
- Building boundary XMLs are only supported in the GSI "Fundamental Geospatial Data — Building Outlines" (FGD BldA) GML format.
- Confirm `input/image_cut/png/` contains PNG images before running STEP 1. If you only have a raw GeoTIFF, run `tile_geotiff.py` first.
- **When switching regions, make sure `input/image_cut/png/`/`tfw/` contain only the tiles for the region you're about to process.** Mixing tiles from multiple regions causes all of them to be processed under a single EPSG code, producing incorrect coordinates.
- Cache files are automatically namespaced per region, but updating the images for an already-cached region still requires manually deleting that region's cache.
