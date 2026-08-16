# -*- coding: utf-8 -*-
"""
input/create_tile/ に置かれた任意サイズのGeoTIFFを1000x750px単位でタイル分割し、
detect_object_YOLO_carobock_ver8_1.py が読み込めるPNG+TFW形式で出力するスクリプト。

- タイルサイズで割り切れない場合でも、パディングなしで元画像全体をカバーできるよう、
  必要なオーバーラップ率を自動算出し、タイル間に均等に分配する
  （画像サイズに関わらず同じアルゴリズムで対応できる）。
- 元GeoTIFFの ModelTransformationTag（GeoTIFFタグ34264）から
  ピクセル→実座標のアフィン変換を読み取り、各タイル用に平行移動成分だけを
  再計算してTFWとして書き出す。
- GTRasterTypeGeoKey（GeoKeyDirectoryTag内、ID=1025）が
  RasterPixelIsArea（値=1、既定）の場合、原点はピクセルの外角を指すため、
  TFWが要求する「左上ピクセル中心」に合わせて半ピクセル分の補正を行う。
- 元GeoTIFFにはCRS/EPSGの情報が含まれないため、このスクリプトはCRSに一切
  関与しない。どのEPSGを使うかは detect_object_YOLO_carobock_ver8_1.py
  実行時に area_config 経由で対話的に決定する。
"""

import os
import sys
import glob
import math

from PIL import Image

# =====================================================
# 定数定義
# =====================================================
SOURCE_TIF_DIR = os.path.join("input", "create_tile")  # ここにtifファイルを配置する
OUTPUT_PNG_DIR = os.path.join("input", "image_cut", "png")
OUTPUT_TFW_DIR = os.path.join("input", "image_cut", "tfw")
TILE_WIDTH = 1000
TILE_HEIGHT = 750

MODEL_TRANSFORMATION_TAG = 34264
GEO_KEY_DIRECTORY_TAG = 34735
GT_RASTER_TYPE_GEO_KEY_ID = 1025


# =====================================================
# タイル分割オフセット計算
# =====================================================
def compute_tile_offsets(total_size, tile_size):
    """
    total_size を tile_size のタイルで隙間なくカバーする各タイルの開始オフセットを返す。
    画像サイズが tile_size で割り切れない場合、必要なタイル数(n)を算出したうえで、
    必要なオーバーラップをタイル間(n-1箇所)に均等に分配する。これにより、
    どのようなサイズの画像でも同じアルゴリズムでパディングなしにカバーできる
    （特定の1箇所だけにオーバーラップが偏ることがない）。

    戻り値: (offsets: list[int], overlap_px: float, overlap_ratio: float)
    """
    if total_size <= tile_size:
        # 画像がタイルサイズ以下 -> タイル1枚のみ（オーバーラップの概念なし）
        return [0], 0.0, 0.0

    n = math.ceil(total_size / tile_size)
    if n == 1:
        return [0], 0.0, 0.0

    # 隣接タイルの開始位置の間隔（ストライド）。tile_sizeちょうどなら重複ゼロ、
    # それより小さければその分だけ隣接タイルと重複する。
    stride = (total_size - tile_size) / (n - 1)
    offsets = [round(i * stride) for i in range(n)]
    offsets[-1] = total_size - tile_size  # 丸め誤差対策で最後は厳密値に固定

    overlap_px = tile_size - stride
    overlap_ratio = overlap_px / tile_size
    return offsets, overlap_px, overlap_ratio


# =====================================================
# GeoTIFFタグ読み込み
# =====================================================
def read_model_transformation(im, tif_path):
    """PIL Image から ModelTransformationTag(34264) を読み、
    a, b, tx, d, e, ty を返す（一般アフィン変換の回転項を含む係数）。
    X = a*px + b*py + tx, Y = d*px + e*py + ty"""
    if MODEL_TRANSFORMATION_TAG not in im.tag_v2:
        raise ValueError(
            f"{tif_path} に ModelTransformationTag(34264) が見つかりません。"
            "GeoTIFFとしてジオリファレンス情報を持たないファイルの可能性があります。"
        )
    mtt = im.tag_v2[MODEL_TRANSFORMATION_TAG]  # 16要素タプル（4x4行列、行優先）
    a, b, tx = mtt[0], mtt[1], mtt[3]
    d, e, ty = mtt[4], mtt[5], mtt[7]
    return a, b, tx, d, e, ty


def is_area_convention(im):
    """GeoKeyDirectoryTag(34735)内のGTRasterTypeGeoKey(ID=1025)を確認する。
    1=RasterPixelIsArea（半ピクセル補正が必要）、2=RasterPixelIsPoint（不要）。
    キーが見つからない場合はGeoTIFF既定のArea規約とみなし警告を出す。"""
    gk = im.tag_v2.get(GEO_KEY_DIRECTORY_TAG)
    if not gk:
        print("[警告] GeoKeyDirectoryTagが見つかりません。RasterPixelIsArea規約を仮定します。")
        return True

    n_keys = gk[3]
    for i in range(n_keys):
        key_id, _loc, _count, value = gk[4 + i * 4: 8 + i * 4]
        if key_id == GT_RASTER_TYPE_GEO_KEY_ID:
            if value == 2:
                return False
            if value != 1:
                print(f"[警告] 未知のGTRasterTypeGeoKey={value}です。RasterPixelIsArea規約を仮定します。")
            return True

    print("[警告] GTRasterTypeGeoKeyが指定されていません。RasterPixelIsArea規約を仮定します。")
    return True


# =====================================================
# TFW書き出し
# =====================================================
def write_tfw(tfw_path, a, b, tx, d, e, ty, x0, y0, half_pixel_correct=True):
    """タイル左上コーナー（ソース画像ピクセル座標 x0, y0）から、そのタイル用の
    TFWを書き出す。RasterPixelIsArea規約では原点がピクセルの外角を指すため、
    TFWが要求する「ピクセル中心」に合わせて +0.5px のオフセットを加える。
    回転・スケール成分（a, b, d, e）はタイル間で不変、平行移動成分のみ再計算する。"""
    ox, oy = (0.5, 0.5) if half_pixel_correct else (0.0, 0.0)
    new_c = a * (x0 + ox) + b * (y0 + oy) + tx
    new_f = d * (x0 + ox) + e * (y0 + oy) + ty
    with open(tfw_path, 'w') as f:
        f.write(f"{a}\n")
        f.write(f"{d}\n")
        f.write(f"{b}\n")
        f.write(f"{e}\n")
        f.write(f"{new_c}\n")
        f.write(f"{new_f}\n")


# =====================================================
# 出力先の別データセット混在チェック
# =====================================================
def check_existing_tiles_for_mixing(output_png_dir, source_basenames):
    """出力先に、今回処理する予定のないデータセット由来のタイルが残っていないか
    確認する。混在した状態で検出スクリプトを実行すると、複数地域の画像が単一の
    EPSGで処理され、緯度経度が誤って計算される危険があるため、既定では中断する。"""
    expected_prefixes = tuple(f"{name}_r" for name in source_basenames)
    existing = glob.glob(os.path.join(output_png_dir, "*.png"))
    foreign = [p for p in existing
               if not os.path.basename(p).startswith(expected_prefixes)]
    if foreign:
        print(f"[警告] {output_png_dir} には別データセットのタイルが {len(foreign)} 件あります")
        print(f"       例: {os.path.basename(foreign[0])}")
        print("       このまま実行すると、検出スクリプトが複数エリアの画像を")
        print("       1つのEPSGで混在処理し、緯度経度が誤って計算される危険があります。")
        print("       事前に古いタイルを別フォルダへ退避することを強く推奨します。")
        resp = input("       続行しますか？ [y/N] > ").strip().lower()
        if resp != 'y':
            print("中断しました。")
            sys.exit(1)


# =====================================================
# 1ファイル分のタイル分割処理
# =====================================================
def tile_one_geotiff(tif_path):
    """1枚のGeoTIFFを読み込み、TILE_WIDTH x TILE_HEIGHT のタイルに分割して
    OUTPUT_PNG_DIR / OUTPUT_TFW_DIR に出力する。出力タイル数を返す。"""
    print("-"*80)
    print(f"タイル分割: {tif_path}")

    src = Image.open(tif_path)
    width, height = src.size
    print(f"  元画像サイズ: {width} x {height} px")

    a, b, tx, d, e, ty = read_model_transformation(src, tif_path)
    half_pixel = is_area_convention(src)
    src_basename = os.path.splitext(os.path.basename(tif_path))[0]

    col_offsets, col_overlap_px, col_overlap_ratio = compute_tile_offsets(width, TILE_WIDTH)
    row_offsets, row_overlap_px, row_overlap_ratio = compute_tile_offsets(height, TILE_HEIGHT)
    total = len(col_offsets) * len(row_offsets)
    print(f"  タイル分割: {len(col_offsets)}列 x {len(row_offsets)}行 = {total}タイル "
          f"({TILE_WIDTH}x{TILE_HEIGHT}px)")
    print(f"  オーバーラップ率（自動調整）: 列方向 {col_overlap_ratio*100:.2f}%"
          f"（{col_overlap_px:.1f}px） / 行方向 {row_overlap_ratio*100:.2f}%"
          f"（{row_overlap_px:.1f}px）")

    count = 0
    for row_idx, y0 in enumerate(row_offsets):
        for col_idx, x0 in enumerate(col_offsets):
            box = (x0, y0, x0 + TILE_WIDTH, y0 + TILE_HEIGHT)
            tile_img = src.crop(box)
            tile_name = f"{src_basename}_r{row_idx}_c{col_idx}"
            tile_img.save(os.path.join(OUTPUT_PNG_DIR, f"{tile_name}.png"))
            write_tfw(
                os.path.join(OUTPUT_TFW_DIR, f"{tile_name}.tfw"),
                a, b, tx, d, e, ty, x0, y0, half_pixel_correct=half_pixel,
            )
            count += 1
            if count % 100 == 0:
                print(f"    進捗: {count}/{total}")

    print(f"  完了: {count} タイルを出力しました")
    return count


# =====================================================
# メイン処理
# =====================================================
def main():
    # 数千万〜数億pxのGeoTIFFはPIL既定の約89Mpx制限（DecompressionBombError）を
    # 超えることがあるため、読み込み前に制限を解除する
    Image.MAX_IMAGE_PIXELS = None

    print("="*80)
    print(f"GeoTIFFタイル分割: {SOURCE_TIF_DIR} 内の全tifファイルを処理します")
    print("="*80)

    tif_paths = sorted(glob.glob(os.path.join(SOURCE_TIF_DIR, "*.tif")))
    if not tif_paths:
        print(f"[エラー] {SOURCE_TIF_DIR} にtifファイルが見つかりません。")
        print(f"         タイル分割したいGeoTIFFを {SOURCE_TIF_DIR}/ に配置してください。")
        sys.exit(1)

    print(f"対象ファイル: {len(tif_paths)}件")
    for p in tif_paths:
        print(f"  - {p}")

    os.makedirs(OUTPUT_PNG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TFW_DIR, exist_ok=True)

    source_basenames = [os.path.splitext(os.path.basename(p))[0] for p in tif_paths]
    check_existing_tiles_for_mixing(OUTPUT_PNG_DIR, source_basenames)

    total_count = 0
    for tif_path in tif_paths:
        total_count += tile_one_geotiff(tif_path)

    print("="*80)
    print(f"全て完了: 合計 {total_count} タイルを {OUTPUT_PNG_DIR} / {OUTPUT_TFW_DIR} に出力しました")
    print("="*80)


if __name__ == "__main__":
    main()
