"""
detection_results_imagesフォルダの衛星画像をimage_cutフォルダのtfwの座標データをもとに1つに統合するスクリプト
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

def read_tfw(tfw_path):
    """
    tfwファイルを読み込んで座標情報を取得
    
    tfwファイルの形式:
    - Line 1: x方向のピクセルサイズ（解像度）
    - Line 2: y軸の回転
    - Line 3: x軸の回転
    - Line 4: y方向のピクセルサイズ（解像度、通常負の値）
    - Line 5: 左上ピクセル中心のx座標
    - Line 6: 左上ピクセル中心のy座標
    """
    with open(tfw_path, 'r') as f:
        lines = f.readlines()
    
    pixel_size_x = float(lines[0].strip())  # x方向の解像度
    rotation_y = float(lines[1].strip())    # 回転（通常0）
    rotation_x = float(lines[2].strip())    # 回転（通常0）
    pixel_size_y = float(lines[3].strip())  # y方向の解像度（負の値）
    x_coord = float(lines[4].strip())       # 左上のx座標
    y_coord = float(lines[5].strip())       # 左上のy座標
    
    return {
        'pixel_size_x': pixel_size_x,
        'pixel_size_y': pixel_size_y,
        'x_coord': x_coord,
        'y_coord': y_coord,
        'rotation_x': rotation_x,
        'rotation_y': rotation_y
    }

def get_image_bounds(img_path, tfw_data):
    """画像の境界座標を取得"""
    img = Image.open(img_path)
    width, height = img.size
    
    # 左上座標
    x_min = tfw_data['x_coord']
    y_max = tfw_data['y_coord']
    
    # 右下座標（ピクセルサイズを考慮）
    x_max = x_min + width * tfw_data['pixel_size_x']
    y_min = y_max + height * tfw_data['pixel_size_y']  # pixel_size_yは負の値
    
    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'width': width,
        'height': height
    }

def merge_images_with_tfw(detection_images_dir, image_cut_dir, output_path):
    """
    detection_results_imagesフォルダの画像をtfwファイルの座標情報を使って統合
    """
    detection_images_dir = Path(detection_images_dir)
    image_cut_dir = Path(image_cut_dir)
    
    # 検出画像とtfwファイルのペアを作成
    image_data = []
    
    print("画像とtfwファイルの情報を収集中...")
    for detected_img in tqdm(sorted(detection_images_dir.glob("*.png"))):
        # 検出画像のファイル名から元の画像名を取得（_detected.pngを削除）
        base_name = detected_img.stem.replace("_detected", "")
        
        # 対応するtfwファイルを探す
        tfw_path = image_cut_dir / f"{base_name}.tfw"
        original_img_path = image_cut_dir / f"{base_name}.png"
        
        if tfw_path.exists() and original_img_path.exists():
            # tfwファイルを読み込み
            tfw_data = read_tfw(tfw_path)
            
            # 画像の境界を取得
            bounds = get_image_bounds(detected_img, tfw_data)
            
            image_data.append({
                'path': detected_img,
                'tfw_data': tfw_data,
                'bounds': bounds
            })
        else:
            print(f"警告: {detected_img.name} に対応するtfwファイルまたは元画像が見つかりません")
    
    if not image_data:
        print("エラー: 処理する画像が見つかりませんでした")
        return
    
    print(f"処理対象の画像数: {len(image_data)}")
    
    # 全体の境界を計算
    print("全体の境界を計算中...")
    global_x_min = min(img['bounds']['x_min'] for img in image_data)
    global_x_max = max(img['bounds']['x_max'] for img in image_data)
    global_y_min = min(img['bounds']['y_min'] for img in image_data)
    global_y_max = max(img['bounds']['y_max'] for img in image_data)
    
    print(f"全体の境界: x=[{global_x_min}, {global_x_max}], y=[{global_y_min}, {global_y_max}]")
    
    # 解像度（最初の画像から取得）
    pixel_size_x = image_data[0]['tfw_data']['pixel_size_x']
    pixel_size_y = abs(image_data[0]['tfw_data']['pixel_size_y'])
    
    # 統合画像のサイズを計算
    merged_width = int((global_x_max - global_x_min) / pixel_size_x)
    merged_height = int((global_y_max - global_y_min) / pixel_size_y)
    
    print(f"統合画像のサイズ: {merged_width} x {merged_height} ピクセル")
    
    # メモリ使用量を確認
    estimated_memory_mb = (merged_width * merged_height * 4) / (1024 * 1024)
    print(f"推定メモリ使用量: {estimated_memory_mb:.2f} MB")
    
    if estimated_memory_mb > 10000:  # 10GB以上
        print("警告: 画像サイズが非常に大きいです。処理に時間がかかる可能性があります。")
    
    # 統合画像を作成（RGBA形式）
    print("統合画像を作成中...")
    merged_image = Image.new('RGBA', (merged_width, merged_height), (255, 255, 255, 0))
    
    # 各画像を適切な位置に配置
    print("各画像を配置中...")
    for img_info in tqdm(image_data):
        # 画像を読み込み
        img = Image.open(img_info['path'])
        
        # RGBAモードに変換
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # 統合画像内での位置を計算
        x_offset = int((img_info['bounds']['x_min'] - global_x_min) / pixel_size_x)
        y_offset = int((global_y_max - img_info['bounds']['y_max']) / pixel_size_y)
        
        # 画像を配置
        merged_image.paste(img, (x_offset, y_offset), img)
    
    # 画像を保存
    print(f"統合画像を保存中: {output_path}")
    
    # RGB形式で保存（ファイルサイズを削減）
    merged_image_rgb = merged_image.convert('RGB')
    merged_image_rgb.save(output_path, quality=95, optimize=True)
    
    print("完了!")
    print(f"保存先: {output_path}")
    
    # 統合画像のtfwファイルも作成
    tfw_output_path = output_path.rsplit('.', 1)[0] + '.tfw'
    with open(tfw_output_path, 'w') as f:
        f.write(f"{pixel_size_x}\n")
        f.write("0.0\n")
        f.write("0.0\n")
        f.write(f"{-pixel_size_y}\n")
        f.write(f"{global_x_min}\n")
        f.write(f"{global_y_max}\n")
    
    print(f"tfwファイルを保存: {tfw_output_path}")

if __name__ == "__main__":
    # ディレクトリの設定
    base_dir = Path(__file__).parent
    detection_images_dir = base_dir / "detection_results_images"
    image_cut_dir = base_dir / "image_cut"
    output_path = base_dir / "merged_detection_result.png"
    
    # 統合処理を実行
    merge_images_with_tfw(detection_images_dir, image_cut_dir, str(output_path))
