# -*- coding: utf-8 -*-
"""
YOLO検出と固定閾値による設備判定プログラム (ver8.0)
- 閾値0.1で全オブジェクト検出
- CT, ACC, MUL, PACの閾値を固定（CT=0.4, ACC=0.4, MUL=0.3, PAC=0.3）
- 入力CSVに plant 列を追加して出力
"""

import os
import glob
import pandas as pd
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from natsort import natsorted
from pyproj import Transformer
from PIL import Image

# =====================================================
# 定数定義
# =====================================================
GSD_METERS_PER_PIXEL = 0.075
EPSG_CODE = 6677
INITIAL_THRESHOLD = 0.1  # 初期検出閾値

# =====================================================
# TFWファイル読み込み
# =====================================================
def read_tfw(tfw_path):
    """TFWファイルを読み込む"""
    with open(tfw_path, 'r') as f:
        lines = f.readlines()
    return tuple(float(lines[i].strip()) for i in range(6))


# =====================================================
# ピクセル座標→緯度経度変換
# =====================================================
def pixel_to_latlng(px, py, A, D, B, E, C, F, epsg_code=EPSG_CODE):
    """ピクセル座標を緯度経度に変換"""
    x_meters = A * px + B * py + C
    y_meters = D * px + E * py + F
    
    try:
        transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
        lng, lat = transformer.transform(x_meters, y_meters)
        return lat, lng
    except:
        return None, None


# =====================================================
# 建物境界線XML読み込み
# =====================================================
def parse_building_boundary_xml(xml_path):
    """建物境界線XMLファイルを解析"""
    import xml.etree.ElementTree as ET
    buildings = {}
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        ns = {
            'gml': 'http://www.opengis.net/gml/3.2',
            'default': 'http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema'
        }
        
        for bld in root.findall('.//default:BldA', ns):
            gml_id = bld.get('{http://www.opengis.net/gml/3.2}id')
            pos_list_elem = bld.find('.//gml:posList', ns)
            
            if pos_list_elem is not None and pos_list_elem.text:
                coords_text = pos_list_elem.text.strip()
                coords = coords_text.split()
                
                coordinates = []
                for i in range(0, len(coords), 2):
                    lat = float(coords[i])
                    lng = float(coords[i + 1])
                    coordinates.append((lat, lng))
                
                buildings[gml_id] = coordinates
    except Exception as e:
        print(f"  [Error] XMLファイルの解析に失敗: {e}")
    
    return buildings


def load_all_building_boundaries(boundary_dir):
    """全XMLファイルを読み込む"""
    all_buildings = {}
    xml_files = glob.glob(os.path.join(boundary_dir, "*.xml"))
    
    for xml_file in xml_files:
        buildings = parse_building_boundary_xml(xml_file)
        all_buildings.update(buildings)
    
    return all_buildings


# =====================================================
# YOLO検出実行（閾値0.1）
# =====================================================
def detect_all_objects(image_folder, model, threshold=INITIAL_THRESHOLD):
    """全画像でYOLO検出を実行"""
    print("="*80)
    print(f"YOLO検出を実行中（閾値={threshold}）...")
    print("="*80)
    
    all_detections = []
    output_dir = "detection_results_images"
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = natsorted(glob.glob(os.path.join(image_folder, "*.png")))
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] 検出中: {os.path.basename(image_path)}")
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        tfw_path = os.path.join(image_folder, f"{base_name}.tfw")
        
        if not os.path.exists(tfw_path):
            continue
        
        A, D, B, E, C, F = read_tfw(tfw_path)
        img = Image.open(image_path)
        
        # YOLO検出（OBB）
        results = model.predict(img, conf=threshold, verbose=False, task='obb')
        
        # 検出結果を保存
        if results and len(results) > 0:
            result_img = results[0].plot()
            output_path = os.path.join(output_dir, f"{base_name}_detected.png")
            Image.fromarray(result_img).save(output_path)
        
        # 検出結果を処理
        if results[0].obb is not None:
            class_names = results[0].names
            
            for box in results[0].obb:
                cx = box.xywhr[0][0].item()
                cy = box.xywhr[0][1].item()
                
                class_id = int(box.cls[0].item())
                class_name = class_names.get(class_id, "Unknown")
                confidence = box.conf[0].item()
                
                lat, lng = pixel_to_latlng(cx, cy, A, D, B, E, C, F)
                
                if lat is not None and lng is not None:
                    all_detections.append({
                        'lat': lat,
                        'lng': lng,
                        'class_name': class_name,
                        'confidence': confidence,
                        'image_file': os.path.basename(image_path)
                    })
    
    print(f"\n合計 {len(all_detections)} 個のオブジェクトを検出")
    print(f"  CT: {sum(1 for d in all_detections if d['class_name'] == 'CT')}")
    print(f"  ACC: {sum(1 for d in all_detections if d['class_name'] == 'ACC')}")
    print(f"  MUL: {sum(1 for d in all_detections if d['class_name'] == 'MUL')}")
    print(f"  PAC: {sum(1 for d in all_detections if d['class_name'] == 'PAC')}\n")
    
    return all_detections


# =====================================================
# 建物が映っている画像を判定
# =====================================================
def find_images_containing_building(building_lat, building_lng, image_folder):
    """
    建物の緯度経度が含まれる画像を検索
    
    Args:
        building_lat: 建物の緯度
        building_lng: 建物の経度
        image_folder: 画像フォルダ
    
    Returns:
        list: 建物が含まれる画像ファイル名のリスト
    """
    matching_images = []
    image_files = glob.glob(os.path.join(image_folder, "*.png"))
    
    for image_path in image_files:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        tfw_path = os.path.join(image_folder, f"{base_name}.tfw")
        
        if not os.path.exists(tfw_path):
            continue
        
        A, D, B, E, C, F = read_tfw(tfw_path)
        
        # 画像のサイズを仮定（1000x750ピクセル）
        img_width, img_height = 1000, 750
        
        # 画像と4隅の座標を計算
        # 左上 (0, 0)
        lat_tl, lng_tl = pixel_to_latlng(0, 0, A, D, B, E, C, F)
        # 右上 (width, 0)
        lat_tr, lng_tr = pixel_to_latlng(img_width, 0, A, D, B, E, C, F)
        # 左下 (0, height)
        lat_bl, lng_bl = pixel_to_latlng(0, img_height, A, D, B, E, C, F)
        # 右下 (width, height)
        lat_br, lng_br = pixel_to_latlng(img_width, img_height, A, D, B, E, C, F)
        
        if any(x is None for x in [lat_tl, lng_tl, lat_tr, lng_tr, lat_bl, lng_bl, lat_br, lng_br]):
            continue
        
        # 緯度・経度の範囲を計算
        lat_min = min(lat_tl, lat_tr, lat_bl, lat_br)
        lat_max = max(lat_tl, lat_tr, lat_bl, lat_br)
        lng_min = min(lng_tl, lng_tr, lng_bl, lng_br)
        lng_max = max(lng_tl, lng_tr, lng_bl, lng_br)
        
        # 建物が範囲内にあるかチェック
        if lat_min <= building_lat <= lat_max and lng_min <= building_lng <= lng_max:
            matching_images.append(base_name)
    
    return matching_images


# =====================================================
# 建物ごとの画像マッピングを事前に作成
# =====================================================
def create_building_image_mapping(building_df, image_folder):
    """
    各建物がどの画像に含まれるかを事前にマッピング
    
    Returns:
        dict: {建物インデックス: [画像名リスト]}
    """
    print("\n" + "="*80)
    print("建物と画像のマッピングを作成中...")
    print("="*80)
    
    building_image_map = {}
    
    for idx, row in building_df.iterrows():
        building_lat = row['緯度']
        building_lng = row['経度']
        
        images = find_images_containing_building(building_lat, building_lng, image_folder)
        building_image_map[idx] = images
        
        if (idx + 1) % 10 == 0:
            print(f"  進捗: {idx + 1}/{len(building_df)} 建物")
    
    buildings_with_images = sum(1 for imgs in building_image_map.values() if len(imgs) > 0)
    print(f"\nマッピング完了:")
    print(f"  画像が見つかった建物数: {buildings_with_images}/{len(building_image_map)}")
    
    return building_image_map


# =====================================================
# 建物ごとに含まれる検出オブジェクトをマッピング（事前処理）
# =====================================================
def map_detections_to_buildings(all_detections, building_df, building_boundaries):
    """
    各建物に含まれる検出オブジェクトを事前にマッピング（高速化のため1回だけ実行）
    
    Returns:
        dict: {建物インデックス: [検出リスト]}
    """
    print("\n" + "="*80)
    print("建物ごとのオブジェクトマッピングを実行中...")
    print("="*80)
    
    building_detections_map = {}
    
    for idx, row in building_df.iterrows():
        building_lat = row['緯度']
        building_lng = row['経度']
        building_point = Point(building_lng, building_lat)
        
        # 建物境界線を検索
        boundary_polygon = None
        for bld_id, coords in building_boundaries.items():
            if len(coords) < 3:
                continue
            try:
                poly = Polygon([(lng, lat) for lat, lng in coords])
                if poly.contains(building_point):
                    boundary_polygon = poly
                    break
            except:
                continue
        
        if boundary_polygon is None:
            building_detections_map[idx] = []
            continue
        
        # この建物境界内の検出を抽出
        building_detections = []
        for detection in all_detections:
            det_point = Point(detection['lng'], detection['lat'])
            if boundary_polygon.contains(det_point):
                building_detections.append(detection)
        
        building_detections_map[idx] = building_detections
        
        if (idx + 1) % 10 == 0:
            print(f"  進捗: {idx + 1}/{len(building_df)} 建物")
    
    # 統計情報を表示
    total_mapped = sum(len(dets) for dets in building_detections_map.values())
    buildings_with_detections = sum(1 for dets in building_detections_map.values() if len(dets) > 0)
    
    print(f"\nマッピング完了:")
    print(f"  総建物数: {len(building_detections_map)}")
    print(f"  検出ありの建物数: {buildings_with_detections}")
    print(f"  マッピングされた検出総数: {total_mapped}")
    
    return building_detections_map


# =====================================================
# 固定閾値で plant を判定
# =====================================================
def assign_plant_with_fixed_thresholds(building_df, building_detections_map,
                                       th_ct=0.4, th_acc=0.4, th_mul=0.3, th_pac=0.3):
    """
    固定閾値で建物ごとに plant を1つ割り当てる
    優先順位: CT > ACC > MUL > PAC

    Returns:
        pd.DataFrame: 入力CSVに plant 列を追加したデータフレーム
    """
    output_df = building_df.copy()
    plant_values = []

    for idx, _ in building_df.iterrows():
        building_detections = building_detections_map.get(idx, [])

        has_ct = any(d['class_name'] == 'CT' and d['confidence'] >= th_ct for d in building_detections)
        has_acc = any(d['class_name'] == 'ACC' and d['confidence'] >= th_acc for d in building_detections)
        has_mul = any(d['class_name'] == 'MUL' and d['confidence'] >= th_mul for d in building_detections)
        has_pac = any(d['class_name'] == 'PAC' and d['confidence'] >= th_pac for d in building_detections)

        if has_ct:
            plant = 'CT'
        elif has_acc:
            plant = 'ACC'
        elif has_mul:
            plant = 'MUL'
        elif has_pac:
            plant = 'PAC'
        else:
            plant = ''

        plant_values.append(plant)

    output_df['plant'] = plant_values
    return output_df


# =====================================================
# メイン処理
# =====================================================
def main():
    print("="*80)
    print("YOLO検出と固定閾値による設備判定プログラム (ver8.0)")
    print("="*80)
    
    # パス設定
    image_folder = "image_cut"
    model_path = "models/best.pt"
    building_csv_path = "BPO13102_地域冷暖房.csv"
    boundary_folder = "bld_boundary"
    
    # YOLOモデル読み込み
    print(f"\nYOLOモデルを読み込んでいます: {model_path}")
    model = YOLO(model_path)
    print("モデル読み込み完了")
    
    # 建物データ読み込み
    building_df = pd.read_csv(building_csv_path, encoding='utf-8')
    print(f"\n建物CSVデータを読み込みました: {len(building_df)}件")
    
    # 建物境界線データ読み込み
    print(f"\n建物境界線データを読み込んでいます...")
    building_boundaries = load_all_building_boundaries(boundary_folder)
    print(f"建物境界線データを読み込みました: {len(building_boundaries)}件")
    
    # ステップ1: 閾値0.1で全オブジェクト検出
    all_detections = detect_all_objects(image_folder, model, threshold=INITIAL_THRESHOLD)
    
    # 検出結果を保存
    detections_df = pd.DataFrame(all_detections)
    detections_df.to_csv("all_detections_th0.1.csv", index=False, encoding='utf-8-sig')
    print(f"全検出結果を保存: all_detections_th0.1.csv")
    
    # ステップ2: 建物ごとのオブジェクトマッピング（高速化のため事前処理）
    building_detections_map = map_detections_to_buildings(
        all_detections, building_df, building_boundaries
    )

    # ステップ3: 固定閾値で plant を判定
    print("\n固定閾値で plant を判定中...")
    print("  CT: 0.4 / ACC: 0.4 / MUL: 0.3 / PAC: 0.3")
    output_df = assign_plant_with_fixed_thresholds(
        building_df,
        building_detections_map,
        th_ct=0.4,
        th_acc=0.4,
        th_mul=0.3,
        th_pac=0.3
    )

    # ステップ4: 入力CSVに plant 列を追加したファイルを保存
    output_csv_path = os.path.splitext(building_csv_path)[0] + "_with_plant.csv"
    output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n出力CSVを保存: {output_csv_path}")
    
    print("\n" + "="*80)
    print("処理完了")
    print("="*80)


if __name__ == "__main__":
    main()
