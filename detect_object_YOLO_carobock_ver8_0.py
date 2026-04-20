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
import itertools

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
# 閾値を適用して建物ごとに設備有無を判定（マッピング済みデータ使用）
# =====================================================
def apply_thresholds_and_aggregate(building_df, building_detections_map, building_image_map,
                                   th_ct, th_acc, th_decentralized):
    """
    閾値を適用して建物ごとに設備有無を判定（事前マッピング済みデータを使用）
    
    Args:
        building_df: 建物データフレーム
        building_detections_map: 建物ごとの検出マッピング
        building_image_map: 建物ごとの画像マッピング
        th_ct: CTの閾値
        th_acc: ACCの閾値
        th_decentralized: decentralizedの閾値
    
    Returns:
        建物ごとの検出結果リスト
    """
    results = []
    
    for idx, row in building_df.iterrows():
        # マッピング済みの検出を取得
        building_detections = building_detections_map.get(idx, [])
        
        # 建物が含まれる画像名を取得
        image_names = building_image_map.get(idx, [])
        image_names_str = ', '.join(image_names) if image_names else ''
        
        # 閾値を適用して設備有無を判定
        has_ct = any(d['class_name'] == 'CT' and d['confidence'] >= th_ct 
                     for d in building_detections)
        has_acc = any(d['class_name'] == 'ACC' and d['confidence'] >= th_acc 
                      for d in building_detections)
        has_decentralized = any(d['class_name'] == 'decentralized' and d['confidence'] >= th_decentralized 
                                for d in building_detections)
        
        results.append({
            '建物名': row.get('建物名', ''),
            'タイル画像': image_names_str,
            '検出_CT': 1 if has_ct else 0,
            '検出_ACC': 1 if has_acc else 0,
            '検出_decentralized': 1 if has_decentralized else 0,
            '観測_CT': row['observable_CT'],
            '観測_ACC': row['observable_ACC'],
            '観測_decentralized': row['observable_decentralized']
        })
    
    return results


# =====================================================
# 一致率とTP/FP/FN/TN計算
# =====================================================
def calculate_metrics(results):
    """
    一致率とTP/FP/FN/TNを計算
    
    Returns:
        dict: 各種メトリクス
    """
    total_buildings = len(results)
    
    # 建物ごとに3つすべて一致した数
    perfect_match_count = sum(1 for r in results if
                              r['検出_CT'] == r['観測_CT'] and
                              r['検出_ACC'] == r['観測_ACC'] and
                              r['検出_decentralized'] == r['観測_decentralized'])
    
    accuracy = perfect_match_count / total_buildings if total_buildings > 0 else 0
    
    # 設備種別ごとのTP/FP/FN/TN
    metrics = {'一致率': accuracy, '一致建物数': perfect_match_count, '総建物数': total_buildings}
    
    for equipment in ['CT', 'ACC', 'decentralized']:
        tp = sum(1 for r in results if r[f'検出_{equipment}'] == 1 and r[f'観測_{equipment}'] == 1)
        fp = sum(1 for r in results if r[f'検出_{equipment}'] == 1 and r[f'観測_{equipment}'] == 0)
        fn = sum(1 for r in results if r[f'検出_{equipment}'] == 0 and r[f'観測_{equipment}'] == 1)
        tn = sum(1 for r in results if r[f'検出_{equipment}'] == 0 and r[f'観測_{equipment}'] == 0)
        
        metrics[f'{equipment}_TP'] = tp
        metrics[f'{equipment}_FP'] = fp
        metrics[f'{equipment}_FN'] = fn
        metrics[f'{equipment}_TN'] = tn
    
    return metrics


# =====================================================
# 全閾値組み合わせを評価
# =====================================================
def evaluate_all_threshold_combinations(building_df, building_detections_map, building_image_map):
    """
    全閾値組み合わせ（6561通り）を評価（事前マッピング済みデータを使用）
    """
    import time
    
    print("\n" + "="*80)
    print("全閾値組み合わせの評価を開始（6561通り）...")
    print("="*80)
    
    # 閾値の組み合わせ（0.1〜0.9、0.1刻み）
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 4つのクラス（CT, ACC, MUL, PAC）の閾値組み合わせ
    all_combinations = list(itertools.product(thresholds, repeat=4))
    
    print(f"評価する組み合わせ数: {len(all_combinations)}")
    print("\n進捗状況:")
    
    evaluation_results = []
    start_time = time.time()
    
    for i, (th_ct, th_acc, th_mul, th_pac) in enumerate(all_combinations, 1):
        # MULとPACの閾値の平均をdecentralizedの閾値として使用
        th_decentralized = (th_mul + th_pac) / 2
        
        # 建物ごとに判定（マッピング済みデータを使用）
        results = apply_thresholds_and_aggregate(
            building_df, building_detections_map, building_image_map,
            th_ct, th_acc, th_decentralized
        )
        
        # メトリクス計算
        metrics = calculate_metrics(results)
        
        evaluation_results.append({
            'CT閾値': th_ct,
            'ACC閾値': th_acc,
            'MUL閾値': th_mul,
            'PAC閾値': th_pac,
            'decentralized閾値': th_decentralized,
            '一致率': metrics['一致率'],
            '一致建物数': metrics['一致建物数'],
            '総建物数': metrics['総建物数'],
            'CT_TP': metrics['CT_TP'],
            'CT_FP': metrics['CT_FP'],
            'CT_FN': metrics['CT_FN'],
            'CT_TN': metrics['CT_TN'],
            'ACC_TP': metrics['ACC_TP'],
            'ACC_FP': metrics['ACC_FP'],
            'ACC_FN': metrics['ACC_FN'],
            'ACC_TN': metrics['ACC_TN'],
            'decentralized_TP': metrics['decentralized_TP'],
            'decentralized_FP': metrics['decentralized_FP'],
            'decentralized_FN': metrics['decentralized_FN'],
            'decentralized_TN': metrics['decentralized_TN'],
            '詳細結果': results
        })
        
        # 100件ごとまたは1%ごとに進捗と推定残り時間を表示
        if i % 100 == 0 or i % max(1, len(all_combinations) // 100) == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_combo = elapsed_time / i
            remaining_combos = len(all_combinations) - i
            estimated_remaining_time = avg_time_per_combo * remaining_combos
            
            # 時間を分秒に変換
            remaining_minutes = int(estimated_remaining_time // 60)
            remaining_seconds = int(estimated_remaining_time % 60)
            
            print(f"  [{i}/{len(all_combinations)}] {i/len(all_combinations)*100:.1f}% 完了 | "
                  f"残り時間: 約{remaining_minutes}分{remaining_seconds}秒 | "
                  f"現在: CT={th_ct}, ACC={th_acc}, MUL={th_mul}, PAC={th_pac}")
    
    total_time = time.time() - start_time
    total_minutes = int(total_time // 60)
    total_seconds = int(total_time % 60)
    
    print(f"\n評価完了: {len(evaluation_results)}通りの組み合わせ")
    print(f"処理時間: {total_minutes}分{total_seconds}秒")
    
    return evaluation_results


# =====================================================
# 上位結果の抽出とCSV出力
# =====================================================
def extract_top_results(evaluation_results, top_summary_n=500, top_detail_n=5):
    """
    0.1と0.2を含まない組み合わせで上位を抽出
    - サマリー用: 一致率上位500件
    - 詳細用: 一致率 → CT_F1 → ACC_F1 → decentralized_F1 の優先順位で上位5件
    """
    print("\n" + "="*80)
    print("上位結果の抽出...")
    print("="*80)
    
    # 0.1と0.2を含まない組み合わせでフィルタ
    filtered = [r for r in evaluation_results 
                if 0.1 not in [r['CT閾値'], r['ACC閾値'], r['MUL閾値'], r['PAC閾値']]
                and 0.2 not in [r['CT閾値'], r['ACC閾値'], r['MUL閾値'], r['PAC閾値']]]
    
    print(f"フィルタ後の組み合わせ数: {len(filtered)}")
    
    # 各結果にF1値を事前計算
    for result in filtered:
        for equipment in ['CT', 'ACC', 'decentralized']:
            tp = result[f'{equipment}_TP']
            fp = result[f'{equipment}_FP']
            fn = result[f'{equipment}_FN']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            result[f'{equipment}_F1_calculated'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 一致率 → CT_F1 → ACC_F1 → decentralized_F1 の優先順位でソート
    sorted_results = sorted(filtered, 
                           key=lambda x: (x['一致率'], 
                                         x['CT_F1_calculated'], 
                                         x['ACC_F1_calculated'], 
                                         x['decentralized_F1_calculated']), 
                           reverse=True)
    
    # サマリー用: 上位500件を取得
    top_summary_results = sorted_results[:top_summary_n]
    print(f"\nサマリー用上位{len(top_summary_results)}件を抽出")
    
    # 詳細用: 上位5件を取得
    top_detail_results = sorted_results[:top_detail_n]
    
    print(f"詳細用: 一致率 → CT_F1 → ACC_F1 → decentralized_F1 の優先順位で上位{len(top_detail_results)}件を抽出")
    print(f"\n詳細表示する{len(top_detail_results)}件の閾値組み合わせ:")
    for i, result in enumerate(top_detail_results, 1):
        print(f"  [{i}] CT={result['CT閾値']}, ACC={result['ACC閾値']}, "
              f"MUL={result['MUL閾値']}, PAC={result['PAC閾値']} "
              f"→ 一致率={result['一致率']:.4f}, CT_F1={result['CT_F1_calculated']:.4f}, "
              f"ACC_F1={result['ACC_F1_calculated']:.4f}, dec_F1={result['decentralized_F1_calculated']:.4f}")
    
    return top_summary_results, top_detail_results


def save_top_results_to_csv(top_summary_results, top_detail_results, output_dir="threshold_evaluation_results"):
    """上位結果をCSVに保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # サマリーCSV（上位500件）
    summary_data = []
    for i, result in enumerate(top_summary_results, 1):
        # 各クラスのprecision, recall, F1-scoreを計算
        metrics_per_class = {}
        for equipment in ['CT', 'ACC', 'decentralized']:
            tp = result[f'{equipment}_TP']
            fp = result[f'{equipment}_FP']
            fn = result[f'{equipment}_FN']
            tn = result[f'{equipment}_TN']
            
            # Positive class (オブジェクトあり)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Negative class (オブジェクトなし)
            neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
            neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
            neg_f1_score = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0
            
            metrics_per_class[f'{equipment}_Precision'] = precision
            metrics_per_class[f'{equipment}_Recall'] = recall
            metrics_per_class[f'{equipment}_F1'] = f1_score
            metrics_per_class[f'{equipment}_Negative_Precision'] = neg_precision
            metrics_per_class[f'{equipment}_Negative_Recall'] = neg_recall
            metrics_per_class[f'{equipment}_Negative_F1'] = neg_f1_score
        
        summary_data.append({
            'ランク': i,
            'CT閾値': result['CT閾値'],
            'ACC閾値': result['ACC閾値'],
            'MUL閾値': result['MUL閾値'],
            'PAC閾値': result['PAC閾値'],
            'decentralized閾値': result['decentralized閾値'],
            '一致率': result['一致率'],
            '一致建物数': result['一致建物数'],
            '総建物数': result['総建物数'],
            'CT_TP': result['CT_TP'],
            'CT_FP': result['CT_FP'],
            'CT_FN': result['CT_FN'],
            'CT_TN': result['CT_TN'],
            'CT_Precision': metrics_per_class['CT_Precision'],
            'CT_Recall': metrics_per_class['CT_Recall'],
            'CT_F1': metrics_per_class['CT_F1'],
            'CT_Negative_Precision': metrics_per_class['CT_Negative_Precision'],
            'CT_Negative_Recall': metrics_per_class['CT_Negative_Recall'],
            'CT_Negative_F1': metrics_per_class['CT_Negative_F1'],
            'ACC_TP': result['ACC_TP'],
            'ACC_FP': result['ACC_FP'],
            'ACC_FN': result['ACC_FN'],
            'ACC_TN': result['ACC_TN'],
            'ACC_Precision': metrics_per_class['ACC_Precision'],
            'ACC_Recall': metrics_per_class['ACC_Recall'],
            'ACC_F1': metrics_per_class['ACC_F1'],
            'ACC_Negative_Precision': metrics_per_class['ACC_Negative_Precision'],
            'ACC_Negative_Recall': metrics_per_class['ACC_Negative_Recall'],
            'ACC_Negative_F1': metrics_per_class['ACC_Negative_F1'],
            'decentralized_TP': result['decentralized_TP'],
            'decentralized_FP': result['decentralized_FP'],
            'decentralized_FN': result['decentralized_FN'],
            'decentralized_TN': result['decentralized_TN'],
            'decentralized_Precision': metrics_per_class['decentralized_Precision'],
            'decentralized_Recall': metrics_per_class['decentralized_Recall'],
            'decentralized_F1': metrics_per_class['decentralized_F1'],
            'decentralized_Negative_Precision': metrics_per_class['decentralized_Negative_Precision'],
            'decentralized_Negative_Recall': metrics_per_class['decentralized_Negative_Recall'],
            'decentralized_Negative_F1': metrics_per_class['decentralized_Negative_F1']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "top500_threshold_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\nサマリーCSV（上位{len(top_summary_results)}件）を保存: {summary_path}")
    
    # 各ランクの詳細結果CSV
    for i, result in enumerate(top_detail_results, 1):
        detail_df = pd.DataFrame(result['詳細結果'])
        
        # 各クラスの一致状況と全体の一致状況を追加
        detail_df['CT一致'] = detail_df.apply(lambda row: '○' if row['検出_CT'] == row['観測_CT'] else '×', axis=1)
        detail_df['ACC一致'] = detail_df.apply(lambda row: '○' if row['検出_ACC'] == row['観測_ACC'] else '×', axis=1)
        detail_df['decentralized一致'] = detail_df.apply(lambda row: '○' if row['検出_decentralized'] == row['観測_decentralized'] else '×', axis=1)
        detail_df['全体一致'] = detail_df.apply(lambda row: '○' if (row['検出_CT'] == row['観測_CT'] and 
                                                                    row['検出_ACC'] == row['観測_ACC'] and 
                                                                    row['検出_decentralized'] == row['観測_decentralized']) else '×', axis=1)
        
        # メトリクス行を追加
        metrics_rows = []
        for equipment in ['CT', 'ACC', 'decentralized']:
            tp = result[f'{equipment}_TP']
            fp = result[f'{equipment}_FP']
            fn = result[f'{equipment}_FN']
            tn = result[f'{equipment}_TN']
            
            # Positive class (オブジェクトあり)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Negative class (オブジェクトなし)
            neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
            neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
            neg_f1_score = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0
            
            metrics_rows.append({
                '建物名': f'【{equipment}メトリクス】',
                'タイル画像': '',
                '検出_CT': '',
                '検出_ACC': '',
                '検出_decentralized': '',
                '観測_CT': f'TP={tp}',
                '観測_ACC': f'FP={fp}',
                '観測_decentralized': f'FN={fn}, TN={tn}',
                'CT一致': '',
                'ACC一致': '',
                'decentralized一致': '',
                '全体一致': ''
            })
            metrics_rows.append({
                '建物名': f'【{equipment}指標(Positive)】',
                'タイル画像': '',
                '検出_CT': '',
                '検出_ACC': '',
                '検出_decentralized': '',
                '観測_CT': f'Precision={precision:.4f}',
                '観測_ACC': f'Recall={recall:.4f}',
                '観測_decentralized': f'F1={f1_score:.4f}',
                'CT一致': '',
                'ACC一致': '',
                'decentralized一致': '',
                '全体一致': ''
            })
            metrics_rows.append({
                '建物名': f'【{equipment}指標(Negative)】',
                'タイル画像': '',
                '検出_CT': '',
                '検出_ACC': '',
                '検出_decentralized': '',
                '観測_CT': f'Precision={neg_precision:.4f}',
                '観測_ACC': f'Recall={neg_recall:.4f}',
                '観測_decentralized': f'F1={neg_f1_score:.4f}',
                'CT一致': '',
                'ACC一致': '',
                'decentralized一致': '',
                '全体一致': ''
            })
        
        # 空行を追加してからメトリクスを追加
        metrics_rows.insert(0, {
            '建物名': '',
            'タイル画像': '',
            '検出_CT': '',
            '検出_ACC': '',
            '検出_decentralized': '',
            '観測_CT': '',
            '観測_ACC': '',
            '観測_decentralized': '',
            'CT一致': '',
            'ACC一致': '',
            'decentralized一致': '',
            '全体一致': ''
        })
        
        metrics_df = pd.DataFrame(metrics_rows)
        combined_df = pd.concat([detail_df, metrics_df], ignore_index=True)
        
        detail_path = os.path.join(output_dir, f"rank{i}_detail_results.csv")
        combined_df.to_csv(detail_path, index=False, encoding='utf-8-sig')
        print(f"  ランク{i}の詳細CSVを保存: {detail_path}")


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
