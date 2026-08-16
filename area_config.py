# -*- coding: utf-8 -*-
"""
対象地域（都道府県）に応じた平面直角座標系（JGD2011 Japan Plane Rectangular CS）の
EPSGコードを対話的に決定するためのモジュール。

系番号（I〜XIX） <-> EPSG:6669〜6687 の対応、および都道府県ごとにどの系を
使用するかは国土地理院の公式区分に基づく。北海道・鹿児島県・東京都・沖縄県は
複数の系にまたがるため、該当する場合はサブ選択を挟む。

どの経路でも解決できない場合に備え、EPSGコード・系番号の直接指定も常に受け付ける。
"""

import glob
import os


# =====================================================
# 系番号 <-> EPSGコード
# =====================================================
ZONE_TO_EPSG = {zone: 6668 + zone for zone in range(1, 20)}  # I(1)->6669 ... XIX(19)->6687
EPSG_TO_ZONE = {epsg: zone for zone, epsg in ZONE_TO_EPSG.items()}

ROMAN_NUMERALS = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX",
]
ROMAN_TO_ZONE = {roman: i + 1 for i, roman in enumerate(ROMAN_NUMERALS)}


# =====================================================
# 都道府県 -> 系番号（国土地理院の公式区分）
# 単一の系のみに属する場合は int。
# 複数の系にまたがる場合は [(選択肢ラベル, 系番号), ...] のリスト。
# =====================================================
PREFECTURE_ZONE_MAP = {
    # 系I・II（九州）
    "長崎県": 1,
    "鹿児島県": [("本土（種子島・屋久島を含む）", 2),
                 ("奄美群島など離島（北緯27°-32°、東経128°18′-130°）", 1)],
    "福岡県": 2, "佐賀県": 2, "熊本県": 2, "大分県": 2, "宮崎県": 2,
    # 系III（山陰・広島）
    "山口県": 3, "島根県": 3, "広島県": 3,
    # 系IV（四国）
    "香川県": 4, "愛媛県": 4, "徳島県": 4, "高知県": 4,
    # 系V（兵庫・鳥取・岡山）
    "兵庫県": 5, "鳥取県": 5, "岡山県": 5,
    # 系VI（近畿+福井）
    "京都府": 6, "大阪府": 6, "福井県": 6, "滋賀県": 6,
    "三重県": 6, "奈良県": 6, "和歌山県": 6,
    # 系VII（北陸東部・東海）
    "石川県": 7, "富山県": 7, "岐阜県": 7, "愛知県": 7,
    # 系VIII（甲信越・静岡）
    "新潟県": 8, "長野県": 8, "山梨県": 8, "静岡県": 8,
    # 系IX（関東+福島、東京本土）
    "東京都": [("本土", 9),
               ("小笠原諸島（北緯28°以南、東経140°30′-143°）", 14),
               ("沖ノ鳥島（北緯28°以南、東経140°30′以西）", 18),
               ("南鳥島（北緯28°以南、東経143°以東）", 19)],
    "福島県": 9, "栃木県": 9, "茨城県": 9, "埼玉県": 9,
    "千葉県": 9, "群馬県": 9, "神奈川県": 9,
    # 系X（東北北部）
    "青森県": 10, "秋田県": 10, "山形県": 10, "岩手県": 10, "宮城県": 10,
    # 系XI・XII・XIII（北海道）
    "北海道": [("道南西部（小樽市・函館市・伊達市など）", 11),
               ("道央（札幌市・旭川市・稚内市など）", 12),
               ("道東（北見市・帯広市・釧路市・網走市・根室市など）", 13)],
    # 系XV・XVI・XVII（沖縄）
    "沖縄県": [("沖縄本島など（東経126°-130°）", 15),
               ("宮古・八重山諸島（東経126°以西）", 16),
               ("大東諸島（東経130°以東）", 17)],
}


_PREF_SUFFIXES = ("都", "道", "府", "県")


def _try_parse_epsg(s):
    """'6677' / 'EPSG:6677' のような文字列をEPSGコードとして解釈する。"""
    s = s.strip().upper().replace("EPSG:", "").replace("EPSG", "").strip()
    if s.isdigit() and int(s) in EPSG_TO_ZONE:
        return int(s)
    return None


def _try_parse_zone(s):
    """'IX' / '9' / '系9' / '9系' のような文字列を系番号(1-19)として解釈する。"""
    s = s.strip().upper().replace("ZONE", "").replace("系", "").strip()
    if s.isdigit() and 1 <= int(s) <= 19:
        return int(s)
    return ROMAN_TO_ZONE.get(s)


def _canonicalize_pref(raw):
    """'東京' -> '東京都' のように都道府県の接尾辞を補ってテーブルに一致させる。"""
    raw = raw.strip()
    if raw in PREFECTURE_ZONE_MAP:
        return raw
    for suf in _PREF_SUFFIXES:
        if raw + suf in PREFECTURE_ZONE_MAP:
            return raw + suf
    return None


def _prompt_subregion(pref_name, options, input_fn):
    print(f"  {pref_name} は複数の平面直角座標系にまたがります。該当する地域を選択してください:")
    for i, (label, zone) in enumerate(options, 1):
        print(f"    {i}. {label} (系{ROMAN_NUMERALS[zone - 1]})")
    while True:
        raw = input_fn(f"  番号を入力 [1-{len(options)}] > ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        print("  [エラー] 有効な番号を入力してください。")


def resolve_epsg_code(raw_input, input_fn=input):
    """
    ユーザー入力からEPSGコードを解決する。
    優先順位: 1) EPSGコード直接指定 2) 系番号 3) 都道府県名（複数系ならサブ選択）
    解決できない場合は None を返す。
    """
    s = raw_input.strip()

    epsg = _try_parse_epsg(s)
    if epsg is not None:
        return epsg

    zone = _try_parse_zone(s)
    if zone is not None:
        return ZONE_TO_EPSG[zone]

    pref = _canonicalize_pref(s)
    if pref is not None:
        entry = PREFECTURE_ZONE_MAP[pref]
        zone = entry if isinstance(entry, int) else _prompt_subregion(pref, entry, input_fn)
        return ZONE_TO_EPSG[zone]

    return None


def prompt_epsg_code(input_fn=input):
    """対話的に対象地域を入力してもらい、EPSGコードを確定させる。"""
    print("\n" + "=" * 80)
    print("対象地域の平面直角座標系（EPSGコード）を設定します。")
    print("入力例: 都道府県名（東京都 / 兵庫県）、系番号（IX / 9 / 系9）、EPSGコード直接指定（6677 / EPSG:6677）")
    print("=" * 80)
    while True:
        raw = input_fn("対象地域 > ")
        if not raw.strip():
            continue
        epsg = resolve_epsg_code(raw, input_fn)
        if epsg is None:
            print(f"  [エラー] '{raw}' を認識できませんでした。再入力してください。")
            continue
        zone = EPSG_TO_ZONE[epsg]
        print(f"  → EPSG:{epsg}（系{ROMAN_NUMERALS[zone - 1]}）を使用します。")
        confirm = input_fn("  よろしいですか？ [Y/n] > ").strip().lower()
        if confirm in ("", "y", "yes"):
            return epsg


def prompt_building_csv_path(building_list_dir="input/building_list", input_fn=input):
    """
    input/building_list/ 配下のCSVを列挙し、対象の建物リスト
    （ZENRIN建物ポイントデータ）を選択させる。1件のみの場合は自動選択する。
    """
    csv_files = sorted(glob.glob(os.path.join(building_list_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"{building_list_dir} に建物リスト（ZENRIN建物ポイントデータ）のCSVが見つかりません。"
        )

    if len(csv_files) == 1:
        print(f"建物リスト（ZENRIN建物ポイントデータ）を自動選択しました: {csv_files[0]}")
        return csv_files[0]

    print("\n" + "=" * 80)
    print("複数の建物リスト（ZENRIN建物ポイントデータ）CSVが見つかりました。使用するファイルを選択してください:")
    print("=" * 80)
    for i, path in enumerate(csv_files, 1):
        print(f"  {i}. {os.path.basename(path)}")
    while True:
        raw = input_fn(f"番号を入力 [1-{len(csv_files)}] > ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(csv_files):
            return csv_files[int(raw) - 1]
        print("  [エラー] 有効な番号を入力してください。")
