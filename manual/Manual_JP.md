# KTRtracker ユーザーマニュアル

## 概要

KTRtrackerは、タイムラプス顕微鏡画像から細胞をトラッキングし、細胞質/核（C/N）比を解析するためのPythonパッケージです。核のセグメンテーション、Gap closing機能付きのオブジェクトトラッキング、蛍光強度解析のツールを提供します。

## インストール

### 必要なパッケージ

```bash
pip install numpy pandas matplotlib seaborn tifffile scikit-image scipy cellpose
```

### オプション（インタラクティブな可視化用）

```bash
pip install napari
```

### KTRtrackerのインストール

```bash
pip install -e .
```

## クイックスタート

```python
from KTRtracker import ImageAnalyzer

# フルワークフロー
analyzer = ImageAnalyzer('your_image.tif')
(analyzer
    .load_image()
    .segment_nuclei()
    .track_objects()
    .convert_to_tracked_labels()
    .generate_cytoplasmic_rings())

# 結果の可視化
intensity_df = analyzer.extract_intensity_features()
analyzer.visualize_cn_ratio(intensity_df)
```

---

## ユースケース

### ケース1: バックグラウンド減算済み画像からスタート

セグメンテーションとトラッキングが必要な生の蛍光画像がある場合に使用します。

#### ステップバイステップのワークフロー

```python
from KTRtracker import ImageAnalyzer

# 画像パスでアナライザーを初期化
analyzer = ImageAnalyzer(
    'background_subtracted.tif',
    min_track_length=5,        # 最小トラック長（フレーム数）
    max_linking_distance=20,   # フレーム間リンクの最大距離（ピクセル）
    max_gap_closing=3,         # ギャップを埋める最大フレーム数
    max_gap_distance=25        # Gap closingの最大距離（ピクセル）
)

# ステップ1: 画像の読み込み
analyzer.load_image()
print(f"読み込んだ画像のシェイプ: {analyzer.original_images.shape}")

# ステップ2: Cellposeによる核のセグメンテーション
analyzer.segment_nuclei(
    gpu=True,                    # GPUを使用（利用可能な場合）
    output_dir='segmentation/',  # セグメンテーション結果の出力ディレクトリ
    flow_threshold=0.4,          # Cellposeのflowしきい値
    cellprob_threshold=0.0       # Cellposeの細胞確率しきい値
)

# ステップ3: Gap closing付きでオブジェクトをトラッキング
analyzer.track_objects(
    min_track_length=5,
    max_linking_distance=20,
    max_gap_closing=3,
    max_gap_distance=25
)
print(f"トラック数: {analyzer.tracking_df['track_id'].nunique()}")

# ステップ4: トラッキング結果をラベル画像に変換
analyzer.convert_to_tracked_labels(output_dir='tracked_labels/')

# ステップ5: 細胞質リングを生成
analyzer.generate_cytoplasmic_rings(ring_width=2)

# ステップ6: 蛍光強度特徴量を抽出して可視化
intensity_df = analyzer.extract_intensity_features()
analyzer.visualize_cn_ratio(intensity_df)

# ステップ7: タイムラプス可視化を作成
analyzer.visualize_cn_ratio_timelapse(
    intensity_df,
    save_path='cn_ratio_timelapse.gif'
)
```

#### メソッドチェーン（コンパクト版）

```python
from KTRtracker import ImageAnalyzer

analyzer = ImageAnalyzer('background_subtracted.tif')
(analyzer
    .load_image()
    .segment_nuclei(gpu=True)
    .track_objects()
    .convert_to_tracked_labels()
    .generate_cytoplasmic_rings())

intensity_df = analyzer.extract_intensity_features()
analyzer.visualize_cn_ratio(intensity_df)
```

---

### ケース2: ラベル画像からスタート

セグメンテーション済みのラベル画像がすでにある場合（Cellpose、StarDist、手動セグメンテーションなど）に使用します。

#### ステップバイステップのワークフロー

```python
from KTRtracker import ImageAnalyzer
from KTRtracker.pre_processing import load_tiff_image
import tifffile

# アナライザーを初期化
analyzer = ImageAnalyzer('original_images.tif')

# 蛍光強度解析用にオリジナル画像を読み込み
analyzer.load_image()

# 既存のラベル画像を読み込み
label_images = tifffile.imread('your_labels.tif')
# 必要に応じてリストに変換
if label_images.ndim == 3:
    analyzer.segmentation_labels = [label_images[i] for i in range(label_images.shape[0])]
else:
    analyzer.segmentation_labels = [label_images]

# トラッキング以降を続行
(analyzer
    .track_objects()
    .convert_to_tracked_labels()
    .generate_cytoplasmic_rings())

# 蛍光強度特徴量を抽出して可視化
intensity_df = analyzer.extract_intensity_features()
analyzer.visualize_cn_ratio(intensity_df)
```

#### SimpleLAPTrackerを直接使用

```python
from KTRtracker import SimpleLAPTracker
from KTRtracker.tracking_utils import save_tracking_as_labels
import tifffile

# ラベル画像を読み込み
label_images = tifffile.imread('your_labels.tif')
label_list = [label_images[i] for i in range(label_images.shape[0])]

# トラッカーを初期化
tracker = SimpleLAPTracker(
    max_linking_distance=20,
    max_gap_closing=3,
    max_gap_distance=25,
    min_track_length=5
)

# オブジェクトをトラッキング
tracking_df = tracker.track(label_list)
print(f"初期トラック数: {tracking_df['track_id'].nunique()}")

# Gap closingを適用
tracking_df = tracker.gap_closing(tracking_df)
print(f"Gap closing後: {tracking_df['track_id'].nunique()}")

# 短いトラックをフィルタリング
tracking_df = tracker.filter_tracks(tracking_df)
print(f"フィルタリング後: {tracking_df['track_id'].nunique()}")

# トラッキング済みラベル画像として保存
tracked_labels = save_tracking_as_labels(
    label_list,
    tracking_df,
    output_dir='tracked_labels/'
)
```

---

### ケース3: トラッキング済みラベル画像からスタート

トラッキング済みのラベル画像がすでにある場合（TrackMateや以前の解析結果など）に使用します。

#### ステップバイステップのワークフロー

```python
from KTRtracker import ImageAnalyzer
from KTRtracker.post_processing import (
    generate_cytoplasmic_ring,
    extract_intensity_features,
    visualize_cn_ratio
)
import tifffile

# アナライザーを初期化
analyzer = ImageAnalyzer('original_images.tif')

# オリジナル画像を読み込み
analyzer.load_image()

# トラッキング済みラベル画像を読み込み
tracked_labels = tifffile.imread('tracked_labels.tif')
if tracked_labels.ndim == 3:
    analyzer.tracked_labels = [tracked_labels[i] for i in range(tracked_labels.shape[0])]
else:
    analyzer.tracked_labels = [tracked_labels]

# 細胞質リングを生成して解析
analyzer.generate_cytoplasmic_rings(ring_width=2)

# 蛍光強度特徴量を抽出して可視化
intensity_df = analyzer.extract_intensity_features()
analyzer.visualize_cn_ratio(intensity_df)
```

#### 関数を直接使用

```python
from KTRtracker.post_processing import (
    generate_cytoplasmic_ring,
    extract_intensity_features,
    visualize_cn_ratio,
    visualize_cn_ratio_timelapse
)
import tifffile

# データを読み込み
original_images = tifffile.imread('original_images.tif')
tracked_labels = tifffile.imread('tracked_labels.tif')

# リストに変換
label_list = [tracked_labels[i] for i in range(tracked_labels.shape[0])]

# 細胞質リングを生成
cyto_rings = generate_cytoplasmic_ring(label_list, ring_width=2)

# 蛍光強度特徴量を抽出
intensity_df = extract_intensity_features(label_list, cyto_rings, original_images)

# 可視化
visualize_cn_ratio(intensity_df)
```

---

## パラメータリファレンス

### ImageAnalyzer

| パラメータ | デフォルト | 説明 |
|-----------|---------|-------------|
| `filepath` | - | 入力TIFFファイルのパス |
| `min_track_length` | 3 | 有効なトラックの最小フレーム数 |
| `max_linking_distance` | 15 | フレーム間リンクの最大距離（ピクセル） |
| `max_gap_closing` | 2 | ギャップを埋める最大フレーム数 |
| `max_gap_distance` | 15 | Gap closingの最大距離（ピクセル） |

### SimpleLAPTracker

| パラメータ | デフォルト | 説明 |
|-----------|---------|-------------|
| `max_linking_distance` | 15 | 連続フレーム間のリンク最大距離 |
| `max_gap_closing` | 2 | Gap closingの最大フレーム数 |
| `max_gap_distance` | 15 | Gap closingの最大距離 |
| `min_track_length` | 3 | 保持する最小トラック長 |

### segment_nuclei

| パラメータ | デフォルト | 説明 |
|-----------|---------|-------------|
| `gpu` | True | CellposeでGPUを使用 |
| `output_dir` | 'segmentation_nuc/' | 出力ディレクトリ |
| `flow_threshold` | 0.4 | Cellposeのflowしきい値 |
| `cellprob_threshold` | 0.0 | Cellposeの細胞確率しきい値 |

### generate_cytoplasmic_rings

| パラメータ | デフォルト | 説明 |
|-----------|---------|-------------|
| `ring_width` | 2 | 細胞質リングの幅（ピクセル） |

---

## 出力ファイル

| ファイル/ディレクトリ | 説明 |
|----------------|-------------|
| `segmentation_nuc/` | Cellposeセグメンテーション結果 |
| `tracked_labels/` | トラッキング済みラベル画像（TIFF） |
| `cyto.tif` | 細胞質リング画像 |
| `cn_ratio_timelapse.gif` | C/N比のアニメーション可視化 |

---

## トラッキングアルゴリズム

KTRtrackerは2段階のLAP（線形割当問題）アプローチを使用します：

1. **Frame-to-frame linking**: ハンガリアン法を使用して連続フレーム間でオブジェクトをリンク
2. **Gap closing**: 途切れたトラックを再接続
   - トラックの終端と始端を検出
   - 距離とフレームギャップに基づいてコスト行列を計算
   - ハンガリアン法で最適マッチング
   - 欠損フレームを線形補間（x、y、面積、C/N比）

---

## トラブルシューティング

### セグメンテーションの問題

- `flow_threshold`を調整（低いほど検出が緩やか）
- `cellprob_threshold`を調整（低いほど多くの細胞を検出）
- 画像の前処理を確認（バックグラウンド減算、コントラスト）

### トラッキングの問題

- オブジェクトが高速に移動する場合は`max_linking_distance`を増加
- オブジェクトが複数フレームで消える場合は`max_gap_closing`を増加
- より短いトラックを保持するには`min_track_length`を減少

### Gap Closingが機能しない

- `max_gap_distance`が十分に大きいか確認
- `max_gap_closing`がギャップの長さをカバーしているか確認
- トラッキング結果を確認: `print(analyzer.tracking_df)`

---

## ライセンス

MIT License

## 著者

後藤祐平, 2025
