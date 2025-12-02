"""
Napari Tools for Tracking
==========================
napariを使用したトラッキング結果の可視化と対話的編集

Functions:
    view_tracking_in_napari: napariでトラッキング結果を表示
    get_edited_labels_from_napari: napariで編集したラベルを取得

Author: Claude
Date: 2024
"""

import numpy as np


def view_tracking_in_napari(original_images, tracked_labels, tracking_df=None):
    """
    napariでトラッキング結果を表示
    
    Parameters
    ----------
    original_images : list of ndarray
        元の画像（グレースケールまたはラベル）
    tracked_labels : list of ndarray
        track_idでラベリングされた画像
    tracking_df : DataFrame, optional
        トラッキング結果のDataFrame
    
    Returns
    -------
    viewer : napari.Viewer
        napariビューアオブジェクト
    
    Notes
    -----
    napariがインストールされている必要があります:
        pip install napari[all] --break-system-packages
    """
    try:
        import napari
    except ImportError:
        raise ImportError(
            "napariがインストールされていません。\n"
            "以下のコマンドでインストールしてください:\n"
            "  pip install napari[all] --break-system-packages"
        )
    
    # 3D配列に変換（T, Y, X）
    original_stack = np.stack(original_images, axis=0)
    tracked_stack = np.stack(tracked_labels, axis=0)
    
    # napariビューアを起動
    viewer = napari.Viewer()
    
    # 元の画像を追加
    viewer.add_image(
        original_stack,
        name='Original',
        colormap='gray',
        opacity=0.7
    )
    
    # トラッキングラベルを追加（編集可能）
    labels_layer = viewer.add_labels(
        tracked_stack,
        name='Tracked Labels (editable)',
        opacity=0.5
    )
    
    # track_idごとの色をランダムに設定
    unique_labels = np.unique(tracked_stack)
    unique_labels = unique_labels[unique_labels > 0]  # 0（背景）を除く
    
    print(f"\nnapariで表示中...")
    print(f"- 総トラック数: {len(unique_labels)}")
    print(f"- フレーム数: {len(tracked_labels)}")
    print("\n操作方法:")
    print("- スライダーでフレームを移動")
    print("- ペイントツールで編集可能")
    print("- 消しゴムツールで削除可能")
    print("- 特定のtrack_idを選択して編集")
    
    if tracking_df is not None:
        print(f"\ntrack_idの範囲: {tracking_df['track_id'].min()} - {tracking_df['track_id'].max()}")
    
    return viewer


def get_edited_labels_from_napari(viewer, layer_name='Tracked Labels (editable)'):
    """
    napariで編集したラベルを取得
    
    Parameters
    ----------
    viewer : napari.Viewer
        napariビューア
    layer_name : str, default='Tracked Labels (editable)'
        ラベルレイヤーの名前
    
    Returns
    -------
    edited_labels : list of ndarray
        編集後のラベル画像リスト
    """
    # レイヤーからラベルデータを取得
    labels_layer = viewer.layers[layer_name]
    edited_stack = labels_layer.data
    
    # リストに変換
    edited_labels = [edited_stack[i] for i in range(edited_stack.shape[0])]
    
    print(f"編集後のラベルを取得しました（{len(edited_labels)} フレーム）")
    
    return edited_labels
