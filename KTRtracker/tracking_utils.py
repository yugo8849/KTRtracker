"""
Tracking Utilities
==================
Utility functions of save, visualize, and analyze the results of tracking


Functions:
    save_tracking_as_labels: save tracking results as the labeled images
    remove_tracks_by_id: remove track_id
    reconstruct_dataframe_from_labels: reconstruct DataFrame from the labeled images
    analyze_tracks: Calculate the statics of tracking
    plot_tracks: visualize the tracking results
    custom_filter: 
    create_test_data: 

Author: Yuhei Goto
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage import measure, io
import os


def create_test_data(n_frames=10, n_objects=5, image_size=512):
    """
    create test labeled images
    
    Parameters
    ----------
    n_frames : int, default=10
        
    n_objects : int, default=5
        
    image_size : int, default=512
       
    
    Returns
    -------
    label_images : list of ndarray
        
    """
    np.random.seed(42)
    label_images = []
    
    # Randamize initial position
    positions = np.random.rand(n_objects, 2) * (image_size - 100) + 50
    velocities = (np.random.rand(n_objects, 2) - 0.5) * 10  # Random velocities
    
    for frame in range(n_frames):
        label_image = np.zeros((image_size, image_size), dtype=np.uint16)
        
        for obj_id in range(n_objects):
            # update the positions
            positions[obj_id] += velocities[obj_id]
            
            # reflect at the boundary
            for dim in range(2):
                if positions[obj_id, dim] < 20 or positions[obj_id, dim] > image_size - 20:
                    velocities[obj_id, dim] *= -1
                    positions[obj_id, dim] = np.clip(positions[obj_id, dim], 20, image_size - 20)
            
            # depict the object (circle)
            y, x = int(positions[obj_id, 0]), int(positions[obj_id, 1])
            radius = 10
            yy, xx = np.ogrid[:image_size, :image_size]
            mask = (yy - y)**2 + (xx - x)**2 <= radius**2
            label_image[mask] = obj_id + 1
        
        label_images.append(label_image)
    
    return label_images


def save_tracking_as_labels(original_labels, tracking_df, output_dir='tracked_labels'):
    """
    Generate the tracked labeled image from original labeled images by swapping objects with track_id
    Supports interpolated points from gap_closing (draws circles based on area)
    
    Parameters
    ----------
    original_labels : list of ndarray
        
    tracking_df : DataFrame
        
    output_dir : str, default='tracked_labels'
   
    Returns
    -------
    tracked_labels : list of ndarray
    """
    from skimage import draw
    
    # make an output directory
    os.makedirs(output_dir, exist_ok=True)
    
    tracked_labels = []
    image_shape = original_labels[0].shape
    
    for frame_num in range(len(original_labels)):
        # original labeled images
        original = original_labels[frame_num].copy()
        
        # new labeled images（labeled with track_id）
        tracked = np.zeros_like(original, dtype=np.uint16)
        
        # Get the trackig imnfomation at the frame 
        frame_data = tracking_df[tracking_df['frame'] == frame_num]
        
        # Replcae labels with track_id
        for _, row in frame_data.iterrows():
            original_label = int(row['label'])
            track_id = int(row['track_id'])
            
            # Check if original mask exists
            original_mask = original == original_label
            
            if np.any(original_mask):
                # Use original mask if exists
                tracked[original_mask] = track_id + 1
            else:
                # Interpolated point: draw circle based on area
                y, x = int(row['y']), int(row['x'])
                
                # Calculate radius from area (area = π * r^2 -> r = sqrt(area / π))
                area = row.get('area', 317)  # default radius ~10
                if pd.notna(area) and area > 0:
                    radius = int(np.sqrt(area / np.pi))
                else:
                    radius = 10  # default
                
                # Draw circle
                if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
                    rr, cc = draw.disk((y, x), radius, shape=image_shape)
                    tracked[rr, cc] = track_id + 1
        
        tracked_labels.append(tracked)
        
        # Save as file
        filename = os.path.join(output_dir, f'tracked_frame_{frame_num:04d}.tif')
        io.imsave(filename, tracked, check_contrast=False)
    
    return tracked_labels


def remove_tracks_by_id(tracked_labels, tracking_df, track_ids_to_remove):
    """
        
    Parameters
    ----------
    tracked_labels : list of ndarray
        
    tracking_df : DataFrame

    track_ids_to_remove : list
    
    Returns
    -------
    cleaned_labels : list of ndarray
        
    cleaned_df : DataFrame
        
    """
    # Remove track_id from DataFrame
    cleaned_df = tracking_df[~tracking_df['track_id'].isin(track_ids_to_remove)].copy()
    
    # Remove labels of track_id from labeled iamges
    cleaned_labels = []
    for label_img in tracked_labels:
        cleaned_img = label_img.copy()
        for track_id in track_ids_to_remove:
            cleaned_img[cleaned_img == track_id + 1] = 0  
        cleaned_labels.append(cleaned_img)
    
    return cleaned_labels, cleaned_df


def reconstruct_dataframe_from_labels(edited_labels):
    """
    
    Parameters
    ----------
    edited_labels : list of ndarray
    
    Returns
    -------
    df : DataFrame
    
    """
    all_data = []
    
    for frame_num, label_img in enumerate(edited_labels):
   
        props = measure.regionprops(label_img)
        
        for prop in props:
            all_data.append({
                'frame': frame_num,
                'track_id': prop.label - 1,  # Revert offset applied during saving
                'label': prop.label,
                'y': prop.centroid[0],
                'x': prop.centroid[1],
                'area': prop.area,
                'perimeter': prop.perimeter,
                'eccentricity': prop.eccentricity
            })
    
    df = pd.DataFrame(all_data)
    
    return df


def analyze_tracks(df):
    """
    
    Parameters
    ----------
    df : DataFrame
    
    Returns
    -------
    track_stats : DataFrame

    """
    track_stats = []
    
    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id].sort_values('frame')
        
        # Calculate track length
        track_length = len(track_data)
        
        # Calculate the total moving length
        positions = track_data[['x', 'y']].values
        if len(positions) > 1:
            displacements = np.diff(positions, axis=0)
            total_distance = np.sum(np.sqrt((displacements**2).sum(axis=1)))
        else:
            total_distance = 0
        
        # Straight length
        if len(positions) > 1:
            straight_distance = np.sqrt(
                (positions[-1, 0] - positions[0, 0])**2 + 
                (positions[-1, 1] - positions[0, 1])**2
            )
        else:
            straight_distance = 0
        
        track_stats.append({
            'track_id': track_id,
            'length': track_length,
            'start_frame': track_data['frame'].min(),
            'end_frame': track_data['frame'].max(),
            'total_distance': total_distance,
            'straight_distance': straight_distance,
            'mean_area': track_data['area'].mean(),
            'mean_speed': total_distance / track_length if track_length > 1 else 0
        })
    
    return pd.DataFrame(track_stats)


def plot_tracks(df, label_images, frame_to_show=0, figsize=(16, 7)):
    """
    
    Parameters
    ----------
    df : DataFrame
        tracking results df
    label_images : list of ndarray
        
    frame_to_show : int, default=0
        
    figsize : tuple, default=(16, 7)
        
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    ax.imshow(label_images[frame_to_show], cmap='gray')
    
    frame_data = df[df['frame'] == frame_to_show]
    
    for _, row in frame_data.iterrows():
        circle = Circle((row['x'], row['y']), 15, 
                       color=plt.cm.tab20(row['track_id'] % 20), 
                       fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.text(row['x'], row['y'], str(int(row['track_id'])), 
               color='white', ha='center', va='center', fontsize=8)
    
    ax.set_title(f'Frame {frame_to_show}', fontsize=14)
    ax.axis('off')
    

    ax = axes[1]
    
    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id].sort_values('frame')
        ax.plot(track_data['x'], track_data['y'], 
               color=plt.cm.tab20(track_id % 20), 
               linewidth=2, alpha=0.7, marker='o', markersize=4)
        
        first_point = track_data.iloc[0]
        ax.text(first_point['x'], first_point['y'], str(int(track_id)), 
               fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, label_images[0].shape[1])
    ax.set_ylim(label_images[0].shape[0], 0)
    ax.set_title('All Tracks', fontsize=14)
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def custom_filter(df, min_length=5, min_distance=50, min_area=50, max_area=500):
    """
    
    Parameters
    ----------
    df : DataFrame
        tracking results df
    min_length : int, default=5
        
    min_distance : float, default=50
        
    min_area : float, default=50
        
    max_area : float, default=500
        
    
    Returns
    -------
    df_filtered : DataFrame
        
    """
    # Filtering with track length
    track_lengths = df.groupby('track_id').size()
    valid_by_length = track_lengths[track_lengths >= min_length].index
    
    # Filtering with moving distance
    track_distances = []
    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id].sort_values('frame')
        positions = track_data[['x', 'y']].values
        if len(positions) > 1:
            displacements = np.diff(positions, axis=0)
            total_distance = np.sum(np.sqrt((displacements**2).sum(axis=1)))
            track_distances.append((track_id, total_distance))
    
    valid_by_distance = [tid for tid, dist in track_distances if dist >= min_distance]
    
    # Filtering with area
    mean_areas = df.groupby('track_id')['area'].mean()
    valid_by_area = mean_areas[(mean_areas >= min_area) & (mean_areas <= max_area)].index
    
    # satisy all requirements
    valid_tracks = set(valid_by_length) & set(valid_by_distance) & set(valid_by_area)
    
    df_filtered = df[df['track_id'].isin(valid_tracks)].copy()
    
    print(f"Track frame number flitering: {len(valid_by_length)} tracks")
    print(f"Track distance filtering: {len(valid_by_distance)} tracks")
    print(f"Track area filtering: {len(valid_by_area)} tracks")
    print(f"Satisfy all filters: {len(valid_tracks)} tracks")
    
    return df_filtered
