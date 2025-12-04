"""
Simple LAP Tracker with Gap Closing
Python implementation equivalent to ImageJ Trackmate Simple LAP tracker
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage import measure


def extract_objects_from_labels(label_image, frame_number):
    """
    Extract object features from a labeled image
    
    Parameters
    ----------
    label_image : ndarray
        2D labeled image where each object has a unique label number
    frame_number : int
        Frame number
    
    Returns
    -------
    df : DataFrame
        DataFrame containing features of each object
    """
    props = measure.regionprops(label_image)
    
    data = []
    for prop in props:
        data.append({
            'frame': frame_number,
            'label': prop.label,
            'y': prop.centroid[0],
            'x': prop.centroid[1],
            'area': prop.area,
        })
    
    return pd.DataFrame(data)


class SimpleLAPTracker:
    """
    Simple LAP (Linear Assignment Problem) Tracker
    Python implementation equivalent to ImageJ Trackmate Simple LAP tracker
    """
    
    def __init__(self, max_linking_distance=15, max_gap_closing=2, 
                 max_gap_distance=15, min_track_length=3):
        """
        Parameters
        ----------
        max_linking_distance : float
            Maximum distance for linking objects between consecutive frames (pixels)
        max_gap_closing : int
            Maximum number of frames for gap closing
        max_gap_distance : float
            Maximum distance for gap closing (pixels)
        min_track_length : int
            Minimum track length (number of frames)
        """
        self.max_linking_distance = max_linking_distance
        self.max_gap_closing = max_gap_closing
        self.max_gap_distance = max_gap_distance
        self.min_track_length = min_track_length
        self.tracks = {}
        self.next_track_id = 0
        
    def calculate_cost_matrix(self, objects1, objects2, max_distance):
        """
        Calculate cost matrix between two sets of objects
        """
        n1 = len(objects1)
        n2 = len(objects2)
        
        if n1 == 0 or n2 == 0:
            return np.array([]).reshape(n1, n2)
        
        # Initialize cost matrix with large values
        cost_matrix = np.full((n1, n2), max_distance * 10)
        
        # Calculate Euclidean distance
        for i, (_, obj1) in enumerate(objects1.iterrows()):
            for j, (_, obj2) in enumerate(objects2.iterrows()):
                dist = np.sqrt((obj1['x'] - obj2['x'])**2 + (obj1['y'] - obj2['y'])**2)
                if dist <= max_distance:
                    cost_matrix[i, j] = dist
        
        return cost_matrix
    
    def track(self, label_images):
        """
        Perform tracking from a list of labeled images
        
        Parameters
        ----------
        label_images : list of ndarray
            List of labeled images (in chronological order)
        
        Returns
        -------
        tracking_df : DataFrame
            Tracking results
        """
        # Extract object information from all frames
        all_objects = []
        for frame_idx, label_image in enumerate(label_images):
            objects = extract_objects_from_labels(label_image, frame_idx)
            all_objects.append(objects)
        
        # Perform tracking
        return self.link_objects(all_objects)
    
    def link_objects(self, all_objects):
        """
        Link objects across all frames (frame-to-frame linking)
        
        Parameters
        ----------
        all_objects : list of DataFrame
            Object information for each frame
        
        Returns
        -------
        tracking_df : DataFrame
            Tracking results
        """
        self.tracks = {}
        self.next_track_id = 0
        
        if len(all_objects) == 0 or len(all_objects[0]) == 0:
            return pd.DataFrame()
        
        # Initialize with objects from the first frame
        for _, obj in all_objects[0].iterrows():
            self.tracks[self.next_track_id] = [obj.to_dict()]
            self.next_track_id += 1
        
        # Frame-to-frame linking
        for frame_idx in range(1, len(all_objects)):
            prev_objects = all_objects[frame_idx - 1]
            curr_objects = all_objects[frame_idx]
            
            if len(curr_objects) == 0:
                continue
            
            if len(prev_objects) == 0:
                # No objects in previous frame, start new tracks
                for _, obj in curr_objects.iterrows():
                    self.tracks[self.next_track_id] = [obj.to_dict()]
                    self.next_track_id += 1
                continue
            
            # Calculate cost matrix
            cost_matrix = self.calculate_cost_matrix(
                prev_objects, curr_objects, self.max_linking_distance
            )
            
            # Optimal assignment using Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Record linked objects in current frame
            linked_curr = set()
            
            # Process assignment results
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] <= self.max_linking_distance:
                    # Valid link - add to existing track
                    prev_obj = prev_objects.iloc[i]
                    curr_obj = curr_objects.iloc[j]
                    
                    # Find the track this object belongs to
                    for track_id, track in self.tracks.items():
                        if (track[-1]['frame'] == prev_obj['frame'] and 
                            track[-1]['label'] == prev_obj['label']):
                            track.append(curr_obj.to_dict())
                            linked_curr.add(j)
                            break
            
            # Unlinked objects in current frame start new tracks
            for j, (_, obj) in enumerate(curr_objects.iterrows()):
                if j not in linked_curr:
                    self.tracks[self.next_track_id] = [obj.to_dict()]
                    self.next_track_id += 1
        
        # Convert tracking results to DataFrame
        return self._tracks_to_dataframe()
    
    def _tracks_to_dataframe(self):
        """Convert tracks dictionary to DataFrame"""
        data = []
        for track_id, track in self.tracks.items():
            for spot in track:
                row = spot.copy()
                row['track_id'] = track_id
                data.append(row)
        
        if len(data) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        return df.sort_values(['track_id', 'frame']).reset_index(drop=True)
    
    def gap_closing(self, tracking_df, max_gap_frames=None, max_gap_distance=None):
        """
        Gap closing: Reconnect broken tracks
        
        Reconnects tracks that were broken during initial frame-to-frame linking
        based on distance and frame gap between track endpoints.
        
        Parameters
        ----------
        tracking_df : pandas.DataFrame
            Initial tracking results (output from track())
        max_gap_frames : int, optional
            Maximum number of frames to bridge (default: self.max_gap_closing)
        max_gap_distance : float, optional
            Maximum distance for gap closing (default: self.max_gap_distance)
        
        Returns
        -------
        updated_tracking_df : pandas.DataFrame
            Tracking results with gap closing applied
        """
        if max_gap_frames is None:
            max_gap_frames = self.max_gap_closing
        if max_gap_distance is None:
            max_gap_distance = self.max_gap_distance
        
        if tracking_df.empty:
            return tracking_df
        
        df = tracking_df.copy()
        
        # Get endpoints of each track
        track_endpoints = self._get_track_endpoints(df)
        
        # Build cost matrix for gap closing
        # Rows: track endpoints, Columns: track start points
        track_ids = list(track_endpoints.keys())
        n_tracks = len(track_ids)
        
        if n_tracks <= 1:
            return df
        
        # Build cost matrix (from endpoints to start points)
        cost_matrix = np.full((n_tracks, n_tracks), np.inf)
        
        for i, track_i in enumerate(track_ids):
            end_info = track_endpoints[track_i]['end']
            end_frame = end_info['frame']
            end_x = end_info['x']
            end_y = end_info['y']
            
            for j, track_j in enumerate(track_ids):
                if track_i == track_j:
                    continue  # Skip same track
                
                start_info = track_endpoints[track_j]['start']
                start_frame = start_info['frame']
                start_x = start_info['x']
                start_y = start_info['y']
                
                # Check frame gap (track_i endpoint must be before track_j start point)
                frame_gap = start_frame - end_frame - 1
                
                if frame_gap < 1 or frame_gap > max_gap_frames:
                    continue  # Invalid gap
                
                # Calculate distance
                dist = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                
                if dist <= max_gap_distance:
                    # Include frame gap in cost
                    cost_matrix[i, j] = dist * (1 + 0.1 * frame_gap)
        
        # Find pairs to merge (greedy approach)
        merge_pairs = self._find_gap_closing_pairs(cost_matrix, track_ids)
        
        # Merge tracks
        df = self._merge_tracks(df, merge_pairs, track_endpoints, max_gap_frames)
        
        return df.sort_values(['track_id', 'frame']).reset_index(drop=True)
    
    def _get_track_endpoints(self, df):
        """Get start and end point information for each track"""
        endpoints = {}
        
        for track_id in df['track_id'].unique():
            track = df[df['track_id'] == track_id].sort_values('frame')
            
            first = track.iloc[0]
            last = track.iloc[-1]
            
            endpoints[track_id] = {
                'start': {
                    'frame': first['frame'],
                    'x': first['x'],
                    'y': first['y'],
                    'label': first['label']
                },
                'end': {
                    'frame': last['frame'],
                    'x': last['x'],
                    'y': last['y'],
                    'label': last['label']
                }
            }
        
        return endpoints
    
    def _find_gap_closing_pairs(self, cost_matrix, track_ids):
        """
        Find track pairs to merge for gap closing
        
        Uses Hungarian algorithm for optimal matching and returns only valid pairs
        """
        n = len(track_ids)
        
        # Check if there are any finite costs
        if not np.any(np.isfinite(cost_matrix)):
            return []
        
        # Create extended cost matrix (add option not to link)
        # Add option not to link with cost higher than threshold
        threshold = self.max_gap_distance * 2
        extended_cost = np.full((n * 2, n * 2), threshold)
        
        # Top-left: actual cost matrix
        extended_cost[:n, :n] = np.where(np.isinf(cost_matrix), threshold, cost_matrix)
        
        # Bottom-right: zero cost (for not linking)
        np.fill_diagonal(extended_cost[n:, n:], 0)
        
        # Top-right: cost for not linking
        np.fill_diagonal(extended_cost[:n, n:], threshold * 0.9)
        
        # Bottom-left: cost for not linking
        np.fill_diagonal(extended_cost[n:, :n], threshold * 0.9)
        
        # Solve using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(extended_cost)
        
        # Extract valid pairs
        merge_pairs = []
        for i, j in zip(row_ind, col_ind):
            if i < n and j < n and np.isfinite(cost_matrix[i, j]):
                merge_pairs.append((track_ids[i], track_ids[j]))
        
        return merge_pairs
    
    def _merge_tracks(self, df, merge_pairs, track_endpoints, max_gap_frames):
        """
        Merge tracks and fill gaps with interpolation
        """
        if not merge_pairs:
            return df
        
        df = df.copy()
        
        # Cache track data before any modifications
        track_data_cache = {}
        for track_id in df['track_id'].unique():
            track_data_cache[track_id] = df[df['track_id'] == track_id].sort_values('frame').copy()
        
        # Build mapping for tracks to merge
        # Merge track_j into track_i
        merge_map = {}
        for track_i, track_j in merge_pairs:
            # Handle chain cases
            target = track_i
            while target in merge_map:
                target = merge_map[target]
            merge_map[track_j] = target
        
        # Store interpolated data
        interpolated_data = []
        
        # Identify numeric columns for interpolation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['frame', 'track_id', 'label']
        interp_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        for track_i, track_j in merge_pairs:
            # Merge track_j into track_i
            target_id = track_i
            while target_id in merge_map and merge_map[target_id] != target_id:
                if merge_map[target_id] == target_id:
                    break
                target_id = merge_map[target_id]
            
            # Get endpoint and start point data from cache
            track_i_data = track_data_cache.get(track_i)
            track_j_data = track_data_cache.get(track_j)
            
            # Skip if either track data is missing or empty
            if track_i_data is None or track_j_data is None:
                continue
            if len(track_i_data) == 0 or len(track_j_data) == 0:
                continue
            
            end_row = track_i_data.iloc[-1]
            start_row = track_j_data.iloc[0]
            
            end_frame = int(end_row['frame'])
            start_frame = int(start_row['frame'])
            
            # Use label from previous segment for interpolated points
            interp_label = end_row['label']
            
            # Interpolate gap frames
            gap_size = start_frame - end_frame - 1
            
            if gap_size > 0 and gap_size <= max_gap_frames:
                for k in range(1, gap_size + 1):
                    gap_frame = end_frame + k
                    # Interpolation weight (0 to 1)
                    weight = k / (gap_size + 1)
                    
                    # Basic data
                    interp_row = {
                        'frame': gap_frame,
                        'label': interp_label,  # Use label from previous segment
                        'track_id': target_id
                    }
                    
                    # Linear interpolation for all numeric columns
                    for col in interp_cols:
                        end_val = end_row[col]
                        start_val = start_row[col]
                        if pd.notna(end_val) and pd.notna(start_val):
                            interp_row[col] = end_val + (start_val - end_val) * weight
                        else:
                            interp_row[col] = np.nan
                    
                    interpolated_data.append(interp_row)
            
            # Update track_id of track_j to target_id
            df.loc[df['track_id'] == track_j, 'track_id'] = target_id
        
        # Add interpolated data
        if interpolated_data:
            interp_df = pd.DataFrame(interpolated_data)
            df = pd.concat([df, interp_df], ignore_index=True)
        
        return df
    
    def filter_tracks(self, tracking_df, min_length=None):
        """
        Filter out short tracks
        
        Parameters
        ----------
        tracking_df : DataFrame
            Tracking results
        min_length : int, optional
            Minimum track length (default: self.min_track_length)
        
        Returns
        -------
        filtered_df : DataFrame
            Filtered tracking results
        """
        if min_length is None:
            min_length = self.min_track_length
        
        # Calculate length of each track
        track_lengths = tracking_df.groupby('track_id').size()
        
        # Get track IDs with length >= min_length
        valid_tracks = track_lengths[track_lengths >= min_length].index
        
        # Filter
        filtered_df = tracking_df[tracking_df['track_id'].isin(valid_tracks)].copy()
        
        return filtered_df.reset_index(drop=True)


def create_gap_closing_test_data(n_frames=20, n_objects=3, image_size=100, 
                                  gap_frames=None, seed=42):
    """
    Generate labeled images for gap closing testing
    
    Parameters
    ----------
    n_frames : int
        Number of frames
    n_objects : int
        Number of objects
    image_size : int
        Image size
    gap_frames : dict, optional
        Frames where each object disappears {obj_id: [frame1, frame2, ...]}
        If None, gaps are inserted automatically
    seed : int
        Random seed
    
    Returns
    -------
    label_images : list of ndarray
        List of labeled images
    ground_truth : dict
        Ground truth track information
    """
    np.random.seed(seed)
    
    # Initial positions and velocities of objects
    positions = np.random.rand(n_objects, 2) * (image_size - 40) + 20
    velocities = (np.random.rand(n_objects, 2) - 0.5) * 4
    
    # Default gap frame settings
    if gap_frames is None:
        gap_frames = {
            0: [7, 8],      # Object 0 disappears at frames 7, 8
            1: [12],        # Object 1 disappears at frame 12
            2: [5, 15, 16], # Object 2 disappears at frames 5, 15, 16
        }
    
    label_images = []
    ground_truth = {i: [] for i in range(n_objects)}
    
    for frame in range(n_frames):
        label_image = np.zeros((image_size, image_size), dtype=np.uint16)
        
        for obj_id in range(n_objects):
            # Check if object disappears at this frame
            if obj_id in gap_frames and frame in gap_frames[obj_id]:
                continue
            
            # Update position
            positions[obj_id] += velocities[obj_id]
            
            # Reflect at boundaries
            for dim in range(2):
                if positions[obj_id, dim] < 15 or positions[obj_id, dim] > image_size - 15:
                    velocities[obj_id, dim] *= -1
                    positions[obj_id, dim] = np.clip(positions[obj_id, dim], 15, image_size - 15)
            
            # Draw object
            y, x = int(positions[obj_id, 0]), int(positions[obj_id, 1])
            radius = 8
            yy, xx = np.ogrid[:image_size, :image_size]
            mask = (yy - y)**2 + (xx - x)**2 <= radius**2
            label_image[mask] = obj_id + 1
            
            # Add to ground truth
            ground_truth[obj_id].append({
                'frame': frame,
                'x': x,
                'y': y,
                'label': obj_id + 1
            })
        
        label_images.append(label_image)
    
    return label_images, ground_truth


# Usage example
if __name__ == '__main__':
    # Generate test data
    print("Generating test data...")
    label_images, ground_truth = create_gap_closing_test_data(
        n_frames=20, 
        n_objects=3,
        gap_frames={0: [7, 8], 1: [12], 2: [5, 15, 16]}
    )
    
    print(f"Number of frames generated: {len(label_images)}")
    print(f"Gap frames: Object 0=[7,8], Object 1=[12], Object 2=[5,15,16]")
    
    # Initialize tracker
    tracker = SimpleLAPTracker(
        max_linking_distance=20,
        max_gap_closing=3,
        max_gap_distance=25,
        min_track_length=3
    )
    
    # Initial tracking
    print("\n--- Initial Tracking ---")
    tracking_df = tracker.track(label_images)
    print(f"Number of tracks: {tracking_df['track_id'].nunique()}")
    print(f"Track IDs: {sorted(tracking_df['track_id'].unique())}")
    
    # Display length of each track
    for tid in sorted(tracking_df['track_id'].unique()):
        track = tracking_df[tracking_df['track_id'] == tid]
        frames = sorted(track['frame'].tolist())
        print(f"  Track {tid}: frames {frames[0]}-{frames[-1]}, length={len(frames)}")
    
    # Gap closing
    print("\n--- Gap Closing ---")
    gap_closed_df = tracker.gap_closing(tracking_df, max_gap_frames=3, max_gap_distance=25)
    print(f"Number of tracks after gap closing: {gap_closed_df['track_id'].nunique()}")
    
    # Display length of each track
    for tid in sorted(gap_closed_df['track_id'].unique()):
        track = gap_closed_df[gap_closed_df['track_id'] == tid]
        frames = sorted(track['frame'].tolist())
        interpolated = track[track['label'] == -1]
        print(f"  Track {tid}: frames {frames[0]}-{frames[-1]}, length={len(frames)}, interpolated={len(interpolated)}")
    
    # Filtering
    print("\n--- Filtering ---")
    filtered_df = tracker.filter_tracks(gap_closed_df, min_length=5)
    print(f"Number of tracks after filtering: {filtered_df['track_id'].nunique()}")
