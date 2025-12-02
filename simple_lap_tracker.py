"""
Simple LAP Tracker
==================
Implementation of ImageJ Trackmate Simple LAP tracker for Python

Main Classes:
    SimpleLAPTracker: Main Class of LAP tracker

Functions:
    extract_objects_from_labels: Extract objects from labeled images created by cellpose or smothing

Author: Yuhei Goto
Date: 2025/11/25
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage import measure


def extract_objects_from_labels(label_image, frame_number):
    """
    Extract objects from labeled images created by cellpose or smothing
    
    Parameters
    ----------
    label_image : ndarray
        labeled 2D images
    frame_number : int
        frame number
    
    Returns
    -------
    df : DataFrame
        Dataframe including the chracteristics of each labeled images
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
            'perimeter': prop.perimeter,
            'eccentricity': prop.eccentricity
        })
    
    return pd.DataFrame(data)


class SimpleLAPTracker:
    """
    Simple LAP (Linear Assignment Problem) Tracker
    
    Parameters
    ----------
    max_linking_distance : float, default=15
        Maximum distance between two successive images (pixel)
    max_gap_closing : int, default=2
        Maximum frame for gap closing（not implemented yet）
    max_gap_distance : float, default=15
        Maximum distance for gap closing（not implemented yet）
    min_track_length : int, default=3
        Varid minimum tracking frame numbers
    
    Attributes
    ----------
    tracks : dict
         {track_id: [spot_dicts]}
    next_track_id : int
        next assinged track_id
    
    Examples
    --------
    >>> tracker = SimpleLAPTracker(max_linking_distance=20, min_track_length=5)
    >>> results = tracker.track(label_images)
    >>> filtered = tracker.filter_tracks(results)
    """
    
    def __init__(self, max_linking_distance=15, max_gap_closing=2, 
                 max_gap_distance=15, min_track_length=3):
        self.max_linking_distance = max_linking_distance
        self.max_gap_closing = max_gap_closing
        self.max_gap_distance = max_gap_distance
        self.min_track_length = min_track_length
        self.tracks = {}
        self.next_track_id = 0
        
    def calculate_cost_matrix(self, objects1, objects2, max_distance):
        """
        Calculate cost matrix between two object sets
        
        Parameters
        ----------
        objects1, objects2 : DataFrame
        max_distance : float
        
        Returns
        -------
        cost_matrix : ndarray
        
        """
        n1 = len(objects1)
        n2 = len(objects2)
        
        if n1 == 0 or n2 == 0:
            return np.array([]).reshape(n1, n2)
        
        # calculate the difference of the positions
        pos1 = objects1[['y', 'x']].values
        pos2 = objects2[['y', 'x']].values
        
        # calculate the Euclidean distance
        cost_matrix = np.sqrt(
            ((pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
        
        # Large cost if it exceed maximum distance
        cost_matrix[cost_matrix > max_distance] = 1e10
        
        return cost_matrix
    
    def link_objects(self, prev_objects, curr_objects, prev_track_ids):
        """
        Link objects between successive frames
        
        Parameters
        ----------
        prev_objects : DataFrame
            
        curr_objects : DataFrame
            
        prev_track_ids : list
            
        
        Returns
        -------
        track_ids : list
            
        """
        if len(prev_objects) == 0:
            # First frame: create new frame
            track_ids = []
            for idx, row in curr_objects.iterrows():
                track_id = self.next_track_id
                self.tracks[track_id] = [row.to_dict()]
                track_ids.append(track_id)
                self.next_track_id += 1
            return track_ids
        
        # Calculate cost matrix
        cost_matrix = self.calculate_cost_matrix(
            prev_objects, curr_objects, self.max_linking_distance
        )
        
        track_ids = [None] * len(curr_objects)
        
        if cost_matrix.size > 0:
            # Assign an optimal parameters by Hungarian methods
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for prev_idx, curr_idx in zip(row_ind, col_ind):
                if cost_matrix[prev_idx, curr_idx] < 1e10:
                    # Add to exiting tracks
                    track_id = prev_track_ids[prev_idx]
                    curr_row = curr_objects.iloc[curr_idx].to_dict()
                    self.tracks[track_id].append(curr_row)
                    track_ids[curr_idx] = track_id
        
        # Make new tracks with unassinged objects
        for idx, (curr_idx, row) in enumerate(curr_objects.iterrows()):
            if track_ids[idx] is None:
                track_id = self.next_track_id
                self.tracks[track_id] = [row.to_dict()]
                track_ids[idx] = track_id
                self.next_track_id += 1
        
        return track_ids
    
    def track(self, label_images):
        """
        Tracking the sequence of labeled images
        
        Parameters
        ----------
        label_images : list of ndarray
        
        Returns
        -------
        df : DataFrame
            
        """
        self.tracks = {}
        self.next_track_id = 0
        
        prev_objects = pd.DataFrame()
        prev_track_ids = []
        
        all_objects = []
        
        for frame_num, label_image in enumerate(label_images):
            # Extract objects
            curr_objects = extract_objects_from_labels(label_image, frame_num)
            
            # Linking
            track_ids = self.link_objects(prev_objects, curr_objects, prev_track_ids)
            
            # add track_id
            curr_objects['track_id'] = track_ids
            all_objects.append(curr_objects)
            
            # Save for next iteration
            prev_objects = curr_objects
            prev_track_ids = track_ids
        
        return pd.concat(all_objects, ignore_index=True)
    
    def get_tracks_dict(self):
        """
        
        Returns
        -------
        tracks : dict
            {track_id: [spot_dicts]} 
        """
        return self.tracks
    
    def filter_tracks(self, df, verbose=True):
        """
        Remove shorter tracks
        
        Parameters
        ----------
        df : DataFrame
            tracking results
        verbose : bool, default=True
        
        Returns
        -------
        df_filtered : DataFrame
        
        """
        # Calculate the length of each tracks
        track_lengths = df.groupby('track_id').size()
        
        # get the valid track numbers
        valid_tracks = track_lengths[track_lengths >= self.min_track_length].index
        
        # filtering
        df_filtered = df[df['track_id'].isin(valid_tracks)].copy()
        
        if verbose:
            print(f"All tracks: {len(track_lengths)}")
            print(f"filtered tracks: {len(valid_tracks)}")
            print(f"removed tracks: {len(track_lengths) - len(valid_tracks)}")
        
        return df_filtered
