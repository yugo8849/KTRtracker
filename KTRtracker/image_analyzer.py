"""
KTR tracker class
==================

Examples

# Complete workflow in one go
analyzer = ImageAnalyzer('your_image.tif')
    (analyzer.load_image()
             .segment_nuclei()
             .generate_cytoplasmic_rings()
             .track_objects()
             .convert_to_tracked_labels())

# Visualize results
intensity_df = analyzer.extract_intensity_features()
analyzer.visualize_cn_ratio(intensity_df)
analyzer.visualize_cn_ratio_timelapse(intensity_df)
    
# Analyze tracks
track_stats = analyzer.analyze_tracks()
print(track_stats)
    
# Plot tracks
analyzer.plot_tracks()

Author: Yuhei Goto
Date: 2025
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt

from .pre_processing import (
    load_tiff_image, 
    cellpose_segmentation
)
from .post_processing import (
    generate_cytoplasmic_ring, 
    extract_intensity_features, 
    visualize_cn_ratio, 
    visualize_cn_ratio_timelapse
)
from .simple_lap_tracker import SimpleLAPTracker
from .tracking_utils import (
    save_tracking_as_labels, 
    analyze_tracks, 
    plot_tracks
)

class ImageAnalyzer:
    def __init__(self, filepath, min_track_length=3, max_linking_distance=15):
        """
        Image analysis workflow management
        
        Parameters:
        -----------
        filepath : str
            Path to the input TIFF image
        """
        self.filepath = filepath
        self.original_images = None
        self.segmentation_labels = None
        self.cytoplasmic_rings = None
        self.tracking_df = None
        self.tracked_labels = None

        # Tracking parameters
        self.min_track_length = min_track_length
        self.max_linking_distance = max_linking_distance
        
    def load_image(self):
        """
        Load TIFF image and display basic information
        
        Returns:
        --------
        self : ImageAnalyzer
            Returns self for method chaining
        """
        self.original_images = load_tiff_image(self.filepath)
        return self
    
    def segment_nuclei(self, 
                        gpu=True, 
                        output_dir='segmentation_nuc/', 
                        flow_threshold=0.4, 
                        cellprob_threshold=0.0):
        """
        Perform nuclear segmentation using Cellpose
        
        Parameters:
        -----------
        gpu : bool, optional
            Use GPU for computation
        output_dir : str, optional
            Directory to save segmentation results
        flow_threshold : float, optional
            Cellpose flow threshold
        cellprob_threshold : float, optional
            Cellpose cell probability threshold
        
        Returns:
        --------
        self : ImageAnalyzer
            Returns self for method chaining
        """
        if self.original_images is None:
            raise ValueError("Image not loaded. Call load_image() first.")
        
        self.segmentation_labels = cellpose_segmentation(
            self.original_images, 
            gpu=gpu, 
            output_dir=output_dir, 
            flow_threshold=flow_threshold, 
            cellprob_threshold=cellprob_threshold
        )
        return self
    
    def track_objects(self, min_track_length=None, max_linking_distance=None, max_gap_frames=2, max_gap_distance=20):
        """
        Perform object tracking
        
        Parameters:
        -----------
        min_track_length : int, optional
            Override the default min_track_length
        max_linking_distance : float, optional
            Override the default max_linking_distance
        
        Returns:
        --------
        self : ImageAnalyzer
            Returns self for method chaining
        """
        if self.segmentation_labels is None:
            raise ValueError("Nuclear segmentation not performed. Call segment_nuclei() first.")
        
        # Use provided parameters or fall back to instance parameters
        track_length = min_track_length if min_track_length is not None else self.min_track_length
        link_distance = max_linking_distance if max_linking_distance is not None else self.max_linking_distance
        
        tracker = SimpleLAPTracker(
            max_linking_distance=link_distance, 
            min_track_length=track_length,
            max_gap_frames=None, 
            max_gap_distance=None
        )
        
        self.tracking_df = tracker.track(self.segmentation_labels)
        self.tracking_df = tracker.gap_closing(self.tracking_df)
        self.tracking_df = tracker.filter_tracks(self.tracking_df)
        
        return self
    
    def convert_to_tracked_labels(self, output_dir='tracked_labels'):
        """
        Convert tracking results to labeled images
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save tracked label images
        
        Returns:
        --------
        self : ImageAnalyzer
            Returns self for method chaining
        """
        if self.tracking_df is None:
            raise ValueError("Tracking not performed. Call track_objects() first.")
        
        self.tracked_labels = save_tracking_as_labels(
            self.segmentation_labels, 
            self.tracking_df, 
            output_dir=output_dir
        )
        return self
    
    def generate_cytoplasmic_rings(self, ring_width=2):
            """
            Generate cytoplasmic rings from nuclear labels
            
            Parameters:
            -----------
            ring_width : int, optional
                Width of the cytoplasmic ring
            
            Returns:
            --------
            self : ImageAnalyzer
                Returns self for method chaining
            """
            if self.tracked_labels is None:
                raise ValueError("Nuclear segmentation not performed. Call segment_nuclei() first.")
            
            self.cytoplasmic_rings = generate_cytoplasmic_ring(
                self.tracked_labels, 
                ring_width=ring_width
            )
            return self

    def extract_intensity_features(self):
        """
        Extract intensity features from nuclear and cytoplasmic labels
        
        Returns:
        --------
        DataFrame
            Intensity features for each cell and timepoint
        """
        if (self.tracked_labels is None or 
            self.cytoplasmic_rings is None or 
            self.original_images is None):
            raise ValueError("Segmentation and cytoplasmic rings not generated. Call segment_nuclei() and generate_cytoplasmic_rings() first.")
        
        return extract_intensity_features(
            self.tracked_labels, 
            self.cytoplasmic_rings, 
            self.original_images
        )
    
    def visualize_cn_ratio(self, intensity_features=None):
        """
        Visualize Cytoplasmic/Nuclear (C/N) ratio
        
        Parameters:
        -----------
        intensity_features : DataFrame, optional
            If not provided, will extract from the current images
        """
        if intensity_features is None:
            intensity_features = self.extract_intensity_features()
        
        visualize_cn_ratio(intensity_features)
    
    def visualize_cn_ratio_timelapse(
        self, 
        intensity_features=None, 
        save_path='cn_ratio_timelapse.gif', 
        **kwargs
    ):
        """
        Visualize C/N ratio as a timelapse
        
        Parameters:
        -----------
        intensity_features : DataFrame, optional
            If not provided, will extract from the current images
        save_path : str, optional
            Path to save the timelapse gif
        **kwargs : dict
            Additional arguments for visualize_cn_ratio_timelapse
        """
        if intensity_features is None:
            intensity_features = self.extract_intensity_features()
        
        # Add a time column if not present
        if 'Time' not in intensity_features.columns:
            intensity_features['Time'] = intensity_features.groupby('label').cumcount()
        
        # Rename 'C/N' column to match the function's expectation
        intensity_features = intensity_features.rename(columns={'C/N': 'CN_ratio'})
        
        visualize_cn_ratio_timelapse(
            label_nuc_series=self.tracked_labels,
            df=intensity_features,
            save_path=save_path,
            time_column='Time',
            cn_ratio_column='CN_ratio',
            **kwargs
        )

