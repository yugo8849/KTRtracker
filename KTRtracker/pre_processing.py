import numpy as np
import tifffile
import matplotlib.pyplot as plt
from cellpose import models, io
from skimage import morphology
import pandas as pd
from skimage import measure
import seaborn as sns

def load_tiff_image(filepath):
    """
    Load TIFF image and display basic information
    
    Parameters:
    -----------
    filepath : str
        Path to the image file
    
    Returns:
    --------
    imgs : ndarray
        Loaded image data
    
    Notes:
    ------
    Reads TIFF image, prints shape and timepoints,
    and displays first channel images for visual inspection.
    """
    imgs = tifffile.imread(filepath)
    print(f"Image shape: {imgs.shape}")
    print(f"Timepoints: {imgs.shape[0]}")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imgs[0][0], cmap='gray')
    plt.title('First timepoint Ch1')
    plt.subplot(1, 2, 2)
    plt.imshow(imgs[0][1], cmap='gray')
    plt.title('First timepoint Ch2')
    plt.tight_layout()
    plt.show()
    
    return imgs

def cellpose_segmentation(imgs, gpu=True, output_dir='segmentation_nuc/', 
                           flow_threshold=0.4, cellprob_threshold=0.0):
    """
    Perform nuclear segmentation using Cellpose
    
    Parameters:
    -----------
    imgs : ndarray
        Input images for segmentation
    gpu : bool, optional
        Use GPU for computation (default: True)
    output_dir : str, optional
        Directory to save segmentation results (default: 'segmentation_nuc/')
    flow_threshold : float, optional
        Cellpose flow threshold (default: 0.4)
    cellprob_threshold : float, optional
        Cellpose cell probability threshold (default: 0.0)
    
    Returns:
    --------
    labels : list
        Segmentation labels for each timepoint
    
    Notes:
    ------
    Uses Cellpose model for nuclear segmentation with customizable parameters.
    Saves segmentation results as PNG files in the specified output directory.
    """
    model = models.CellposeModel(gpu=gpu)
    labels = []
    for i, img in enumerate(imgs):
        mask, flow, style = model.eval(img[1], 
                                       flow_threshold=flow_threshold,
                                       cellprob_threshold=cellprob_threshold)
        labels.append(mask)
        io.save_to_png(img, mask, flow, f'{output_dir}{i}.png')
    
    return labels

