# KTRtracker User Manual

## Overview

KTRtracker is a Python package for tracking cells and analyzing Cytoplasmic/Nuclear (C/N) ratio from time-lapse microscopy images. It provides tools for nuclear segmentation, object tracking with gap closing, and intensity analysis.

## Installation

### Requirements

```bash
pip install numpy pandas matplotlib seaborn tifffile scikit-image scipy cellpose
```

### Optional (for interactive visualization)

```bash
pip install napari
```

### Install KTRtracker

```bash
pip install -e .
```

## Quick Start

```python
from KTRtracker import ImageAnalyzer

# Full workflow
analyzer = ImageAnalyzer('your_image.tif')
(analyzer
    .load_image()
    .segment_nuclei()
    .track_objects()
    .convert_to_tracked_labels()
    .generate_cytoplasmic_rings())

# Visualize results
intensity_df = analyzer.extract_intensity_features()
analyzer.visualize_cn_ratio(intensity_df)
```

---

## Use Cases

### Case 1: Starting from Background-Subtracted Images

Use this workflow when you have raw fluorescence images that need segmentation and tracking.

#### Step-by-step Workflow

```python
from KTRtracker import ImageAnalyzer

# Initialize analyzer with image path
analyzer = ImageAnalyzer(
    'background_subtracted.tif',
    min_track_length=5,        # Minimum track length (frames)
    max_linking_distance=20,   # Maximum distance for frame-to-frame linking (pixels)
    max_gap_closing=3,         # Maximum frames to bridge gaps
    max_gap_distance=25        # Maximum distance for gap closing (pixels)
)

# Step 1: Load image
analyzer.load_image()
print(f"Loaded image shape: {analyzer.original_images.shape}")

# Step 2: Nuclear segmentation using Cellpose
analyzer.segment_nuclei(
    gpu=True,                    # Use GPU if available
    output_dir='segmentation/',  # Output directory for segmentation results
    flow_threshold=0.4,          # Cellpose flow threshold
    cellprob_threshold=0.0       # Cellpose cell probability threshold
)

# Step 3: Track objects with gap closing
analyzer.track_objects(
    min_track_length=5,
    max_linking_distance=20,
    max_gap_closing=3,
    max_gap_distance=25
)
print(f"Number of tracks: {analyzer.tracking_df['track_id'].nunique()}")

# Step 4: Convert tracking results to labeled images
analyzer.convert_to_tracked_labels(output_dir='tracked_labels/')

# Step 5: Generate cytoplasmic rings
analyzer.generate_cytoplasmic_rings(ring_width=2)

# Step 6: Extract and visualize intensity features
intensity_df = analyzer.extract_intensity_features()
analyzer.visualize_cn_ratio(intensity_df)

# Step 7: Create timelapse visualization
analyzer.visualize_cn_ratio_timelapse(
    intensity_df,
    save_path='cn_ratio_timelapse.gif'
)
```

#### Method Chaining (Compact Version)

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

### Case 2: Starting from Label Images

Use this workflow when you already have segmented label images (e.g., from Cellpose, StarDist, or manual segmentation).

#### Step-by-step Workflow

```python
from KTRtracker import ImageAnalyzer
from KTRtracker.pre_processing import load_tiff_image
import tifffile

# Initialize analyzer
analyzer = ImageAnalyzer('original_images.tif')

# Load original images for intensity analysis
analyzer.load_image()

# Load pre-existing label images
label_images = tifffile.imread('your_labels.tif')
# Convert to list if needed
if label_images.ndim == 3:
    analyzer.segmentation_labels = [label_images[i] for i in range(label_images.shape[0])]
else:
    analyzer.segmentation_labels = [label_images]

# Continue with tracking
(analyzer
    .track_objects()
    .convert_to_tracked_labels()
    .generate_cytoplasmic_rings())

# Extract and visualize intensity features
intensity_df = analyzer.extract_intensity_features()
analyzer.visualize_cn_ratio(intensity_df)
```

#### Using SimpleLAPTracker Directly

```python
from KTRtracker import SimpleLAPTracker
from KTRtracker.tracking_utils import save_tracking_as_labels
import tifffile

# Load label images
label_images = tifffile.imread('your_labels.tif')
label_list = [label_images[i] for i in range(label_images.shape[0])]

# Initialize tracker
tracker = SimpleLAPTracker(
    max_linking_distance=20,
    max_gap_closing=3,
    max_gap_distance=25,
    min_track_length=5
)

# Track objects
tracking_df = tracker.track(label_list)
print(f"Initial tracks: {tracking_df['track_id'].nunique()}")

# Apply gap closing
tracking_df = tracker.gap_closing(tracking_df)
print(f"After gap closing: {tracking_df['track_id'].nunique()}")

# Filter short tracks
tracking_df = tracker.filter_tracks(tracking_df)
print(f"After filtering: {tracking_df['track_id'].nunique()}")

# Save as tracked label images
tracked_labels = save_tracking_as_labels(
    label_list,
    tracking_df,
    output_dir='tracked_labels/'
)
```

---

### Case 3: Starting from Tracked Label Images

Use this workflow when you already have tracked label images (e.g., from TrackMate or previous analysis).

#### Step-by-step Workflow

```python
from KTRtracker import ImageAnalyzer
from KTRtracker.post_processing import (
    generate_cytoplasmic_ring,
    extract_intensity_features,
    visualize_cn_ratio
)
import tifffile

# Initialize analyzer
analyzer = ImageAnalyzer('original_images.tif')

# Load original images
analyzer.load_image()

# Load pre-tracked label images
tracked_labels = tifffile.imread('tracked_labels.tif')
if tracked_labels.ndim == 3:
    analyzer.tracked_labels = [tracked_labels[i] for i in range(tracked_labels.shape[0])]
else:
    analyzer.tracked_labels = [tracked_labels]

# Generate cytoplasmic rings and analyze
analyzer.generate_cytoplasmic_rings(ring_width=2)

# Extract and visualize intensity features
intensity_df = analyzer.extract_intensity_features()
analyzer.visualize_cn_ratio(intensity_df)
```

#### Using Functions Directly

```python
from KTRtracker.post_processing import (
    generate_cytoplasmic_ring,
    extract_intensity_features,
    visualize_cn_ratio,
    visualize_cn_ratio_timelapse
)
import tifffile

# Load data
original_images = tifffile.imread('original_images.tif')
tracked_labels = tifffile.imread('tracked_labels.tif')

# Convert to lists
label_list = [tracked_labels[i] for i in range(tracked_labels.shape[0])]

# Generate cytoplasmic rings
cyto_rings = generate_cytoplasmic_ring(label_list, ring_width=2)

# Extract intensity features
intensity_df = extract_intensity_features(label_list, cyto_rings, original_images)

# Visualize
visualize_cn_ratio(intensity_df)
```

---

## Parameter Reference

### ImageAnalyzer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filepath` | - | Path to input TIFF image |
| `min_track_length` | 3 | Minimum number of frames for valid track |
| `max_linking_distance` | 15 | Maximum distance (pixels) for frame-to-frame linking |
| `max_gap_closing` | 2 | Maximum number of frames to bridge |
| `max_gap_distance` | 15 | Maximum distance (pixels) for gap closing |

### SimpleLAPTracker

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_linking_distance` | 15 | Maximum distance for linking between consecutive frames |
| `max_gap_closing` | 2 | Maximum number of frames for gap closing |
| `max_gap_distance` | 15 | Maximum distance for gap closing |
| `min_track_length` | 3 | Minimum track length to keep |

### segment_nuclei

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu` | True | Use GPU for Cellpose |
| `output_dir` | 'segmentation_nuc/' | Output directory |
| `flow_threshold` | 0.4 | Cellpose flow threshold |
| `cellprob_threshold` | 0.0 | Cellpose cell probability threshold |

### generate_cytoplasmic_rings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ring_width` | 2 | Width of cytoplasmic ring (pixels) |

---

## Output Files

| File/Directory | Description |
|----------------|-------------|
| `segmentation_nuc/` | Cellpose segmentation results |
| `tracked_labels/` | Tracked label images (TIFF) |
| `cyto.tif` | Cytoplasmic ring images |
| `cn_ratio_timelapse.gif` | Animated C/N ratio visualization |

---

## Tracking Algorithm

KTRtracker uses a two-step LAP (Linear Assignment Problem) approach:

1. **Frame-to-frame linking**: Links objects between consecutive frames using the Hungarian algorithm
2. **Gap closing**: Reconnects broken tracks by:
   - Finding track endpoints and start points
   - Computing cost matrix based on distance and frame gap
   - Optimal matching using Hungarian algorithm
   - Linear interpolation for missing frames (x, y, area, C/N ratio)

---

## Troubleshooting

### Segmentation Issues

- Adjust `flow_threshold` (lower = more permissive)
- Adjust `cellprob_threshold` (lower = detect more cells)
- Check image preprocessing (background subtraction, contrast)

### Tracking Issues

- Increase `max_linking_distance` if objects move fast
- Increase `max_gap_closing` if objects disappear for multiple frames
- Decrease `min_track_length` to keep shorter tracks

### Gap Closing Not Working

- Check if `max_gap_distance` is large enough
- Verify `max_gap_closing` covers the gap duration
- Inspect tracking results: `print(analyzer.tracking_df)`

---

## License

MIT License

## Author

Yuhei Goto, 2025
