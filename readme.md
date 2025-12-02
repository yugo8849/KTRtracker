
A comprehensive Python package for image analysis, segmentation, and tracking.

## Features

- TIFF image loading and preprocessing
- Nuclear segmentation using Cellpose
- Object tracking with LAP tracker
- Intensity feature extraction
- Visualization of tracking and intensity ratios

## Installation
```bash
pip install git+https://github.com/yugo8849/KTRtracker.git
```

## Quick Example
```python
from KTRtracker import ImageAnalyzer

# Complete workflow
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

You can also use SimpleLAPTracker class directly for tracking.
#Tracking with LAP tracker using label image
tracker = SimpleLAPTracker(max_linking_distance=20,min_track_length=10)
results = tracker.track(labels)
filtered = tracker.filter_tracks(results)

#Save results with csv and labeled image
filtered.to_csv('fileneame.csv',index=False)
track_label = save_tracking_as_labels(labels,filtered)
```


## License

MIT License
```

4. `LICENSE`:
```
MIT License

Copyright (c) 2024 Yuhei Goto

Permission is hereby granted, free of charge...
(standard MIT license text)
```

5. `requirements.txt`:
```
numpy
pandas
matplotlib
seaborn
tifffile
scikit-image
cellpose
napari  # optional

