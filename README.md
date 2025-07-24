# Video Motion Magnification

A Python implementation of Gaussian motion magnification for amplifying subtle motions in videos that are invisible to the naked eye.

## Table of Contents
- [Overview](#overview)
- [Motion Magnification Basics](#motion-magnification-basics)
- [Use Cases](#use-cases)
- [Method: Gaussian Motion Magnification](#method-gaussian-motion-magnification)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Output Plots](#output-plots)

## Overview

This project implements video motion magnification, a technique that reveals temporal variations in videos that are difficult or impossible to see with the naked eye. By amplifying small motions, we can visualize phenomena like:
- Heartbeat and pulse detection
- Breathing patterns
- Structural vibrations
- Micro-movements in materials

## Motion Magnification Basics

### What is Motion Magnification?

Motion magnification is a computational technique that amplifies small temporal changes in videos. It works by:

1. **Decomposing** the video into spatial frequency bands
2. **Analyzing** temporal variations in each band
3. **Amplifying** specific frequency ranges
4. **Reconstructing** the video with enhanced motion

### Key Concepts

- **Phase-based Processing**: Uses phase information from Fourier transforms to detect motion
- **Spatial Decomposition**: Applies Gaussian windows to analyze different spatial regions
- **Temporal Analysis**: Tracks phase changes over time to identify motion patterns
- **Selective Amplification**: Magnifies only desired frequency ranges while preserving video quality

## Use Cases

### Medical Applications
- **Pulse Detection**: Visualize blood flow and heart rate from facial videos
- **Breathing Monitoring**: Amplify chest movements for respiratory analysis
- **Non-contact Vital Signs**: Remote monitoring without physical sensors

### Engineering & Industrial
- **Structural Health Monitoring**: Detect micro-vibrations in buildings and bridges
- **Quality Control**: Identify subtle defects in manufacturing processes
- **Mechanical Analysis**: Visualize component vibrations and wear patterns

### Scientific Research
- **Material Science**: Study micro-deformations and stress patterns
- **Biology**: Observe cellular movements and biological processes
- **Physics**: Analyze wave propagation and oscillatory phenomena

## Method: Gaussian Motion Magnification

Our implementation uses **Gaussian Motion Magnification**, which offers several advantages:

### Algorithm Steps

1. **ROI Selection**: User selects region of interest for analysis
2. **Spatial Windowing**: Apply Gaussian windows across the ROI
3. **Frequency Analysis**: Compute 2D FFT for each windowed region
4. **Phase Extraction**: Extract phase information from frequency domain
5. **Temporal Tracking**: Monitor phase changes across video frames
6. **Motion Amplification**: Multiply phase shifts by magnification factor
7. **Reconstruction**: Apply inverse FFT to generate magnified frames

### Key Parameters

- **Magnification Factor**: Controls amplification strength (default: 10)
- **Sigma (σ)**: Gaussian window size for spatial analysis (default: 50)
- **Alpha (α)**: Temporal smoothing factor (default: 0.5)

## Features

- ✅ **Interactive ROI Selection**: Choose specific areas for analysis
- ✅ **Real-time Visualization**: Live display of magnified motion with arrows
- ✅ **Comprehensive Analysis**: Multiple visualization plots
- ✅ **Phase Unwrapping**: Handles phase discontinuities
- ✅ **Frequency Domain Analysis**: Temporal frequency analysis of motion
- ✅ **Motion Vector Display**: Visual arrows showing motion direction

## Installation

1. Clone or download the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.7+
- OpenCV 4.8+
- NumPy 1.24+
- Matplotlib 3.7+

## Usage

### Basic Usage
```python
python final.py
```

### Workflow
1. **Video Loading**: Place your video file in the project directory
2. **ROI Selection**: Click and drag to select the region of interest
3. **Processing**: Algorithm automatically processes the selected region
4. **Visualization**: View real-time magnified motion with directional arrows
5. **Analysis**: Review generated plots for detailed motion analysis

### Customization
```python
# Modify parameters in final.py
magnification_factor = 15  # Increase for stronger amplification
sigma = 30                 # Smaller sigma for finer spatial analysis
alpha = 0.3               # Lower alpha for more temporal smoothing
```

## Technical Details

### Spatial Frequency Analysis
- **2D FFT**: Converts spatial image data to frequency domain
- **Gaussian Windowing**: Localizes analysis to specific spatial regions
- **Phase Extraction**: Uses `numpy.angle()` to get phase information

### Temporal Processing
- **Phase Tracking**: Monitors phase changes across video frames
- **Phase Unwrapping**: Removes 2π discontinuities using `numpy.unwrap()`
- **Temporal Smoothing**: Exponential averaging with alpha parameter

### Motion Reconstruction
- **Phase Magnification**: Multiplies phase shifts by amplification factor
- **Complex Reconstruction**: Applies magnified phase to original magnitude
- **Inverse FFT**: Converts back to spatial domain using `ifft2()`

## Output Plots

The system generates several analytical plots:

### 1. Phase Shift Comparison (Time Domain)
- Shows original vs magnified phase shifts over time
- Demonstrates temporal progression of motion

### 2. Frequency Domain Analysis
- Reveals dominant oscillation frequencies in the motion
- Helps identify periodic patterns (heartbeat, vibration modes)

### 3. Window Phase Comparison
- Direct comparison of window-level phase shifts
- Visualizes magnification effect at granular level

### 4. Pixel Difference Analysis
- Tracks pixel-level changes between consecutive frames
- Provides motion intensity measurements

### 5. Motion Vector Visualization
- Real-time arrows showing motion direction and magnitude
- Overlaid on magnified video for intuitive understanding

## File Structure
```
video_motion/
├── final.py              # Main implementation
├── requirements.txt      # Dependencies
├── README.md            # This file
├── *.mp4               # Input videos
└── output/             # Generated plots and results
    ├── phase_shift_plot.png
    ├── phase_shift_frequency_domain.png
    ├── window_phase_comparison.png
    └── pixel_difference_plot.png
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional magnification algorithms
- Real-time processing optimization
- Enhanced visualization features
- Multi-ROI analysis support

## References

- Wu, H. Y., et al. "Eulerian video magnification for revealing subtle changes in the world." ACM transactions on graphics (TOG) 31.4 (2012): 65.
- Wadhwa, N., et al. "Phase-based video motion processing." ACM Transactions on Graphics (TOG) 32.4 (2013): 80.
