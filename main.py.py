import cv2
import numpy as np
from numpy.fft import fft2, ifft2
from numpy import angle, exp, pi
import matplotlib.pyplot as plt

def plot_phase_shifts(original_phase_shifts, magnified_phase_shifts):
    """
    Plots the original and magnified phase shifts for the selected ROI with phase unwrapping.
    
    This plot shows:
    - TIME DOMAIN: Phase shift values across video frames (temporal progression)
    - Shows HOW MUCH the phase changes from frame to frame
    - Y-axis: Phase shift magnitude (radians)
    - X-axis: Frame number (time progression)
    """
    # Apply phase unwrapping to remove discontinuities
    original_unwrapped = np.unwrap(original_phase_shifts)
    magnified_unwrapped = np.unwrap(magnified_phase_shifts)
    
    plt.figure(figsize=(10, 6))
    plt.plot(original_unwrapped, label="Original Phase Shift (Unwrapped)", color="blue")
    plt.plot(magnified_unwrapped, label="Magnified Phase Shift (Unwrapped)", color="red", alpha=0.6)
    plt.xlabel("Frame Index")
    plt.ylabel("Phase Shift (Mean)")
    plt.title("Original vs Magnified Phase Shift (Unwrapped)")
    plt.legend()
    plt.grid(True)
    plt.savefig("phase_shift_plot.png")
    plt.show()

def draw_arrow_on_frame(frame, dx, dy, color=(0, 0, 255), thickness=2, scale=10):
    """
    Draws an arrow at the center of the frame with direction (dx, dy) and length proportional to magnitude.
    """
    h, w = frame.shape[:2]
    start_point = (w // 2, h // 2)
    end_point = (int(start_point[0] + dx * scale), int(start_point[1] + dy * scale))
    cv2.arrowedLine(frame, start_point, end_point, color, thickness, tipLength=0.3)
    return frame

def compute_mean_motion(prev_frame, curr_frame):
    """
    Computes the mean motion vector (dx, dy) between two frames using optical flow.
    """
    prev_gray = cv2.cvtColor((prev_frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor((curr_frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        pyr_scale=0.5, levels=1, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    dx = np.mean(flow[..., 0])
    dy = np.mean(flow[..., 1])
    return dx, dy

def gaussian_motion_magnification(video_path, magnification_factor=10, sigma=50, alpha=0.5):
    """
    Applies Gaussian motion magnification to the selected ROI in the video.
    Returns the processed data for additional plots.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None, None, None

    # Get video properties
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame for ROI selection.")
        cap.release()
        return None, None, None

    # Allow user to select ROI
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    x_start, y_start, roi_width, roi_height = map(int, roi)

    # Read all frames
    roi_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi_frame = frame[y_start:y_start + roi_height, x_start:x_start + roi_width]
        roi_frames.append(roi_frame / 255.0)

    cap.release()
    roi_frames = np.array(roi_frames)
    num_frames, height, width, _ = roi_frames.shape

    # Meshgrid for Gaussian window
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    # Precompute Gaussian masks
    gaussian_masks = {}
    x_range = list(range(0, width, 2 * sigma))
    y_range = list(range(0, height, 2 * sigma))
    for y in y_range:
        for x in x_range:
            gaussian_masks[(x, y)] = np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))

    # Initialize magnified frames
    magnified_frames = np.zeros_like(roi_frames)

    original_phase_shifts = []
    magnified_phase_shifts = []
    window_phase_shifts = []  # Store window-level phase shifts
    window_magnified_phases = []  # Store window-level magnified phases

    # Process each window
    for (x, y), gaussian_mask in gaussian_masks.items():
        for channel_index in range(3):
            window_average_phase = None
            for frame_index in range(num_frames):
                windowed_frame = gaussian_mask * roi_frames[frame_index, :, :, channel_index]
                # FIRST FFT: 2D FFT on spatial image data (converts image to frequency domain)
                window_dft = fft2(windowed_frame)  # This analyzes spatial frequencies in the image

                if window_average_phase is None:
                    window_average_phase = angle(window_dft)

                window_phase_shift = angle(window_dft) - window_average_phase
                window_phase_shift = np.mod(window_phase_shift + pi, 2 * pi) - pi
                window_magnified_phase = magnification_factor * window_phase_shift
                window_magnified_dft = window_dft * exp(window_magnified_phase * 1j)
                window_magnified = abs(ifft2(window_magnified_dft))

                window_phase_unwrapped = window_average_phase + window_phase_shift
                window_average_phase = alpha * window_average_phase + (1 - alpha) * window_phase_unwrapped

                magnified_frames[frame_index, :, :, channel_index] += window_magnified

                if channel_index == 0:  # For the first channel, store the phase shifts
                    original_phase_shift = angle(window_dft)
                    magnified_phase_shift = original_phase_shift * magnification_factor

                    # Store mean phase shifts for graph plotting (avoid wrapping here)
                    original_phase_shifts.append(np.mean(original_phase_shift))
                    magnified_phase_shifts.append(np.mean(magnified_phase_shift))
                    
                    # Store window-level phase shifts for detailed comparison
                    window_phase_shifts.append(np.mean(window_phase_shift))
                    window_magnified_phases.append(np.mean(window_magnified_phase))

    # Normalize magnified frames
    magnified_frames = np.clip(magnified_frames / np.max(magnified_frames), 0, 1)

    # Display results
    cap = cv2.VideoCapture(video_path)
    for idx, magnified_roi in enumerate(magnified_frames):
        ret, frame = cap.read()
        if not ret:
            break

        magnified_roi_uint8 = (magnified_roi * 255).astype(np.uint8)
        magnified_frame = frame.copy()
        magnified_frame[y_start:y_start + roi_height, x_start:x_start + roi_width] = magnified_roi_uint8

        # Draw motion arrow on magnified ROI
        if idx > 0:
            prev_roi = magnified_frames[idx - 1]
            curr_roi = magnified_frames[idx]
            dx, dy = compute_mean_motion(prev_roi, curr_roi)
            magnified_roi_with_arrow = draw_arrow_on_frame(magnified_roi_uint8.copy(), dx, dy)
            magnified_frame[y_start:y_start + roi_height, x_start:x_start + roi_width] = magnified_roi_with_arrow

        # Display magnified ROI with arrow
        cv2.imshow('Magnified ROI with Motion Arrow', magnified_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot the phase shifts after processing all frames
    plot_phase_shifts(original_phase_shifts, magnified_phase_shifts)

    return original_phase_shifts, magnified_phase_shifts, roi_frames, window_phase_shifts, window_magnified_phases

def plot_phase_and_time_domain(original_phase_shifts, magnified_phase_shifts):
    """
    Plots the phase shift in the frequency domain and its inverse FFT in the time domain.
    
    IMPORTANT: This FFT is different from the image processing FFT above!
    - Image FFT (fft2): Analyzes spatial frequencies within each frame
    - This FFT (fft): Analyzes temporal frequencies of phase changes over time
    """
    # Apply phase unwrapping before FFT
    original_unwrapped = np.unwrap(original_phase_shifts)
    magnified_unwrapped = np.unwrap(magnified_phase_shifts)
    
    # SECOND FFT: 1D FFT on temporal phase shift data (converts time series to frequency domain)
    # This analyzes how fast the phase shifts oscillate over time (temporal frequencies)
    original_fft = np.fft.fft(original_unwrapped)    # Temporal frequency analysis
    magnified_fft = np.fft.fft(magnified_unwrapped)  # Temporal frequency analysis

    freq = np.fft.fftfreq(len(original_unwrapped))

    # Phase plot in frequency domain
    # This shows WHICH FREQUENCIES are dominant in your phase shift signal
    plt.figure(figsize=(10, 4))
    plt.plot(freq, np.abs(original_fft), label="Original Phase Shift (Freq Domain)", color="blue")
    plt.plot(freq, np.abs(magnified_fft), label="Magnified Phase Shift (Freq Domain)", color="red", alpha=0.6)
    plt.xlabel("Frequency (cycles per frame)")
    plt.ylabel("Magnitude")
    plt.title("Phase Shift in Frequency Domain (Shows oscillation rates)")
    plt.legend()
    plt.grid(True)
    plt.savefig("phase_shift_frequency_domain.png")
    plt.show()

    # Inverse FFT to get back to time domain
    original_ifft = np.fft.ifft(original_fft)
    magnified_ifft = np.fft.ifft(magnified_fft)

    # Time plot (inverse of phase)
    plt.figure(figsize=(10, 4))
    plt.plot(np.real(original_ifft), label="Original Phase Shift (Time Domain from IFFT)", color="blue")
    plt.plot(np.real(magnified_ifft), label="Magnified Phase Shift (Time Domain from IFFT)", color="red", alpha=0.6)
    plt.xlabel("Frame Index")
    plt.ylabel("Phase Shift (IFFT)")
    plt.title("Phase Shift (IFFT of Frequency Domain) in Time Domain")
    plt.legend()
    plt.grid(True)
    plt.savefig("phase_shift_ifft_time_domain.png")
    plt.show()

def plot_pixel_difference(roi_frames):
    """
    Plots the pixel difference value between consecutive frames of the selected ROI.
    """
    pixel_differences = []
    for i in range(1, len(roi_frames)):
        diff = np.abs(roi_frames[i] - roi_frames[i - 1])
        pixel_differences.append(np.mean(diff))

    plt.figure(figsize=(10, 4))
    plt.plot(pixel_differences, color="green", label="Pixel Difference Between Consecutive Frames")
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Pixel Difference")
    plt.title("Pixel Difference in Consecutive ROI Frames")
    plt.legend()
    plt.grid(True)
    plt.savefig("pixel_difference_plot.png")
    plt.show()

def plot_window_phase_comparison(window_phase_shifts, window_magnified_phases):
    """
    Plots the window-level phase shifts vs magnified phase shifts to show magnification effect.
    
    This plot shows:
    - Direct comparison of original window phase shifts vs magnified versions
    - Demonstrates the amplification effect of the magnification factor
    - Shows how small phase changes are enhanced for visibility
    """
    # Apply phase unwrapping for better visualization
    window_unwrapped = np.unwrap(window_phase_shifts)
    magnified_unwrapped = np.unwrap(window_magnified_phases)
    
    plt.figure(figsize=(12, 6))
    plt.plot(window_unwrapped, label="Original Window Phase Shift", color="blue", linewidth=2)
    plt.plot(magnified_unwrapped, label="Magnified Window Phase Shift", color="red", alpha=0.7, linewidth=2)
    plt.xlabel("Sample Index (Window × Frame)")
    plt.ylabel("Phase Shift (radians)")
    plt.title("Window Phase Shift vs Magnified Phase Shift (Shows Magnification Effect)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("window_phase_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create a scatter plot to show the relationship
    plt.figure(figsize=(8, 8))
    plt.scatter(window_unwrapped, magnified_unwrapped, alpha=0.6, s=10, color='purple')
    plt.xlabel("Original Window Phase Shift (radians)")
    plt.ylabel("Magnified Window Phase Shift (radians)")
    plt.title("Correlation: Original vs Magnified Window Phase Shifts")
    plt.grid(True, alpha=0.3)
    
    # Add diagonal line to show magnification factor
    min_val = min(np.min(window_unwrapped), np.min(magnified_unwrapped))
    max_val = max(np.max(window_unwrapped), np.max(magnified_unwrapped))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='1:1 ratio')
    plt.legend()
    plt.savefig("window_phase_correlation.png", dpi=300, bbox_inches='tight')
    plt.show()

# Updated main function to include new plots
if __name__ == "__main__":
    video_path = "lars.mp4"
    original_phase_shifts, magnified_phase_shifts, roi_frames, window_phase_shifts, window_magnified_phases = gaussian_motion_magnification(video_path)

    if original_phase_shifts is not None and magnified_phase_shifts is not None and roi_frames is not None:
        # Generate additional plots
        plot_phase_and_time_domain(original_phase_shifts, magnified_phase_shifts)
        plot_pixel_difference(roi_frames)
        plot_window_phase_comparison(window_phase_shifts, window_magnified_phases)

    cv2.destroyAllWindows()


