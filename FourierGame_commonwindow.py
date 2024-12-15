import numpy as np
from scipy.signal import stft, istft, get_window
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from os import sys

# Add pydub's installation directory to sys.path
pydub_path = "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages"  # Replace with your actual path
if pydub_path not in sys.path:
    sys.path.append(pydub_path)

from pydub import AudioSegment
from pydub.playback import play

# Parameters
BPM = 60
LEVEL = 8  # Lower LEVEL keeps fewer frequencies, e.g., 10 means keeping top 10% of frequencies
SAMPLING_RATE, signal = read('furELISE.wav')

# Convert to mono if stereo
if len(signal.shape) > 1:
    signal = signal.mean(axis=1)

def get_interval():
    # Convert BPM to approximate number of samples per beat
    beat_interval_seconds = (60 / BPM) / 8  # 32nd note length
    sample_count = int(beat_interval_seconds * SAMPLING_RATE)
    # Ensure window size is appropriate (e.g., capped at 3000 samples)
    sample_count = min(sample_count, 3000)
    print(f"Sample count: {sample_count}")
    window = get_window("triang", sample_count)
    return window

def Spectrogram(times, frequencies, Zxx): 
    # Plot Frequency Spectrum (STFT Spectrogram) with Log Scaling
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(np.abs(Zxx) + 1e-6), shading='gouraud', cmap='inferno')
    plt.title("STFT Spectrogram (Log-Scaled Amplitude)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Amplitude (dB)")
    plt.tight_layout()
    plt.show()

# Perform STFT
window = get_interval()
nperseg = len(window)
noverlap = int(0.5 * nperseg)
frequencies, times, Zxx = stft(signal, fs=SAMPLING_RATE, window=window, nperseg=nperseg, noverlap=noverlap)

# Find common significant frequencies across all windows
magnitude = np.abs(Zxx)
threshold = np.percentile(magnitude, 100 - (100 / LEVEL))
print(f"Threshold for significant frequencies: {threshold}")

significant_frequencies = np.sum(magnitude > threshold, axis=1) == len(times)
if not significant_frequencies.any():
    print("No significant frequencies found with current LEVEL. Lowering threshold.")
    significant_frequencies = np.sum(magnitude > threshold / 2, axis=1) == len(times)

# Create a mask to retain only the common significant frequencies
mask = np.zeros_like(Zxx, dtype=bool)
mask[significant_frequencies, :] = True

# Apply the mask to the STFT
modified_stft = np.zeros_like(Zxx, dtype=complex)
modified_stft[mask] = Zxx[mask]

# Debugging output for modified STFT
if np.all(modified_stft == 0):
    print("Modified STFT is completely zero. Check thresholding logic.")

# Perform inverse STFT for reconstruction
_, reconstructed_signal = istft(modified_stft, fs=SAMPLING_RATE, window=window, nperseg=nperseg, noverlap=noverlap)

# Normalize the reconstructed signal to range [-1, 1]
if np.max(np.abs(reconstructed_signal)) > 0:
    reconstructed_signal /= np.max(np.abs(reconstructed_signal))
else:
    print("Reconstructed signal is zero. Check mask and STFT logic.")

# Convert to int16 for playback
signal_int16 = np.int16(reconstructed_signal * 32767)
raw_audio = signal_int16.tobytes()

# Create an AudioSegment for playback
audio_segment = AudioSegment(
    data=raw_audio,
    sample_width=2,  # 2 bytes (16-bit PCM)
    frame_rate=SAMPLING_RATE,  # Sampling rate
    channels=1  # Mono audio
)

# Play the reconstructed audio
play(audio_segment)
