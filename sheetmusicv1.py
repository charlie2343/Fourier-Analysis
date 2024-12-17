import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import stft, get_window
import math

from music21 import *
import os

# Normalize the signal
#signal = signal / max(abs(signal))

# Function to compute a reasonable window size based on BPM
def get_interval():
    # Convert BPM to approximate number of samples per beat
    beat_interval_seconds = (60 / BPM) / 8 
    #32nd note length
    sample_count = 6000
    # Ensure window size is appropriate (e.g., capped at 512 samples)
    # sample_count = min(sample_count,5000)
    print(f"sample count: {sample_count}")
    window = get_window("triang", sample_count)
    print("Window type: ", type(window), " Window values: ", window)
    return window

# Adjust STFT to scale frequency range to 0 - 1000 Hz
def short_time_fourier_transform(signal, overlap=0.5):
    window = get_interval()  # Get the window for the STFT
    nperseg = len(window)
    noverlap = int(overlap * nperseg)
    f, t, Zxx = stft(signal, sampling_rate, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, t, Zxx



def average(): 
    for time_idx, time in enumerate(times):
        # Skip first and last indices to avoid out-of-bounds error
        if time_idx == 0 or time_idx == len(times) - 1:
            continue
        
        # Amplitudes for the current, previous, and next time segments
        time_amplitudes1 = amplitudes[:, time_idx]
        time_amplitudes2 = amplitudes[:, time_idx - 1]
        time_amplitudes3 = amplitudes[:, time_idx + 1]
        
        # Calculate the average amplitudes
        avg = (time_amplitudes1 + time_amplitudes2 + time_amplitudes3) / 3
        return avg
    
def findnotes():    
    # Check for amplitudes above the threshold
    for i, amp in enumerate(avg):  # Loop through each frequency bin
        max = 80
        currnote = ""
        top_index = np.argsort(time_amplitudes)[-1:]
        if amp > 3000: 
            for note in oct4notes: 
                if abs(frequencies[i] - oct4notes[note]) <= max: 
                    max = abs(frequencies[i] - oct4notes[note])
                    currnote = note
            
            print(f"Time: {time:.2f}s")
            print(f"  Frequency: {frequencies[i]:.2f} Hz, Note {currnote} Amplitude: {amp:.2f}")
# def findnotes2():
#     for time_idx, time in enumerate(times):
#         time_amplitudes_cur = amplitudes[:, time_idx]
        
#         if time_idx == 0 or time_idx == len(times) - 1:
#             continue
        
#         time_amplitudes_nxt = amplitudes[:, time_idx + 1]
        
#         # Get indices of the top 2 amplitudes for the current and next time step. Argsort sorts indices not values
#         top_indices_cur = np.argsort(time_amplitudes_cur)[-2:] 
#         top_indices_nxt = np.argsort(time_amplitudes_nxt)[-2:]
        
#         print(f"Time: {time:.2f}s")
#         count = 0
#         frequency = 0
        
#         # Compare the top frequencies between current and next time step
#         if top_indices_cur[1] == top_indices_nxt[1]:  # Check for persistence of the highest peak
#             count += 1
#             frequency = frequencies[top_indices_cur[1]]
#         else:
#             print(f"  Frequency: {frequency} Hz gone, Amplitude: {time_amplitudes_cur[top_indices_cur[1]]:.2f}, Count: {count}")
#             count = 0

def getnotes():
    tolerance = 1.0  # Tolerance for frequency persistence in Hz
    frequency = 0
    count = 0
    
    for time_idx, time in enumerate(times):
        if time_idx == 0 or time_idx == len(times) - 1:
            continue

        time_amplitudes_cur = amplitudes[:, time_idx]
        time_amplitudes_nxt = amplitudes[:, time_idx + 1]

        top_indices_cur = np.argsort(time_amplitudes_cur)[-2:] 
        top_indices_nxt = np.argsort(time_amplitudes_nxt)[-2:]

        print(f"Time: {time:.2f}s")
        
        

        # Compare top frequencies based on indices and apply tolerance
        freq_cur = frequencies[top_indices_cur[1]]
        freq_nxt = frequencies[top_indices_nxt[1]]
        # print(f"Time: {time:.2f}s")
        # print(f"Top indices (current): {top_indices_cur}, Top indices (next): {top_indices_nxt}")
        print(f"\n Frequencies (current): {frequencies[top_indices_cur]}, Frequencies (next): {frequencies[top_indices_nxt]}")
        # print(f"Amplitudes (current): {time_amplitudes_cur[top_indices_cur]}"

        # !prints the top four amplitudes
        top_indices = np.argsort(time_amplitudes_cur)[-4:]  # Get indices of top 4 amplitudes
        for idx in reversed(top_indices):  # Reverse to show the largest first
            print(f"  Frequency: {frequencies[idx]:.2f} Hz, Amplitude: {time_amplitudes_cur[idx]:.2f}")
        
        # ! if in consecutively in top 2 frequencies count increase
        # ! If not, get duration of the frequency, compare it to closest note, and append to list  
        if freq_cur in frequencies[top_indices_nxt]:
            #np.abs(freq_cur - freq_nxt) < tolerance
            count += 1
            frequency = freq_cur
        elif count != 0:
            print(f" \n Frequency: {frequency:.2f} Hz gone, Amplitude: {time_amplitudes_cur[top_indices_cur[1]]:.2f}, Count: {count}")
            length = get_duration(count)
            note = comparenotes(frequency)
            score.append((frequency,note,length,time))
            count = 0
        else: 
            pass


def comparenotes(frequency): 
    max = 999
    currnote = ""
    for note in oct4notes: 
        if abs(frequency - oct4notes[note]) <= min: 
            min = abs(frequency - oct4notes[note])
            currnote = note
    return currnote
    
def get_duration(count): 
    min = 10
    closestnote = ""
    durations = {}
    time = count * (sample_count/48000)
    durations["whole"] = 60/BPM * 4
    durations["half"] = durations["whole"] /2 
    durations["quarter"] = durations["whole"]/4
    durations["eigth"] = durations["whole"]/8
    durations["sixteenth"] = durations["whole"]/16
    durations["thirty2nd"] = durations["whole"]/32
    
    for note in durations: 
        if abs(time-durations[note]) < min:
            closestnote = note
            min = abs(time-durations[note])
    return closestnote

def Spectrogram(stft): 
    # Plot Frequency Spectrum (STFT Spectrogram) with Log Scaling
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(np.abs(stft) + 1e-6), shading='gouraud', cmap='inferno')

    tick_positions = np.linspace(times[0], times[-1], num=24)  # More ticks

    plt.xticks(tick_positions, [f"{tick:.1f}" for tick in tick_positions])  # Custom labels
    plt.title(f"STFT Spectrogram, Sample count: {3000}, Precision: {3000/48000}ms", fontsize = 20)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Amplitude (dB)")
    plt.grid()
    plt.tight_layout()
    plt.show()
    

def makesheetmusic(score): 
    for data in score: 
        frequency = score[data][0]




BPM = 60
sample_count = 3000
oct4notes = { 
             "D": 293,
             "E": 329,
             "C": 261,
             "G": 392
             }

score = []
avg = []  # List to store average amplitudes    
# Load the WAV file

sampling_rate, signal = read('mhll.wav')

print("Sampling rate:", sampling_rate)

# Convert to mono if stereo
if len(signal.shape) > 1:
    signal = signal.mean(axis=1)

frequencies, times, stft_result = short_time_fourier_transform(signal)

# Filter frequency range to 0 - 1000 Hz
max_freq = 1000  # Define maximum frequency for display
freq_filter = frequencies <= max_freq  # Create filter
frequencies = frequencies[freq_filter]  # Apply filter to frequencies
stft_result = stft_result[freq_filter, :]  # Apply filter to STFT result
#print(frequencies)

# Filter: Remove frequencies with amplitude > threshold
amplitudes = stft_result  # Get amplitudes
print(amplitudes)
threshold = 0.15  # Define amplitude threshold
mask = amplitudes >= threshold  # Create mask
Zxx_filtered = stft_result * mask  # Zero out high-amplitude bins
# for i in range(0,len(amplitudes)): 
#     if amplitudes[i] >= threshold:
#         filtered = stft_result[i]
        
#print(filtered)
        # Calculate the amplitudes in linear scale
amplitudes = np.abs(stft_result)


getnotes()

print(Zxx_filtered)

print("Max amplitude (linear):", np.max(amplitudes))

#Spectrogram(stft_result)
#Spectrogram(Zxx_filtered)
#print(score)
findnotes()
print("Sample Count", sample_count) 

