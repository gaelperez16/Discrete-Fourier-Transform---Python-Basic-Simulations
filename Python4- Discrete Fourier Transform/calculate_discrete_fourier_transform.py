import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_discrete_fourier_transform

# Constants
T = 2  # duration secs
f = 5  # Frequency hz
Fs1 = 5 * f  # Sampling rate for first signal
Fs2 = (3 / 2) * f  # Sampling rate for  second signal

time1 = np.arange(0, T, 1 / Fs1)  # Time vector for the first signal
time2 = np.arange(0, T, 1 / Fs2)  # Time vector for the second signal

# Generate sinusoidal signals
signal1 = np.sin(2 * np.pi * f * time1)  # First signal
signal2 = np.sin(2 * np.pi * f * time2)  # Second signal

# number of samples
N1 = len(signal1)

# Compute the N-point DFT of the first signal
dft_signal1_N = calculate_discrete_fourier_transform(signal1, N1)
# Compute the N-point DFT of the first signal
N2 = 5 * N1
dft_signal1_N2 = calculate_discrete_fourier_transform(signal1, N2)

# Compute the DFT of the second signal
dft_signal2_N2 = calculate_discrete_fourier_transform(signal2, N2)

# Compare with numpy's FFT
fft_signal1_N = np.fft.fft(signal1, N1)
fft_signal1_N2 = np.fft.fft(signal1, N2)
fft_signal2_N2 = np.fft.fft(signal2, N2)

# Frequency axes for plotting
freq1_N = np.fft.fftfreq(N1, d=1/Fs1)
freq2_N2 = np.fft.fftfreq(N2, d=1/Fs2)
# Apply fft shift to align the frequency arrays and DFTs for zero-centered frequency
#This eliminates the unwated horizontal line in previous graphs
dft_signal1_N = np.fft.fftshift(dft_signal1_N)
dft_signal1_N2 = np.fft.fftshift(dft_signal1_N2)
dft_signal2_N2 = np.fft.fftshift(dft_signal2_N2)
fft_signal1_N = np.fft.fftshift(fft_signal1_N)
fft_signal1_N2 = np.fft.fftshift(fft_signal1_N2)
fft_signal2_N2 = np.fft.fftshift(fft_signal2_N2)

freq1_N = np.fft.fftshift(freq1_N)
freq2_N2 = np.fft.fftshift(freq2_N2)
# Plot
plt.figure(figsize=(12, 8))
# DFT of the first signal (N1)
plt.subplot(3, 1, 1)
plt.plot(freq1_N, np.abs(dft_signal1_N), label='utils.py', color='black')
plt.plot(freq1_N, np.abs(fft_signal1_N), label='NumPy (signal1)', linestyle='dashed', color='orange')
plt.title('DFT of First Signal ')
plt.xlabel('Frequency (Hz)')
plt.ylabel('magnitude')
plt.legend()
# DFT first signal N = 5 * N1
plt.subplot(3, 1, 2)
plt.plot(freq2_N2, np.abs(dft_signal1_N2), label='utils.py', color='black')
plt.plot(freq2_N2, np.abs(fft_signal1_N2), label='NumPy (signal1, N=5*N)', linestyle='dashed', color='orange')
plt.title('DFT of First Signal (N = 5 length of signal)')
plt.xlabel('Frequency in (Hz)')
plt.ylabel('magnitude')
plt.legend()
# DFT second signal
plt.subplot(3, 1, 3)
plt.plot(freq2_N2, np.abs(dft_signal2_N2), label='utils.py', color='black')
plt.plot(freq2_N2, np.abs(fft_signal2_N2), label='NumPy (signal2, N=5*N])', linestyle='dashed', color='orange')
plt.title('DFT of Second Signal ')
plt.xlabel('Frequency in (Hz)')
plt.ylabel('magnitude')
plt.legend()
plt.tight_layout()
plt.show()
plt.grid()