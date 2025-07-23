import numpy as np

def generate_wave(time, period, wave_type='sine', amp=1): # This is for generation of waves

    if wave_type == 'sine':
        return amp * np.sin(2 * np.pi * time / period)
    elif wave_type == 'square':
        return amp * np.sign(np.sin(2 * np.pi * time / period))
    elif wave_type == 'triangle':
        return amp * (2 * np.abs(2 * ((time / period) - np.floor(0.5 + (time / period)))) - 1)
    else:
        raise ValueError("Invalid") # Error checking
    
def convolve_1d(signal1, signal2):
    len1 = len(signal1)
    len2 = len(signal2)
    # The length of the output signal is len1 + len2 - 1
    output_length = len1 + len2 - 1

    result = np.zeros(output_length) #  zeros
    # Perform the convolution
    for i in range(output_length):
        for j in range(len1):
            if i - j >= 0 and i - j < len2: # Not so sure why its giving a warning?
                result[i] += signal1[j] * signal2[i - j]
    return result

def calculate_fourier_coefficients(sig, time, period, n_harmonics):
    coefficients = []
    omega = 2 * np.pi / period
    dt = time[1] - time[0]  # Time step
    N = len(time)

    # Calculate a0
    a0 = (2.0 / N) * np.sum(sig)
    coefficients.append((a0, 0))  # b0 is always 0

    for n in range(1, n_harmonics + 1):
        an = (2.0 / N) * np.sum(sig * np.cos(n * omega * time))
        bn = (2.0 / N) * np.sum(sig * np.sin(n * omega * time))
        coefficients.append((an, bn))
    return (coefficients)




def approximate_fourier_series(coefficients, n_harmonics, time, period):
    omega = 2.0 * np.pi / period
    approximation = np.zeros_like(time)

    a0 = coefficients[0][0]
    approximation += a0

    for n in range(1, min(n_harmonics + 1, len(coefficients))):
        an, bn = coefficients[n]
        approximation += an * np.cos(n * omega * time) + bn * np.sin(n * omega * time)
    approximation = np.round(approximation, decimals=3)
    return (approximation)

def calculate_discrete_fourier_transform(signal, N):

    # 1. What is the DFT and what is its purpose?
    """
    The Discrete Fourier Transform (DFT) is a mathematical tool used to convert a
    signal from the time domain into the frequency domain, enabling analysis of the
    different frequency components present in a discrete-time signal.
    """


    # 2. How does the DFT transform a discrete time-domain signal into the discrete frequency domain?
    """
   The DFT achieves this transformation by taking a weighted sum 
    of the signal’s samples, where each sample is multiplied by a complex exponential.
    This process isolates individual frequency components within the signal.
    """



    # 3. What is the mathematical formula for calculating the DFT of a signal?
    """
    The DFT is calculated using the formula: X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j * pi * k * n / N) 
    where each X[k] represents a frequency component of the signal.
    """

    # 4. Discuss any limitations or challenges associated with using the DFT, especially when dealing with large datasets.
    """
  The DFT has a computational complexity of O(N^2), meaning it requires significant
  processing power for large datasets. Additionally, it can introduce artifacts or aliasing issues
  when applied to non-periodic signals.
    """

    # 5.What is the Fast Fourier Transform (FFT) and how does it differ from the DFT?
    """
   The Fast Fourier Transform (FFT) is an optimized algorithm for calculating the DFT. By leveraging a
   divide-and-conquer approach, the FFT dramatically reduces the number of calculations needed,
   making it much faster than directly computing the DFT.
    """

    # 6. Explain how the FFT achieves a significant reduction in computational complexity.
    """
    The FFT lowers the computational complexity from O(N^2) in the DFT to O(NlogN). 
    This efficiency improvement enables practical use of Fourier analysis on large datasets by minimizing
    the required computational resources.
    """

    # Pad signal as needed
    sig_padded = np.pad(signal, (0, max(0, N - len(signal))), 'constant')
    dft_result = np.zeros(N, dtype=complex)


    for k in range(N): # For DFT
        for n in range(N):
            dft_result[k] += sig_padded[n] * np.exp(-2j * np.pi * k * n / N)
    return (dft_result)