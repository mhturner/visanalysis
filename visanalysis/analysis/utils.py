import numpy as np

def getLinearFilterByFFT(stimulus, response, filter_len):
    filter_fft = np.fft.fft(response) * np.conj(np.fft.fft(stimulus))
    filt = np.real(np.fft.ifft(filter_fft))[0:filter_len]
    return filt
