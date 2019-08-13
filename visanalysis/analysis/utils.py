import numpy as np

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    From: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def getLinearFilterByFFT(stimulus, response, filter_len):
    nans, x= nan_helper(response)
    response[nans]= np.interp(x(nans), x(~nans), response[~nans]) # if response has nans, fill in with linear interpolation

    filter_fft = np.fft.fft(response) * np.conj(np.fft.fft(stimulus))
    filt = np.real(np.fft.ifft(filter_fft))[0:filter_len]
    return filt
