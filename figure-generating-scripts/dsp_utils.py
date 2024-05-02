import numpy as np
import scipy.signal as signal

def add_awgn_noise(s, SNR_dB, L = 1):
    """A function that adds additive causian white noise to a given (complex) time signal
    Based on: https://www.gaussianwaves.com/2015/06/how-to-generate-awgn-noise-in-matlaboctave-without-using-in-built-awgn-function/
    """

    P = L * np.sum(abs(s)**2)/len(s)
    SNR = 10**(SNR_dB/10)
    N0 = P / SNR # Not entirely sure why we are not accounting for bandwidth here, SNR = Pr/(N0 * B) = Eb/N0 * Rb/B

    if np.isrealobj(s):
        n = np.sqrt(N0/2) * np.random.default_rng().standard_normal(s.shape)
    else:
        n = np.sqrt(N0/2) * ( 1 + 1j ) * np.random.default_rng().standard_normal(s.shape)
    
    r = s + n

    return r

def muller_muller(samples, sps = 8):
    samples_interpolated = signal.resample_poly(samples, 16, 1)
    mu = 0 # initial estimate of phase of sample
    out = np.zeros(len(samples) + 10, dtype=np.complex128)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex128) # stores values, each iteration we need the previous 2 values plus current value
    i_in = 0 # input samples index
    i_out = 2 # output index (let first two outputs be 0)
    while i_out < len(samples) and i_in < len(samples):
        out[i_out] = samples_interpolated[i_in*16 + int(mu*16)]
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
        x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        mm_val = np.real(y - x)
        mu += sps + 0.3*mm_val
        i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
        mu = mu - np.floor(mu) # remove the integer part of mu
        i_out += 1 # increment output index
    out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
    return out

def Q(x):
    from scipy import special
    return 0.5 * special.erfc(x / np.sqrt(2))