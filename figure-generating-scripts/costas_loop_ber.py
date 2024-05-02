#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from dsp_utils import Q, add_awgn_noise, muller_muller

def diff_encode(bits):
    out = np.zeros(len(bits) + 1,dtype=int)
    for i in range(len(bits)):
        out[i + 1] = (bits[i] + out[i]) % 2
    return out[1:]

def diff_decode(bits):
    bits_pre = np.concatenate((np.array([0]), bits))
    return (bits_pre[1:] + bits_pre[:-1]) % 2

def run_sequence(EbNo_dB, num_symbols = 2502, sps = 8, 
                 frac_delay = False, 
                 freq_shift = False, 
                 mult_fnt = lambda x, y: x*y,
                 awgn = True,
                 mm_before = False,
                 mm_after = False,
                 costas = True,
                 shape = True):

    bits_non_diff = np.random.randint(0, 2, num_symbols) # Our data to be transmitted, 1's and 0's
    bits = diff_encode(bits_non_diff)

    # Simulate bits
    x = np.array([])
    for bit in bits:
        pulse = np.zeros(sps)
        pulse[0] = bit*2-1 # set the first value to either a 1 or -1
        x = np.concatenate((x, pulse)) # add the 8 samples to the signal

    # Create our raised-cosine filter
    num_taps = 101
    beta = 0.35
    Ts = sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
    t = np.arange(-51, 52) # remember it's not inclusive of final number
    h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)

    # Filter our signal, in order to apply the pulse shaping
    if shape:
        samples = np.convolve(x, h)
    else:
        samples = np.convolve(x, np.ones(sps))
    
    if awgn:
        # Add noise
        Eb = 1/2 # For NRZ pulse
        EbNo = 10**(EbNo_dB/10)
        No = (Eb / EbNo)
        mu, sigma = 0, np.sqrt(No) # mean and standard deviation
        s = np.random.normal(mu, sigma, len(samples))
        samples = samples + s
        #samples = add_awgn_noise(samples, EbNo_dB, sps)
    # fig, ax = plt.subplots()
    # ax.plot(samples[0:16*8])
    # fig.show()

    # Create and apply fractional delay filter
    delay = 0.4 # fractional delay, in samples
    N = 21 # number of taps
    n = np.arange(-N//2, N//2) # ...-3,-2,-1,0,1,2,3...
    h = np.sinc(n - delay) # calc filter taps
    h *= np.hamming(N) # window the filter to make sure it decays to 0 on both sides
    h /= np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
    if frac_delay:
        samples = np.convolve(samples, h) # apply filter

    # apply a freq offset
    fs = 1e6 # assume our sample rate is 1 MHz
    fo = 500 # simulate freq offset
    Ts = 1/fs # calc sample period
    t = np.arange(0, Ts*len(samples), Ts) # create time vector
    #samples = samples * np.exp(1j*2*np.pi*fo*t) # perform freq shift

    if mm_before:
        samples = muller_muller(samples, sps)

    # Costas loop
    if costas:
        N = len(samples)
        phase = 0
        freq = 0
        # These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
        alpha = 0.005
        beta = 0.001
        out = np.zeros(N, dtype=np.complex128)
        freq_log = []
        ii = 0
        for i in range(N):
            out[i] = samples[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
            #error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
            #error_xnor = ~((np.real(out[i]) > 0) ^ (np.imag(out[i]) > 0)) * 2 - 1
            #error = error_xnor
            error = mult_fnt(np.real(out[i]), np.imag(out[i]))

            # Advance the loop (recalc phase and freq offset)
            freq += (beta * error)
            freq_log.append(freq * fs / (2*np.pi) / 8) # convert from angular velocity to Hz for logging
            phase += freq + (alpha * error)

            # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
            while phase >= 2*np.pi:
                phase -= 2*np.pi
            while phase < 0:
                phase += 2*np.pi
            ii += 1
        
            samples = out

    # fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 7))
    # fig.subplots_adjust(hspace=0.4)
    # ax1.plot(freq_log, '.-')
    # ax1.set_xlabel('Sample')
    # ax1.set_ylabel('Freq Offset [Hz]')
    # ax2.plot(np.real(out[(i-20):i]), np.imag(out[(i-20):i]), '.')
    # ax2.axis([-2, 2, -0.8, 0.8])
    # ax2.set_ylabel('Q')
    # ax2.set_xlabel('I')
    # plt.show()
    
    if mm_after:
        samples = muller_muller(samples, sps)

    if not mm_before and not mm_after:
        samples = samples[int(sps/2)::sps]

    bits_out = (samples > 0) * 1 # Slice the samples
    bits_out_non_diff = diff_decode(bits_out)

    shift = np.argmax(np.correlate(bits_out_non_diff, bits_non_diff))
    print(shift)
    #shift = 0
    min_length = min(len(bits_out_non_diff[shift:]), len(bits_non_diff))
    bit_errors = bits_out_non_diff[shift:][:min_length] != bits_non_diff[:min_length]

    # fig, ax = plt.subplots()
    # ax.plot(bits_out_non_diff[shift:][:10])
    # ax.plot(bits_non_diff[shift:][:10])
    # fig.show()

    bit_error_count = np.sum(bit_errors)
    bit_error_perc = bit_error_count / min_length

    return bit_errors, bit_error_count, bit_error_perc, min_length

#%%

def run_waterfall(EbNo_dB_levels = np.arange(0, 15, 1), mult_fnt = lambda x, y: x*y, num_symbols = 1002, **kwargs):
    bbit_error_perc = []
    
    for EbNo_dB in EbNo_dB_levels:
        bit_errors, bit_error_count, bit_error_perc, min_length = run_sequence(EbNo_dB, num_symbols, mult_fnt=mult_fnt, **kwargs)
        bbit_error_perc.append(bit_error_perc)
        print(f"EbN0:{EbNo_dB} b:{min_length} bec:{bit_error_count} be%:{bit_error_perc}")

    return EbNo_dB_levels, np.array(bbit_error_perc)


#%%
fig2, ax = plt.subplots()

mult_fnts = [lambda x, y: x*y, lambda x, y: ~((x > 0) ^ (y > 0)) * 2 - 1]
mult_fnts = [lambda x, y: x*y]
EbNo_dB_levels = np.arange(0, 15, 1)

for mult_fnt in mult_fnts:
    EbNo_dB_levels, bbit_error_perc = run_waterfall(EbNo_dB_levels=EbNo_dB_levels, mult_fnt = mult_fnt, 
                                                    num_symbols=50002,
                                                    costas = False,
                                                    awgn=True, frac_delay=False, 
                                                    mm_before=False, mm_after=True,
                                                    shape=True)
    ax.plot(EbNo_dB_levels, bbit_error_perc)

ax.plot(EbNo_dB_levels, Q(np.sqrt(2*10**(EbNo_dB_levels/10))))
ax.set_yscale('log')
ax.set_ylim((1e-8, 0))
plt.show()
pass
#%%