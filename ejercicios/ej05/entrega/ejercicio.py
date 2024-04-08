#!python3
import numpy as np
import matplotlib.pyplot as plt
import Signal
from matplotlib.image import NonUniformImage

fs = 10#16000000 # Sampling rate.
fsym = fs/16 # Symbol rate.

prbs_samples = 1000
prbs_gain = 10 #Symbol Energy PAM4 = 5*A**2 (Energy*2 if QAM16 (PAM4 in both axis))
samplingRatio = 4 # Ratio for upsampling and downsampling
FIR_samples = 20
FIR_waveform = "rrc" #rrc, sinc, sine, rect, tria, imp
FIR_mirror = True
channel_samples = 20
channel_waveform = "imp" 
channel_mirror = True
channel_noise_power = 0.0001 #(AWGN)

def normalize_energy(signal, reference):
    energy_signal = np.mean(np.abs(signal)**2)
    #print("energy signal = ", energy_signal)
    energy_reference = np.mean(np.abs(reference)**2)
    #print("energy ref = ", energy_reference)
    normalized_signal = signal * np.sqrt(energy_reference/energy_signal)
    #print("energy result = ", np.mean(np.abs(normalized_signal)**2))
    return normalized_signal

def block_LUT_PAM4(sigIn):
    LUT = [-3, -1, 1, 3]
    sigOut = [LUT[x] for x in sigIn]
    return sigOut

def block_prbs_pam4(samples, plot=False):
    prbs_r = np.round(np.random.uniform(low=-0.5, high=3.5, size=(prbs_samples,))).astype(int)
    prbs_i = np.round(np.random.uniform(low=-0.0, high=0.1, size=(prbs_samples,))).astype(int)
    prbs_r = block_LUT_PAM4(prbs_r)
    #prbs_i = block_LUT_PAM4(prbs_i)
    prbs = np.array([prbs_gain*complex(x, jy) for x, jy in zip(prbs_r, prbs_i)])
    if plot:
        plt.figure()
        plt.ion()
        plt.show()
        plt.subplot(2, 1, 1)
        plt.stem(np.arange(0, len(prbs), 1), prbs.imag, 'r', markerfmt='ro', label="b_imag[n] (PRBS PAM4)")
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.stem(np.arange(0, len(prbs), 1), prbs.real, 'b', markerfmt='bo', label="b_real[n] (PRBS PAM4)")
        plt.legend()
        plt.grid()
        plt.draw()
        plt.pause(0.001)
    return prbs

def block_upSampling(sigIn, ratio, plot=False):
    sigOut = []
    for i in range(len(sigIn)):
        sigOut.append(sigIn[i])
        sigOut.extend([0]*(ratio - 1))
    sigOut = np.array(sigOut)
    if plot:
        plt.figure()
        plt.ion()
        plt.show()
        plt.subplot(2, 1, 1)
        plt.stem(np.arange(0, len(sigOut), 1), sigOut.imag, 'r', markerfmt='ro', label="d_imag[n]")
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.stem(np.arange(0, len(sigOut), 1), sigOut.real, 'b', markerfmt='bo', label="d_real[n]")
        plt.legend()
        plt.grid()
        plt.draw()
        plt.pause(0.001)
    return sigOut

def block_downSampling(sigIn, ratio, plot=False):
    sigOut = []
    for i in range(0, len(sigIn), ratio):
        sigOut.append(sigIn[i])
    sigOut = np.array(sigOut)
    if plot:
        plt.figure()
        plt.ion()
        plt.show()
        plt.subplot(2, 1, 1)
        plt.stem(np.arange(0, len(sigOut), 1), sigOut.imag, 'r', markerfmt='ro', label="b~_imag[n]")
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.stem(np.arange(0, len(sigOut), 1), sigOut.real, 'b', markerfmt='bo', label="b~_real[n]")
        plt.legend()
        plt.grid()
        plt.draw()
        plt.pause(0.001)
    return sigOut

def response_filter_LPF():
    fc = fs/1
    Omega_c = (2*np.pi*fc)/(fs*samplingRatio)
    firOrder = 60
    h = []
    for n in range(0, firOrder + 1):
        if (n - firOrder/2) == 0: #Limit when n tends to 0 (sinc div by 0)
            h_sample = Omega_c / np.pi
        else:
            h_sample = np.sin(Omega_c*(n - firOrder/2)) / (np.pi*(n - firOrder/2))
        h.append(h_sample)
    return h

def rotate_complex_array(arr, angle_radians):
    rotation_factor = np.exp(1j * angle_radians)
    return arr * rotation_factor

def response_filter(waveform="sinc", mirror=False, plot=False):
    amp = 1
    freq = 4 #Hz
    phase = 0 #np.pi/2 #radians
    duty = 0.5 #*100%
    rollOff = 1.0
    rotation = 0#np.pi/8 #radians
    if waveform == "sinc":
        response = Signal.Sinc(amp, freq, phase, fs, FIR_samples)
    if waveform == "rrc":
        response = Signal.RootRaisedCosine(amp, freq, phase, rollOff, fs, FIR_samples)
    if waveform == "sine":
        response = Signal.Sine(amp, freq, phase, fs, FIR_samples)
    if waveform == "rect":
        response = Signal.Rectangular(amp, freq, duty, phase, fs, FIR_samples)
    if waveform == "tria":
        response = Signal.Triangular(amp, freq, duty, phase, fs, FIR_samples)
    if waveform == "imp":
        response = Signal.Impulse(amp, fs, FIR_samples)
    h = rotate_complex_array(np.array(response.amplitudes()), rotation)
    if mirror:
        h = np.concatenate((h[::-1], h[1:]))
    if plot:
        plt.figure()
        plt.ion()
        plt.show()
        plt.subplot(2, 1, 1)
        plt.stem(np.arange(0, len(h), 1), h.imag, 'r', markerfmt='ro', label="p_imag[n]")
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.stem(np.arange(0, len(h), 1), h.real, 'b', markerfmt='bo', label="p_real[n]")
        plt.legend()
        plt.grid()
        plt.draw()
        plt.pause(0.001)
    return h

def block_FIR(sigIn, plot=False):
    sigOut = np.convolve(sigIn, response_filter(FIR_waveform, FIR_mirror, plot))
    #sigOut = normalize_energy(sigOut, sigIn)
    if plot:
        plt.figure()
        plt.ion()
        plt.show()
        plt.subplot(2, 1, 1)
        plt.stem(np.arange(0, len(sigOut), 1), sigOut.imag, 'r', markerfmt='ro', label="xy_imag[n]")
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.stem(np.arange(0, len(sigOut), 1), sigOut.real, 'b', markerfmt='bo', label="xy_real[n]")
        plt.legend()
        plt.grid()
        plt.draw()
        plt.pause(0.001)
    return sigOut

def response_channel(waveform="imp", mirror=False, plot=False):
    amp = 1
    freq = 2 #Hz
    phase = 0#np.pi/2 #radians
    duty = 0.5 #*100%
    rollOff = 1.0
    rotation = 0#np.pi/4 #radians
    if waveform == "sinc":
        response = Signal.Sinc(amp, freq, phase, fs, FIR_samples)
    if waveform == "rrc":
        response = Signal.RootRaisedCosine(amp, freq, phase, rollOff, fs, FIR_samples)
    if waveform == "sine":
        response = Signal.Sine(amp, freq, phase, fs, FIR_samples)
    if waveform == "rect":
        response = Signal.Rectangular(amp, freq, duty, phase, fs, FIR_samples)
    if waveform == "tria":
        response = Signal.Triangular(amp, freq, duty, phase, fs, FIR_samples)
    if waveform == "imp":
        response = Signal.Impulse(amp, fs, FIR_samples)
    h = rotate_complex_array(np.array(response.amplitudes()), rotation)
    if mirror:
        h = np.concatenate((h[::-1], h[1:]))
    if plot:
        plt.figure()
        plt.ion()
        plt.show()
        plt.subplot(2, 1, 1)
        plt.stem(np.arange(0, len(h), 1), h.imag, 'r', markerfmt='ro', label="h_imag[n]")
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.stem(np.arange(0, len(h), 1), h.real, 'b', markerfmt='bo', label="h_real[n]")
        plt.legend()
        plt.grid()
        plt.draw()
        plt.pause(0.001)
    return h

def block_noise(noisePower, samples, plot=False):
    std = np.sqrt(noisePower/2)
    n_r = np.random.normal(loc=0, scale=std, size=samples)
    n_i = np.random.normal(loc=0, scale=std, size=samples)
    n = np.array([complex(x, jy) for x, jy in zip(n_r, n_i)])
    if plot:
        plt.figure()
        plt.ion()
        plt.show()
        plt.subplot(2, 1, 1)
        plt.stem(np.arange(0, len(n), 1), n.imag, 'r', markerfmt='ro', label="n_imag[n]")
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.stem(np.arange(0, len(n), 1), n.real, 'b', markerfmt='bo', label="n_real[n]")
        plt.legend()
        plt.grid()
        plt.draw()
        plt.pause(0.001)
    return n

def block_channel(sigIn, noisePower, plot=False):
    sigOut = np.convolve(sigIn, response_channel(channel_waveform, channel_mirror, plot))
    sigOut += block_noise(noisePower, len(sigOut), plot)
    #sigOut = normalize_energy(sigOut, sigIn)

    if plot:
        plt.figure()
        plt.ion()
        plt.show()
        plt.subplot(2, 1, 1)
        plt.stem(np.arange(0, len(sigOut), 1), sigOut.imag, 'r', markerfmt='ro', label="c_imag[n]")
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.stem(np.arange(0, len(sigOut), 1), sigOut.real, 'b', markerfmt='bo', label="c_real[n]")
        plt.legend()
        plt.grid()
        plt.draw()
        plt.pause(0.001)
    return sigOut

def decode_symbol_PAM4(sigIn, sigGain, plot=False):
    #decode real axis only
    sigOut = []
    for x in sigIn.real:
        if (-4*sigGain) < x and x <= (-2*sigGain):
            sigOut.append(-3*sigGain)
        elif (-2*sigGain) < x and x <= (0*sigGain):
            sigOut.append(-1*sigGain)
        elif (0*sigGain) < x and x <= (2*sigGain):
            sigOut.append(1*sigGain)
        elif (2*sigGain) < x and x <= (4*sigGain):
            sigOut.append(3*sigGain)
        else:
            sigOut.append(0) #Out of range
    return np.array(sigOut)

def count_diffs(array1, array2):
    if array1.shape != array2.shape:
        raise ValueError("Arrays must have the same shape.")
    return np.sum(array1 != array2)

def get_mse(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("Arrays must have the same shape.")

    squared_diff = np.abs(actual - expected)**2
    mse = np.mean(squared_diff)
    return mse

def plot_BER_MSE():
    SNR_list = np.logspace(-3, 1, 20) #From 0 to 10 SNR. #np.linspace(0.001, 10, 20, endpoint=False)
    MSE_list = []
    BER_list = []
    for SNR in SNR_list:
        signal_b = block_prbs_pam4(prbs_samples, False)
        signal_power = np.mean(np.abs(signal_b)**2)
        noise = signal_power / SNR
        signal_d = block_upSampling(signal_b, samplingRatio, False)
        signal_x = block_FIR(signal_d, False)
        if FIR_mirror:
            signal_x = signal_x[FIR_samples-1 : FIR_samples-1 + len(signal_d)]
        else:
            signal_x = signal_x[:len(signal_d)]
        signal_c = block_channel(signal_x, noise, False)
        if channel_mirror:
            signal_c = signal_c[channel_samples-1 : channel_samples-1 + len(signal_x)]
        else:
            signal_c = signal_c[:len(signal_x)]
        signal_y = block_FIR(signal_c, False)
        if FIR_mirror:
            signal_y = signal_y[FIR_samples-1 : FIR_samples-1 + len(signal_c)]
        else:
            signal_y = signal_y[:len(signal_c)]
        signal_b_ = block_downSampling(signal_y, samplingRatio, False)
        signal_b_ = normalize_energy(signal_b_, signal_b)
        errors = count_diffs(signal_b.real, decode_symbol_PAM4(signal_b_.real, prbs_gain))
        BER = errors/prbs_samples
        MSE = get_mse(signal_b_, signal_b)
        BER_list.append(BER)
        MSE_list.append(MSE)
    plt.figure()
    plt.plot(SNR_list, BER_list, label="BER vs SNR")
    plt.title("BER vs SNR")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(SNR_list, MSE_list, label="MSE vs SNR")
    plt.title("MSE vs SNR")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

def plot_eye_diagram(sigIn):
    plt.figure()
    xedges = []
    for idx in np.arange(0, len(sigIn), 1):
        eyeDiagram = []
        xAxis = []
        prevIdx = idx - 1
        nextIdx = (idx + 1)%len(sigIn)
        eyeDiagram.append(sigIn[prevIdx])
        xAxis.append(-1)
        eyeDiagram.append(sigIn[idx])
        xAxis.append(0)
        eyeDiagram.append(sigIn[nextIdx])
        xAxis.append(1)
        plt.plot(xAxis, eyeDiagram)
    plt.title("Eye Diagram")
    plt.grid()
    plt.show()
    



def main():
    signal_b = block_prbs_pam4(prbs_samples, True)
    print("Power of b is: ", np.mean(np.abs(signal_b)**2))

    signal_d = block_upSampling(signal_b, samplingRatio, True)
    
    signal_x = block_FIR(signal_d, True)
    if FIR_mirror:
        signal_x = signal_x[FIR_samples-1 : FIR_samples-1 + len(signal_d)]
    else:
        signal_x = signal_x[:len(signal_d)]

    signal_n = block_noise(channel_noise_power, channel_samples, False)
    print("Power of n is: ", np.mean(np.abs(signal_n)**2))
    
    signal_c = block_channel(signal_x, channel_noise_power, True)
    if channel_mirror:
        signal_c = signal_c[channel_samples-1 : channel_samples-1 + len(signal_x)]
    else:
        signal_c = signal_c[:len(signal_x)]

    signal_y = block_FIR(signal_c, True)
    if FIR_mirror:
        signal_y = signal_y[FIR_samples-1 : FIR_samples-1 + len(signal_c)]
    else:
        signal_y = signal_y[:len(signal_c)]
    
    
    signal_b_ = block_downSampling(signal_y, samplingRatio, True)
    print("Power of b~ is: ", np.mean(np.abs(signal_b_)**2))
    signal_b_ = normalize_energy(signal_b_, signal_b)
    print("Power of b~ normalized is: ", np.mean(np.abs(signal_b_)**2))



    errors = count_diffs(signal_b.real, decode_symbol_PAM4(signal_b_.real, prbs_gain))
    print("Errors = ", errors)
    print("BER = ", errors/prbs_samples)
    
    plt.figure()
    plt.ion()
    plt.show()
    plt.plot(signal_b_.real, signal_b_.imag, 'x', label="b~ symbols")
    plt.legend()
    plt.grid()
    # ax = plt.gca()
    # ax.set_xlim([-4, 4])
    # ax.set_ylim([-4, 4])
    plt.draw()
    plt.pause(0.001)

    plot_eye_diagram(signal_b_.real)



    # signalCos = Signal.RootRaisedCosine(3, 1, 0, 1.0, 50, 100)
    # signalCos2 = Signal.Sinc(3, 1, 0, 50, 100)
    # plt.figure()
    # plt.ion()
    # plt.show()
    # plt.stem(signalCos.samples(), signalCos.amplitudes(), 'r', markerfmt='ro', label="cos")
    # plt.stem(signalCos2.samples(), signalCos2.amplitudes(), 'b', markerfmt='bo', label="sinc")
    # plt.legend()
    # plt.grid()
    # plt.draw()
    # plt.pause(0.001)


    input("Press [enter] to close.")
    


if __name__ == "__main__":
   main()
   #plot_BER_MSE()
   #input("Press [enter] to close.")