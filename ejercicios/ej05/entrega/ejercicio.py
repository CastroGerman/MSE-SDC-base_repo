#!python3
import numpy as np
import matplotlib.pyplot as plt
import Signal
from matplotlib.image import NonUniformImage

fs = 10#16000000 # Sampling rate.
fsym = fs/16 # Symbol rate.
channel_fs_k = 1/10
fir_fs_k = 1/10

prbs_samples = 5000
prbs_gain = 1 #Symbol Energy PAM4 = 5*A**2
samplingRatio = 10 # Ratio for upsampling and downsampling
FIR_samples = 80
FIR_waveform = "rrc" #rrc, sinc, sine, rect, tria, imp
FIR_mirror = True
channel_samples = 100
channel_waveform = "imp" 
channel_mirror = True
channel_noise_power = 0.5 #(AWGN)

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

def block_prbs_pam4(samples):
    prbs_r = np.floor(np.random.uniform(low=0, high=4, size=(prbs_samples,))).astype(int)
    prbs_i = np.floor(np.random.uniform(low=0, high=0, size=(prbs_samples,))).astype(int)
    prbs_r = block_LUT_PAM4(prbs_r)
    #prbs_i = block_LUT_PAM4(prbs_i)
    prbs = np.array([prbs_gain*complex(x, jy) for x, jy in zip(prbs_r, prbs_i)])
    return prbs

def block_upSampling(sigIn, ratio):
    sigOut = []
    for i in range(len(sigIn)):
        sigOut.append(sigIn[i])
        sigOut.extend([0]*(ratio - 1))
    sigOut = np.array(sigOut)
    return sigOut

def block_downSampling(sigIn, ratio):
    sigOut = []
    for i in range(0, len(sigIn), ratio):
        sigOut.append(sigIn[i])
    sigOut = np.array(sigOut)
    return sigOut

def rotate_complex_array(arr, angle_radians):
    rotation_factor = np.exp(1j * angle_radians)
    return arr * rotation_factor

def response_filter(waveform="sinc", mirror=False):
    amp = 1
    freq = fs*fir_fs_k #Hz
    phase = 0 #np.pi/2 #radians
    duty = 0.5 #*100%
    rollOff = 0.5
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
    return h

def block_FIR(sigIn):
    sigOut = np.convolve(sigIn, response_filter(FIR_waveform, FIR_mirror))
    return sigOut

def response_channel(waveform="imp", mirror=False):
    amp = 1
    freq = fs*channel_fs_k #Hz
    phase = 0#np.pi/2 #radians
    duty = 1 #*100%
    rollOff = 1.0
    rotation = (00*np.pi/180) #radians
    if waveform == "sinc":
        response = Signal.Sinc(amp, freq, phase, fs, channel_samples)
    if waveform == "rrc":
        response = Signal.RootRaisedCosine(amp, freq, phase, rollOff, fs, channel_samples)
    if waveform == "sine":
        response = Signal.Sine(amp, freq, phase, fs, channel_samples)
    if waveform == "rect":
        response = Signal.Rectangular(amp, freq, duty, phase, fs, channel_samples)
    if waveform == "tria":
        response = Signal.Triangular(amp, freq, duty, phase, fs, channel_samples)
    if waveform == "imp":
        response = Signal.Impulse(amp, fs, channel_samples)
    h = rotate_complex_array(np.array(response.amplitudes()), rotation)
    if mirror:
        h = np.concatenate((h[::-1], h[1:]))
    return h

def block_noise(noisePower, samples):
    std = np.sqrt(noisePower/2)
    n_r = np.random.normal(loc=0, scale=std, size=samples)
    n_i = np.random.normal(loc=0, scale=std, size=samples)
    n = np.array([complex(x, jy) for x, jy in zip(n_r, n_i)])
    return n

def block_channel(sigIn, noisePower):
    sigOut = np.convolve(sigIn, response_channel(channel_waveform, channel_mirror))
    sigOut += block_noise(noisePower, len(sigOut))
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

def plot_eye_diagram(sigIn, samplesPerPeriod):
    plt.figure()
    for idx in np.arange(0, len(sigIn) - (samplesPerPeriod*2), (samplesPerPeriod*2)):
        eyeDiagram = []
        xAxis = []
        for sppIdx in np.arange(0, (samplesPerPeriod*2) + 1, 1):
            if (idx + sppIdx) < len(sigIn):    
                xAxis.append(sppIdx)
                eyeDiagram.append(sigIn[idx + sppIdx])
        plt.plot(xAxis, eyeDiagram)
        # plt.draw()
        # plt.pause(0.001)
        # input("Press [enter] to continue.")
    plt.title("Eye Diagram")
    plt.grid()
    plt.show()

def plotSignal(signal, name):
    plt.figure()
    plt.ion()
    plt.show()
    plt.subplot(2, 1, 1)
    plt.stem(np.arange(0, len(signal), 1), signal.imag, 'r', markerfmt='ro', label=name+"_imag[n]")
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.stem(np.arange(0, len(signal), 1), signal.real, 'b', markerfmt='bo', label=name+"_real[n]")
    plt.legend()
    plt.grid()
    plt.draw()
    plt.pause(0.001)

def system(sigIn, noisePower, plotSignals=False, plotSymbols=False, plotEye=False):
    samplesToCut_upper = (FIR_samples - 1)*2 + (channel_samples - 1)
    samplesToCut_lower = 0
    if FIR_mirror:
        samplesToCut_lower += (FIR_samples - 1)*2
    if channel_mirror:
        samplesToCut_lower += (channel_samples - 1)
    sigIn_power = np.mean(np.abs(sigIn)**2)
    print("Input signal power is: ", sigIn_power)
    signal_p = response_filter(FIR_waveform, FIR_mirror)
    signal_h = response_channel(channel_waveform, channel_mirror)
    signal_n = block_noise(noisePower, channel_samples)
    signal_n_power = np.mean(np.abs(signal_n)**2)
    print("Power of n is: ", signal_n_power)

    signal_d = block_upSampling(sigIn, samplingRatio)
    signal_x = block_FIR(signal_d)
    signal_c = block_channel(signal_x, noisePower)
    signal_y = block_FIR(signal_c)
    signal_y_trimmed = signal_y[samplesToCut_lower : len(signal_y) - samplesToCut_upper]
    signal_b_ = block_downSampling(signal_y_trimmed, samplingRatio)
    print("Power of b~ is: ", np.mean(np.abs(signal_b_)**2))
    signal_b_ = normalize_energy(signal_b_, sigIn)
    print("Power of b~ normalized is: ", np.mean(np.abs(signal_b_)**2))

    errors = count_diffs(sigIn.real, decode_symbol_PAM4(signal_b_.real, prbs_gain))
    print("Errors = ", errors)
    print("BER = ", errors/prbs_samples)

    if plotSignals:
        plotSignal(sigIn, "b")
        plotSignal(signal_d, "d")
        plotSignal(signal_p, "p")
        plotSignal(signal_x, "x")
        plotSignal(signal_h, "h")
        plotSignal(signal_n, "n")
        plotSignal(signal_c, "c")
        plotSignal(signal_y, "y")
        plotSignal(signal_y_trimmed, "y_trimmed")
        plotSignal(signal_b_, "b~")
        input("Press [enter] to continue.")
    if plotSymbols:
        plt.figure()
        plt.ion()
        plt.show()
        plt.plot(signal_b_.real, signal_b_.imag, 'x', label="b~ symbols")
        plt.legend()
        plt.title("Symbols with SNR = {:.1f} ({:.2f}/{:.3f})".format(sigIn_power/signal_n_power, sigIn_power, signal_n_power))
        plt.grid()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
        pass
    if plotEye:
        plot_eye_diagram(signal_y_trimmed.real, samplingRatio)
        input("Press [enter] to continue.")
    
    return signal_b_

def plot_BER_MSE(sigIn):
    SNR_list = np.logspace(-3, 1, 20) #From 0 to 10 SNR. #np.linspace(0.001, 10, 20, endpoint=False)
    MSE_list = []
    BER_list = []
    signal_power = np.mean(np.abs(sigIn)**2)
    for SNR in SNR_list:
        noise_power = signal_power / SNR
        sigOut = system(sigIn, noise_power, False, False, False)
        errors = count_diffs(sigIn.real, decode_symbol_PAM4(sigOut.real, prbs_gain))
        BER = errors/prbs_samples
        MSE = get_mse(sigOut, sigIn)
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
    input("Press [enter] to continue.")

def main():
    signal_b = block_prbs_pam4(prbs_samples)
    system(signal_b, channel_noise_power, True, True, True)
    plot_BER_MSE(signal_b)

if __name__ == "__main__":
    main()
