#!python3
import numpy as np
import matplotlib.pyplot as plt

def block_prbs():
    prbs = [1, 1, 0, 1, 0, 1, 1, 0, 0, 1]
    return prbs

def block_m(sigIn, M):
    sigOut = []
    for i in range(len(sigIn)):
        sigOut.append(sigIn[i])
        sigOut.extend([0]*(M - 1))
    return sigOut

def convolve(sigIn, transferFunc):
    # Extend with zero padding the original sample and the FIR response
    # in time domain to get N+M-1 points in frec domain and make a
    # direct multiplication.

    # Extend original sample and transfer function in time domain
    sigInExtended_t = list(sigIn) + ([0]*(len(transferFunc) - 1))
    transferFuncExtended_t = list(transferFunc) + ([0]*(len(sigIn) - 1))
    # FFT to get the signals in frec domain.
    sigInExtended_f = np.fft.fft(sigInExtended_t)
    sigInExtended_f = np.fft.fftshift(sigInExtended_f)
    sigInExtended_f = np.abs(sigInExtended_f)
    transferFuncExtended_f = np.fft.fft(transferFuncExtended_t)
    transferFuncExtended_f = np.fft.fftshift(transferFuncExtended_f)
    transferFuncExtended_f = np.abs(transferFuncExtended_f)
    # Multiplication in frec domain.
    sigOut_f = [i*j for i,j in zip(sigInExtended_f, transferFuncExtended_f)]
    # IFFT to obtain output signal in time domain.
    sigOut = np.fft.fftshift(sigOut_f)
    sigOut = np.fft.ifft(sigOut)
    sigOut = np.real(sigOut)
    return sigOut

def canal(sigIn):
    transfer_h = [1] #ideal
    sigIn = np.convolve(sigIn, transfer_h) #convolve(sigIn, transfer_h)
    noise_n = [0]*len(sigIn)
    sigOut = [sum(x) for x in zip(sigIn, noise_n)]
    return sigOut

def main():
    M = 2
    signal_b = block_prbs()
    signal_d = block_m(signal_b, M)
    signal_p = [1, 1, 1, 1, 0, 0, 0, 0]
    signal_x = np.convolve(signal_d, signal_p) #convolve(signal_d, signal_p)
    signal_c = canal(signal_x)

    fig = plt.figure()
    
    subFig5 = fig.add_subplot(5,1,5)
    subFig5.stem(np.arange(0, len(signal_c), 1), signal_c, 'b', markerfmt='bo', label="c[n]")
    subFig1 = fig.add_subplot(5,1,1, sharex=subFig5)
    subFig1.stem(np.arange(0, len(signal_b), 1), signal_b, 'b', markerfmt='bo', label="b[n]")
    subFig2 = fig.add_subplot(5,1,2, sharex=subFig5)
    subFig2.stem(np.arange(0, len(signal_d), 1), signal_d, 'b', markerfmt='bo', label="d[n]")
    subFig3 = fig.add_subplot(5,1,3, sharex=subFig5)
    subFig3.stem(np.arange(0, len(signal_p), 1), signal_p, 'b', markerfmt='bo', label="p[n]")
    subFig4 = fig.add_subplot(5,1,4, sharex=subFig5)
    subFig4.stem(np.arange(0, len(signal_x), 1), signal_x, 'b', markerfmt='bo', label="x[n]")

    subFig1.legend(loc='upper left')
    subFig2.legend(loc='upper left')
    subFig3.legend(loc='upper left')
    subFig4.legend(loc='upper left')
    subFig5.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()