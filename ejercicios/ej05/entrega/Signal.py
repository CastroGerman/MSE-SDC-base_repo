#!python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class Signal:
    def __init__(self, fs, N):
        self.fs = fs # fs = sampling frequency.
        self.N = N # N = samples amount.
    
    def samples(self):
        return np.arange(0, self.N/self.fs, 1/self.fs)
    
    def plot(self):
        print("Nothing to plot here.")

class Sine(Signal):
    def __init__(self, amp, freq, phase, fs, N):
        super().__init__(fs, N)
        self.amp = amp
        self.freq = freq
        self.phase = phase
    
    def evaluate(self, time):
        return self.amp * np.sin(2*np.pi*self.freq*time + self.phase) # Math function f(x)

    def amplitudes(self):
        return self.evaluate(self.samples())
        

    def plot(self, figure=0):
        plt.figure(figure)
        # Get time values of the sine wave
        time = np.arange(0, self.N/self.fs, 1/self.fs)
        # Amplitude of the sine wave is sine of a variable like time
        amplitude = self.evaluate(time)
        # Plot a sine wave using time and amplitude obtained for the sine wave
        plt.plot(time, amplitude)
        plt.title('Sine wave')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        #plt.show()
    
class Rectangular(Signal):
    def __init__(self, amp, freq, duty, phase, fs, N):
        super().__init__(fs, N)
        self.amp = amp
        self.freq = freq
        self.duty = duty
        self.phase = phase
    
    def evaluate(self, time):
        return self.amp * signal.square(2*np.pi*self.freq*time + self.phase, self.duty)
    
    def samples(self):
        return np.arange(0, self.N/self.fs, 1/self.fs)

    def amplitudes(self):
        return self.evaluate(self.samples())

    def plot(self, figure=0):
        plt.figure(figure)
        # Get time values of the sine wave
        time = np.arange(0, self.N/self.fs, 1/self.fs)
        # Amplitude of the sine wave is sine of a variable like time
        amplitude = self.evaluate(time)
        # Plot a sine wave using time and amplitude obtained for the sine wave
        plt.plot(time, amplitude)
        plt.title('Rectangular wave')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        #plt.show()

class Triangular(Signal):
    def __init__(self, amp, freq, duty, phase, fs, N):
        super().__init__(fs, N)
        self.amp = amp
        self.freq = freq
        self.duty = duty
        self.phase = phase
    
    def evaluate(self, time):
        return self.amp * signal.sawtooth(2*np.pi*self.freq*time + self.phase, self.duty)
    
    def samples(self):
        return np.arange(0, self.N/self.fs, 1/self.fs)

    def amplitudes(self):
        return self.evaluate(self.samples())

    def plot(self, figure=0):
        plt.figure(figure)
        # Get time values of the sine wave
        time = np.arange(0, self.N/self.fs, 1/self.fs)
        # Amplitude of the sine wave is sine of a variable like time
        amplitude = self.evaluate(time)
        # Plot a sine wave using time and amplitude obtained for the sine wave
        plt.plot(time, amplitude)
        plt.title('Triangular wave')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        #plt.show()
    
class Impulse(Signal):
    def __init__(self, amp, fs, N):
        super().__init__(fs, N)
        self.amp = amp
    
    def evaluate(self, time):
        return self.amp * signal.unit_impulse(self.N)
    
    def samples(self):
        return np.arange(0, self.N/self.fs, 1/self.fs)

    def amplitudes(self):
        return self.evaluate(self.samples())

def sign_sqrt_list(lst):
    arr = np.array(lst)
    sqrt_arr = np.sqrt(np.abs(arr))
    result = np.where(arr < 0, -sqrt_arr, sqrt_arr)
    return result.tolist()
def sign_power_of_2(lst):
    arr = np.array(lst)
    result = np.sign(arr) * np.power(np.abs(arr), 2)
    return result.tolist()

class RootRaisedCosine(Signal):
    def __init__(self, amp, freq, phase, rollOff, fs, N):
        super().__init__(fs, N)
        self.amp = amp
        self.freq = freq
        self.phase = phase
        self.rollOff = rollOff
    
    def evaluate(self, time):        
        signal = []
        for i in time:
            if i == 0:
                chunk1 = self.amp*self.freq*(1 + self.rollOff*(-1 + 2/(np.pi/2)))
                signal.append(chunk1)
            elif (i == 1/(4*self.rollOff*self.freq)) or (i == -1/(4*self.rollOff*self.freq)):
                chunk1 = (1 + 1/(np.pi/2))*np.sin((np.pi/2)/(2*self.rollOff))
                chunk2 = (1 - 1/(np.pi/2))*np.cos((np.pi/2)/(2*self.rollOff))
                chunk3 = self.rollOff*self.freq/np.sqrt(2)
                signal.append(self.amp * chunk3 * (chunk1 + chunk2))
            else:
                chunk1 = np.sin(2*(np.pi/2)*self.freq*i*(1 - self.rollOff) + self.phase)
                chunk2 = 4*self.rollOff*self.freq*i*np.cos(2*(np.pi/2)*self.freq*i*(1 + self.rollOff) + self.phase)
                chunk3 = 2*(np.pi/2)*i*(1 - (4*self.rollOff*self.freq*i)**2)
                signal.append(self.amp * (chunk1 + chunk2)/chunk3)
        
        return signal

        
    def amplitudes(self):
        return self.evaluate(self.samples())

    def plot(self, figure=0):
        plt.figure(figure)
        # Get time values of the sine wave
        time = np.arange(0, self.N/self.fs, 1/self.fs)
        # Amplitude of the sine wave is sine of a variable like time
        amplitude = self.evaluate(time)
        # Plot a sine wave using time and amplitude obtained for the sine wave
        plt.plot(time, amplitude)
        plt.title('Sine wave')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        #plt.show()

class Sinc(Signal):
    def __init__(self, amp, freq, phase, fs, N):
        super().__init__(fs, N)
        self.amp = amp
        self.freq = freq
        self.phase = phase
    
    def evaluate(self, time):
        signal = []
        for i in time:
            if i == 0:
                signal.append(self.amp)
            else:
                signal.append(self.amp * np.sin(np.pi*self.freq*i + self.phase) / (np.pi*self.freq*i))
        return signal

    def amplitudes(self):
        return self.evaluate(self.samples())

    def plot(self, figure=0):
        plt.figure(figure)
        # Get time values of the sine wave
        time = np.arange(0, self.N/self.fs, 1/self.fs)
        # Amplitude of the sine wave is sine of a variable like time
        amplitude = self.evaluate(time)
        # Plot a sine wave using time and amplitude obtained for the sine wave
        plt.plot(time, amplitude)
        plt.title('Sine wave')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        #plt.show()