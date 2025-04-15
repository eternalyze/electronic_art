

import wave
import struct
import random
import numpy as np
import matplotlib.pyplot as plt


a = "/mnt/c/Users/nullo/OneDrive/Desktop/alien_A.wav"
b = "/mnt/c/Users/nullo/OneDrive/Desktop/alien_B.wav"
c = "/mnt/c/Users/nullo/OneDrive/Desktop/alien_C.wav"
d = "/mnt/c/Users/nullo/OneDrive/Desktop/alien_D.wav"

A = wave.open(a, 'rb')
B = wave.open(b, 'rb')
C = wave.open(c, 'rb')
D = wave.open(d, 'rb')

drugz = [a,b,c,d,b,c,d,b,c,d,b,c,d,b,c,d,a]

data= []
for drug in drugz:
    w = wave.open(drug, 'rb')
    # print(w.getparams())
    data.append( [w.getparams(), w.readframes(w.getnframes())] )
    w.close()

sampleRate = 44100.0/2  # hertz
duration = 1.0  # seconds
frequency = 440.0  # hertz


obj = wave.open('sound.wav', 'w')
obj.setnchannels(1)  # mono
obj.setsampwidth(2)
obj.setframerate(sampleRate)
values = []

import numpy as np
import matplotlib.pyplot as plt

# N = 1024
# fs = 50
# df = fs / N
# f = np.arange(-N / 2, N / 2) * df
# w = 2 * np.pi * f


# def hw(w):
#     h = []
#     for i in np.arange(len(w)):
#         if w[i] > 0:
#             h.append(np.exp(-1 / w[i] / w[i]))
#         else:
#             h.append(0)
#     return np.array(h)


# def gw(w):
#     return hw(4 * np.pi / 3 - w) / (hw(w - 2 * np.pi / 3) + hw(4 * np.pi / 3 - w))


# def phiw(w):
#     return np.sqrt(gw(w) * gw(-w))


# def syw(w):
#     return np.exp(-1j * w / 2) * np.sqrt(phiw(w / 2) * phiw(w / 2) - phiw(w) * phiw(w))


# s = syw(w)
# st = np.fft.ifft(np.fft.ifftshift(s))
# # st = np.fft.fftshift(np.real(st))
# st = np.fft.fftshift(np.real(st)) * fs
# t = np.arange(-N / 2, N / 2) / fs

# plt.figure(1)
# plt.plot(t, st)
# plt.xlim([-10, 10])
# # plt.show()
# plt.savefig("meyer_wavelet.png")

import numpy as np
import matplotlib.pyplot as plt

def ricker_wavelet(t, sigma=1.0, freq=10.0):
    """
    Generates the Ricker (Mexican Hat) wavelet for a given time array.
    
    Parameters:
    t (numpy array): Time values.
    sigma (float): Width parameter of the wavelet.
    
    Returns:
    numpy array: Amplitude values of the wavelet.
    """
    t_normalized = t / sigma
    return (1 - freq*t_normalized**2) * np.exp(-0.5 * freq*t_normalized**2) * np.sin(freq*t_normalized)

# Generate time values from -5 to 5 seconds
t = np.linspace(0, 20, 1000)

# Create the wavelet with default sigma=1.0
wavelet = ricker_wavelet(t-10)

# Plot the wavelet
# plt.figure(figsize=(8, 4))
# plt.plot(t, wavelet, label='Sigma=1.0')
# plt.title("Ricker (Mexican Hat) Wavelet")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.legend()
# # plt.show()
# plt.savefig("ricker_wavelet.png")

for a in range(10*5):
    for i in range(int(sampleRate * duration)):
        if i%5:
            continue
        # value = random.randint(-32767, 32767)
        # if i % 1000 or i==0:
            # pixel = np.random.randint(10)
        # value = int((i % 32767)*
        # if i % 1 == 0:
            # value = (i%32000)*np.sin((i)/(6*10))
            # value = 12000 * ( np.sin(2*i/60-100) - np.sin(i/60-100) )/((i+1)/60-100)
        x = i - sampleRate/2
        # value = 32000* np.exp(-x**2 / (2*6000)) * np.cos(5*x/6000)
        value = 0 
        freqs = [2**x for x in range(12)]
        # for n in range(1,11):
            # value+=32000*ricker_wavelet((i-sampleRate/n)/100, freq=freqs[n])
        for n in range(1,11):
            my_freq = freqs[n]
            if a % 3 == 1:
                my_freq *= 3
            if a % 3 == 2:
                my_freq *= 0.5
            value+=32000*ricker_wavelet((i-sampleRate/n)/100, freq=my_freq)
        value = int(value)
        values.append(value)
        data = struct.pack('h', value)
        obj.writeframesraw(data)

obj.close()

plt.plot(values)
plt.savefig("ricker_wavelets.png")
