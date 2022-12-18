from scipy.io import wavfile # scipy library to read wav files
import numpy as np

#AudioName = "vignesh.wav" # Audio File
AudioName = "test.wav" 
fs, Audiodata = wavfile.read(AudioName)
print(len(Audiodata))
print(Audiodata[int(2232/4):int(2232/4)+40])
# # Plot the audio signal in time
# # import matplotlib.pyplot as plt
# # plt.plot(Audiodata)
# # plt.title('Audio signal in time',size=16)

f = open(AudioName, 'rb')
f.seek(0)
d = f.read()
data = np.frombuffer(d, dtype=np.int16)
i = int((78+2232)/2)
print(data[i:i+80])

class SimpleBeatDetection:
    """
    Simple beat detection algorithm from
    http://archive.gamedev.net/archive/reference/programming/features/beatdetection/index.html
    """
    def __init__(self, history = 43):
        self.local_energy = np.zeros(history) # a simple ring buffer
        self.local_energy_index = 0 # the index of the oldest element

    def detect_beat(self, signal):
        samples = signal
        #samples = signal.astype(np.int) # make room for squares
        # optimized sum of squares, i.e faster version of (samples**2).sum()
        instant_energy = np.dot(samples, samples) #/ float(0xffffffff) # normalize

        local_energy_average = self.local_energy.mean()
        local_energy_variance = self.local_energy.var()

        beat_sensibility = (-0.0025714 * local_energy_variance) + 1.15142857
        beat = instant_energy > beat_sensibility * local_energy_average

        self.local_energy[self.local_energy_index] = instant_energy
        self.local_energy_index -= 1
        if self.local_energy_index < 0:
            self.local_energy_index = len(self.local_energy) - 1

        return beat
		
		
sb = SimpleBeatDetection()
a = []
batch = 0
bsize = 1024


# for i in range(bsize,len(Audiodata),bsize):
	# batch =np.frombuffer(Audiodata[i-bsize:i], np.int16)
	
	# if (sb.detect_beat(batch)):
		# [a.append(10000) for j in range(bsize) ]
	# else: 
		# [a.append(0)for j in range(bsize) ]


# for i,s in enumerate(Audiodata):
	
	# batch += s
	# if i % bsize == 0:
		# if (sb.detect_beat(batch)):
			# [a.append(10000) for j in range(bsize) ]
		# else: 
			# [a.append(0)for j in range(bsize) ]
		# batch = 0

# plt.plot(a)
# plt.show()
# # spectrum
# from scipy.fftpack import fft # fourier transform
# n = len(Audiodata) 
# AudioFreq = fft(Audiodata)
# AudioFreq = AudioFreq[0:int(np.ceil((n+1)/2.0))] #Half of the spectrum
# MagFreq = np.abs(AudioFreq) # Magnitude
# MagFreq = MagFreq / float(n)
# # power spectrum
# MagFreq = MagFreq**2
# if n % 2 > 0: # ffte odd 
    # MagFreq[1:len(MagFreq)] = MagFreq[1:len(MagFreq)] * 2
# else:# fft even
    # MagFreq[1:len(MagFreq) -1] = MagFreq[1:len(MagFreq) - 1] * 2 

# plt.figure()
# freqAxis = np.arange(0,int(np.ceil((n+1)/2.0)), 1.0) * (fs / n);
# plt.plot(freqAxis/1000.0, 10*np.log10(MagFreq)) #Power spectrum
# plt.xlabel('Frequency (kHz)'); plt.ylabel('Power spectrum (dB)');


# #Spectrogram
# from scipy import signal
# N = 512 #Number of point in the fft
# f, t, Sxx = signal.spectrogram(Audiodata, fs,window = signal.blackman(N),nfft=N)
# plt.figure()
# plt.pcolormesh(t, f,10*np.log10(Sxx)) # dB spectrogram
# #plt.pcolormesh(t, f,Sxx) # Lineal spectrogram
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [seg]')
# plt.title('Spectrogram with scipy.signal',size=16);

# plt.show()