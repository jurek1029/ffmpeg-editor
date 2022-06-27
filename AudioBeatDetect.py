from matplotlib import pyplot as plt
from scipy.io import wavfile # scipy library to read wav files
import numpy as np
import os
from MyLogger import printe,printw,printi,setLevel
import subprocess
from numpy.fft import fft, ifft
import numba

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

class AudioProcessor:
    _tmpAudio = './tmp/audio.wav'
    _C = 1.6
    _sampleSize = 1024
    _fs = 48000
    _minLengthInSec = 0.1
    _bufferSizeInSamples = 48000
    _avgEnergySq = []
    _avgVarEnergy = []
    _dataSq = []

    _useFFT = True
    _useLowPassFilter = True
    _useAutoCFromVar = True
    _FFTSubbands = 32
    _a = 2
    _b = 1
    _FFTSubbandsData = []
    _FFTSubbandsAvgBuffs = []
    _FFTSubbandsAvgVar = []

    _debug = False
    _showPlots = False
    _pltAvgEnergy = []
    _pltEnergy = []
    _pltAvgVar = []
    _pltsSubband = []
    _pltsBeatsPerSub = []

    def __init__(self,fileName, C = 1.6, useAutoConst = True ,useFFT = True, useLowPassFilter = True ,buffSizeInSec = 1, minLengthInSec = 0.1,
                sampleSize = 1024, FFTSubbands = 32, filterFuncPrams = (2,1), debug=False, showPlots = False):

        self._debug = debug
        self._showPlots = showPlots
        if self._debug: setLevel(2) # info level
        else:           setLevel(0)

        if not os.path.exists(fileName): printe("Music File not found")
        ext = os.path.splitext(fileName)[1]
        if ext != '.wav':
            printi(f"Converting {ext} to .wav")
            if os.path.exists(self._tmpAudio): os.remove(self._tmpAudio)
            comm = f'ffmpeg -i "{fileName}" "{self._tmpAudio}"'
            subprocess.run(comm, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) #TODO ffmpeg error handling
            fileName = './tmp/audio.wav'
        
        printi(f"opening: {fileName}")
        self.fs, data = wavfile.read(fileName)
        self._C = C
        self._useFFT = useFFT
        self._useLowPassFilter = useLowPassFilter
        self._useAutoCFromVar = useAutoConst
        self._minLengthInSec = minLengthInSec
        self._sampleSize = sampleSize
        self._a = filterFuncPrams[0]
        self._b = filterFuncPrams[1]
        self._bufferSizeInSamples = int(buffSizeInSec*self.fs // self._sampleSize)
        self._FFTSubbands = FFTSubbands
        if sampleSize % FFTSubbands != 0:
            printw(f'FFTSubbands count should devide sample size: {sampleSize} % {FFTSubbands} = {sampleSize % FFTSubbands}')
            self._sampleSize -= sampleSize % FFTSubbands
        if sum([self._a*i + self._b for i in range(self._FFTSubbands)]) != self._sampleSize:
            printe(f'filter function paramiters not suming to sample size a*i + b for i in range(subbands) == sampleSize')
            return None
        printi(f"Values per secend: {self.fs}")
        printi(f"Sample size: {self._sampleSize}")
        printi(f"Buffer size in sampls: {self._bufferSizeInSamples}")
        printi(f"Min Length of clip in Secends: {minLengthInSec}")
        printi(f'Use FFT: {self._useFFT}')
        printi(f'Use LowPassFilter: {self._useLowPassFilter}')
        printi(f'Use Auto Const from Var: {self._useAutoCFromVar}')
        printi(f'FFTSubbands count: {self._FFTSubbands}')
        if(self._useFFT):
            self.initFreqAnal(data)
        else:
            self.init(data)

    def padDataForSampleSize(self,data,value=0,axis=0):
        if len(data) % self._sampleSize == 0: return data
        padSize = self._sampleSize - len(data) % self._sampleSize
        printi(f"Pad Amount: {padSize}, Data Len: {len(data)}")
        return np.append(data,[value]*padSize, axis=axis)
    
    def f(self,i):
        return self._a*i + self._b

    def init(self,data):
        printi("Changing Type to int64")
        data = data.astype('int64')
        self._dataSq = data ** 2
        self._dataSq = self._dataSq[:,0] + self._dataSq[:,1]
        printi("Precalculated Sum of Squers")
        self._dataSq = self.padDataForSampleSize(self._dataSq)
        printi(f"Data padded: {self._dataSq.shape}")
        self._dataSq = np.sum(self._dataSq.reshape(-1,self._sampleSize),axis=1)
        printi("Summed up in sample size")
        printi(f"Data Square len: {len(self._dataSq)}")
        #self._avgEnergySq = np.average(self._dataSq[:self._bufferSizeInSamples])
        printi(f"Average Energy for sample: {self._avgEnergySq}")

        #self._avgVarEnergy = (np.average((self._dataSq[:self._bufferSizeInSamples] - self._avgEnergySq)**2))
        self._avgEnergySq.append(np.average(self._dataSq[:self._bufferSizeInSamples]))
        self._avgVarEnergy.append(np.average((self._dataSq[:self._bufferSizeInSamples] - self._avgEnergySq)**2))
        printi(f"Average Variance in Energy: {self._avgVarEnergy}")
        last = len(self._dataSq) - self._bufferSizeInSamples // 2 - self._bufferSizeInSamples % 2
        for i in range(self._bufferSizeInSamples//2,last):
            head = i + self._bufferSizeInSamples//2 + self._bufferSizeInSamples % 2 # for odd new element for avg is +1
            tail = i - self._bufferSizeInSamples//2
            self._avgEnergySq.append(self._avgEnergySq[-1] + (self._dataSq[head] - self._dataSq[tail])/self._bufferSizeInSamples)
            if self._useAutoCFromVar :
                self._avgVarEnergy.append(self._avgVarEnergy[-1] + ((self._dataSq[head] - self._avgEnergySq[-1]) ** 2 - (self._dataSq[tail] - self._avgEnergySq[-2])**2)/self._bufferSizeInSamples)
        if self._useAutoCFromVar:
            varMax = np.max(self._avgVarEnergy)
            varMin = np.min(self._avgVarEnergy)
            printi(f"varMax: {varMax}, varMin: {varMin}")
            self._avgVarEnergy = (self._avgVarEnergy - varMin)/(varMax - varMin) * 200

    def procesSample(self,i):
        '''
        i - is the middle index of avgEnergy (len(buf)//2)
        assumes i is in range
        '''
        C = (-0.0025714 * self._avgVarEnergy[i - self._bufferSizeInSamples//2]) + 1.5142857
        if not self._useAutoCFromVar: C = self._C
        out = self._dataSq[i] > C * self._avgEnergySq[i - self._bufferSizeInSamples//2]

        return out

    def initFreqAnal(self,data):
        '''
        sampleSize should be devisible by FFTSubbands
        '''
        data = data[:,0] + 1j * data[:,1]
        printi(f"Converted to complex numbers: {len(data)}, {data.shape} ")
        data = self.padDataForSampleSize(data)
        printi(f"Data padded: {data.shape}")
        data = data.reshape(-1,self._sampleSize)
        data = np.apply_along_axis(fft,1,data)
        printi(f"Calculated FFT in sample packs {len(data)}, {data.shape}")
        data = abs2(data)
        printi(f"Calculated Squear module of FFT: {len(data)}, {data.shape}")
        if self._useLowPassFilter:
            prev = 0
            for i in range(self._FFTSubbands):
                self._FFTSubbandsData = np.append(self._FFTSubbandsData, data[:,prev:self.f(i)].sum(axis=1) / self._sampleSize * self.f(i))
                prev = self.f(i)
            self._FFTSubbandsData = self._FFTSubbandsData.reshape(-1,len(data)).T
        else:
            self._FFTSubbandsData = np.sum(data.reshape(-1,self._FFTSubbands, self._sampleSize // self._FFTSubbands),axis=2) /self._sampleSize * self._FFTSubbands
        printi(f'Calculated Avg in subbands: {len(self._FFTSubbandsData)}, {self._FFTSubbandsData.shape}')

        self._FFTSubbandsAvgBuffs = np.append(self._FFTSubbandsAvgBuffs,np.average(self._FFTSubbandsData[:self._bufferSizeInSamples],axis=0))
        self._FFTSubbandsAvgBuffs = self._FFTSubbandsAvgBuffs.reshape(-1,len(self._FFTSubbandsAvgBuffs))

        self._FFTSubbandsAvgVar = np.append(self._FFTSubbandsAvgVar,(np.average((self._FFTSubbandsData[:self._bufferSizeInSamples] - self._FFTSubbandsAvgBuffs)**2,axis=0)))
        self._FFTSubbandsAvgVar = self._FFTSubbandsAvgVar.reshape(-1,len(self._FFTSubbandsAvgVar))

        last = len(self._FFTSubbandsData) - self._bufferSizeInSamples // 2 - self._bufferSizeInSamples % 2
        for i in range(self._bufferSizeInSamples//2,last):
            head = i + self._bufferSizeInSamples//2 + self._bufferSizeInSamples % 2 # for odd new element for avg is +1
            tail = i - self._bufferSizeInSamples//2
            self._FFTSubbandsAvgBuffs = np.append(self._FFTSubbandsAvgBuffs, [self._FFTSubbandsAvgBuffs[-1] + (self._FFTSubbandsData[head] - self._FFTSubbandsData[tail])/self._bufferSizeInSamples],axis=0)
            if self._useAutoCFromVar :
                varhead = (self._FFTSubbandsData[head] - self._FFTSubbandsAvgBuffs[-1]) ** 2
                vartail = (self._FFTSubbandsData[tail] - self._FFTSubbandsAvgBuffs[-2])**2
                self._FFTSubbandsAvgVar = np.append(self._FFTSubbandsAvgVar, [self._FFTSubbandsAvgVar[-1] + (varhead - vartail)/self._bufferSizeInSamples], axis=0)
        if self._useAutoCFromVar:
            varMax = np.max(self._FFTSubbandsAvgVar,axis=0)
            varMin = np.min(self._FFTSubbandsAvgVar,axis=0)
            printi(f"varMax: {varMax.shape}, varMin: {varMin.shape}")
            self._FFTSubbandsAvgVar = (self._FFTSubbandsAvgVar - varMin)/(varMax - varMin) * 200

        printi(f'Calculated Avg in buffer sampls: {len(self._FFTSubbandsAvgBuffs)}, {self._FFTSubbandsAvgBuffs.shape}')
        printi(f"Average Variance in Energy: {self._FFTSubbandsAvgVar.shape}")

        if self._showPlots:
            for i in range(self._FFTSubbands):
                self._pltsBeatsPerSub.append([])
                self._pltsSubband.append([])

    def processFFTSample(self,i):
        '''
        i - is the middle index of avgEnergy (len(buf)//2)
        assumes i is in range
        '''
        C = (-0.0025714*2 * self._FFTSubbandsAvgVar[i - self._bufferSizeInSamples//2]) + 1.5142857*2
        if not self._useAutoCFromVar: C = self._C
        out = np.sum(self._FFTSubbandsData[i] > C * self._FFTSubbandsAvgBuffs[i - self._bufferSizeInSamples//2]) > 0

        if self._showPlots:
            for k,band in enumerate( self._FFTSubbandsData[i]):
                if band > C * self._FFTSubbandsAvgBuffs[i - self._bufferSizeInSamples//2,k]:
                    self._pltsBeatsPerSub[k].append(i)
                self._pltsSubband[k].append(self._FFTSubbandsAvgBuffs[i - self._bufferSizeInSamples//2,k])

        return out


    def processAudio(self):
        beats = np.array([])
        data = self._FFTSubbandsData if self._useFFT else self._dataSq
        f_procesSample = self.processFFTSample if self._useFFT else self.procesSample

        last = len(data) - self._bufferSizeInSamples // 2 - self._bufferSizeInSamples % 2
        printi(f"Last sample to test: {last}")

        for i in range(self._bufferSizeInSamples//2,last):
            if(f_procesSample(i)):
                beats = np.append(beats,i) 
        printi(f"Found: {len(beats)} beats")
        beats = beats*self._sampleSize/self._fs

        beatsFiltered = []
        prev = 0
        for b in beats:
            if b - prev >= self._minLengthInSec:
                beatsFiltered.append(b)
                prev = b

        printi(f"Remaining beats after min len select: {len(beatsFiltered)}")
        if self._showPlots:
            limit = 1000
            if not self._useFFT:
                plt.plot(self._dataSq[self._bufferSizeInSamples//2:limit])
                plt.plot(self._avgEnergySq[:limit])
                if self._useAutoCFromVar:
                    plt.figure()
                    plt.plot(self._avgVarEnergy)
            else:
                plotLim = 12
                for i in range(0,min(plotLim,self._FFTSubbands),4):
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                    ax1.plot(self._pltsSubband[i][:limit])
                    for j in self._pltsBeatsPerSub[i]:
                        if j > limit :break
                        ax1.axvline(x=j)
                    ax2.plot(self._pltsSubband[i + 1][:limit])
                    for j in self._pltsBeatsPerSub[i + 1]:
                        if j > limit :break
                        ax2.axvline(x=j)
                    ax3.plot(self._pltsSubband[i + 2][:limit])
                    for j in self._pltsBeatsPerSub[i + 2]:
                        if j > limit :break
                        ax3.axvline(x=j)
                    ax4.plot(self._pltsSubband[i + 3][:limit])
                    for j in self._pltsBeatsPerSub[i + 3]:
                        if j > limit :break
                        ax4.axvline(x=j)
                    #plt.figure()


        return beatsFiltered

if __name__ == "__main__":
    musicFile = "E:\Filmy\Generator\Auto_PMV_Generator\Resources\TempMusic\Music Compilation Video - Sally.mp4"
    #musicFile = "E:\Filmy\Generator\Auto_PMV_Generator\Resources\TempMusic\\'Bass Slut' Teen Fucking PMV.mp4"
    #musicFile = "test.wav"
    # ap = AudioProcessor(musicFile, 1.6, useFFT=False, buffSizeInSec=4, sampleSize= 1024, debug=True,showPlots=True)
    ap = AudioProcessor(musicFile,10,useAutoConst=True, useFFT=True,useLowPassFilter=False, buffSizeInSec=4, sampleSize=1024,FFTSubbands=32,filterFuncPrams=(2,1), debug=True,showPlots=False)
    '''
    (a,b) 2,1 dla 1024 i 32
    (a,b) 2,33 dla 2048 i 32
    '''
    b = ap.processAudio()
    print(b[:10])
    print(b[-10:])
    print(len(b))

    plt.show()