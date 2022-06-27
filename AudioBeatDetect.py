from functools import lru_cache
from matplotlib import pyplot as plt
from scipy.io import wavfile # scipy library to read wav files
import numpy as np
import os
from MyLogger import printe,printw,printi,setLevel
import subprocess
from numpy.fft import fft, ifft
import numba as nb
from numba import float64 as f64,float32 as f32,int32 as i32,int64 as i64
import time 


@nb.vectorize([nb.float64(nb.complex128),nb.float32(nb.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

# _spec = [
#     (f64[:,:],
#     f64[:,:],
#     nb.types.int32,
#     nb.types.int32,
#     nb.types.int32,
#     nb.types.int32,
#     f64[:])
# ]

# #@numba.vectorize([f64(numba.types.Array(numba.types.float64, 2, 'C'),numba.types.Array(numba.types.float64, 2, 'C'),i32,i32,i32,i32,numba.types.Array(numba.types.float64, 1, 'C'))])
# @nb.guvectorize(_spec,'(x,m),(n,m),(),(),(),()->(n)')
# def movingVar2(data,avg,n,l,k,navg,out):
#     for i in range(navg):
#         d = 0
#         for j in range(n):
#             d += (data[i+l+j,k] - avg[i,k])**2
#         out[i] = d/n 
#     #return out
@nb.njit(fastmath=True)
def movingVar(data,avg,n,l,k,navg,out):
    for i in range(navg):
        d =  np.sum((data[i+l:i+l+n,k] - avg[i,k])**2)
        out[i] = d/n 
    return out


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
    _pltsC = []
    _pltsAvgVar = []

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
        if self._useFFT:
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

    @lru_cache(maxsize=None)
    def filter(self):
        '''
        2x for low freq, 0.25x dla high
        '''
        b = 1.5
        a = (0.2 - b)/self._FFTSubbands
        return [f*a + b for f in range(self._FFTSubbands)]

    def init(self,data):
        printi("Changing Type to int64")
        data = data.astype('int64')
        self._dataSq = data ** 2
        self._dataSq = (self._dataSq[:,0] + self._dataSq[:,1]) / float(0xffffffff)
        printi(f"Precalculated Sum of Squers, min: {np.min(self._dataSq)}, max: {np.max(self._dataSq)}")
        self._dataSq = self.padDataForSampleSize(self._dataSq)
        printi(f"Data padded: {self._dataSq.shape}")
        self._dataSq = np.sum(self._dataSq.reshape(-1,self._sampleSize),axis=1)
        printi("Summed up in sample size")
        printi(f"Data Square len: {len(self._dataSq)}, {self._dataSq.shape}")
        
        self._avgEnergySq = np.convolve(self._dataSq,np.ones(self._bufferSizeInSamples),'valid')/self._bufferSizeInSamples
        printi(f"Average Energy for sample: {self._avgEnergySq[0]}, {self._avgEnergySq.shape}")
        if self._useAutoCFromVar:
            l = (self._bufferSizeInSamples - 1)//2
            self._avgVarEnergy = [np.average((self._dataSq[i+l:i+l+self._bufferSizeInSamples] - self._avgEnergySq[i])**2) for i in range(len(self._avgEnergySq))]
            max = np.max(self._avgVarEnergy)
            self._avgVarEnergy = self._avgVarEnergy / max * 200
            printi(f"Average Variance Energy for sample: {self._avgVarEnergy[0]},{np.shape(self._avgVarEnergy)}")

    def procesSample(self,i):
        '''
        i + self._bufferSizeInSamples//2 is the middle index of avgEnergy (len(buf)//2)
        assumes i is in range of avgEnergy
        '''
        C = self._C
        if self._useAutoCFromVar:
            C = max(1,(-0.0025714 * self._avgVarEnergy[i]) + 1.5142857)
        out = self._dataSq[i+self._bufferSizeInSamples//2] > C * self._avgEnergySq[i]

        return (out, self._dataSq[i+self._bufferSizeInSamples//2])

    def initFreqAnal(self,data):
        '''
        sampleSize should be devisible by FFTSubbands 
        '''
        t = time.time()
        data = data[:,0] + 1j * data[:,1]
        printi(f"Converted to complex numbers: {len(data)}, {data.shape}, time: {time.time() - t} ")
        t = time.time()
        data = self.padDataForSampleSize(data)
        printi(f"Data padded: {data.shape}, time: {t-time.time()}")
        t = time.time()
        data = data.reshape(-1,self._sampleSize)
        data = np.apply_along_axis(fft,1,data)
        printi(f"Calculated FFT in sample packs {len(data)}, {data.shape}, time: {time.time() - t}")
        t = time.time()
        data = abs2(data)
        printi(f"Calculated Squear module of FFT: {len(data)}, {data.shape}, time: {time.time() - t}")
        t = time.time()
        if self._useLowPassFilter:
            prev = 0
            self._FFTSubbandsData = np.empty((len(data),self._FFTSubbands))
            for i in range(self._FFTSubbands):
                self._FFTSubbandsData[:,i] =  data[:,prev:self.f(i)].sum(axis=1) / self._sampleSize * self.f(i)
                prev = self.f(i)
        else:
            self._FFTSubbandsData = np.sum(data.reshape(-1,self._FFTSubbands, self._sampleSize // self._FFTSubbands),axis=2) /self._sampleSize * self._FFTSubbands
        printi(f'Calculated Avg in subbands: {len(self._FFTSubbandsData)}, {self._FFTSubbandsData.shape}, time: {time.time()-t}')

        t = time.time()
        self._FFTSubbandsAvgBuffs = np.apply_along_axis(lambda m: np.convolve(m,np.ones(self._bufferSizeInSamples),'valid')/self._bufferSizeInSamples, axis=0, arr=self._FFTSubbandsData)
        printi(f'Calculated Avg in buffer sampls: {self._FFTSubbandsAvgBuffs.shape}, time: {time.time()-t}')
        if self._useAutoCFromVar:

            l = (self._bufferSizeInSamples - 1)//2
            self._FFTSubbandsAvgVar = np.empty((len(self._FFTSubbandsAvgBuffs),self._FFTSubbands))
            for k in range(self._FFTSubbands):
                #movingVar2(self._FFTSubbandsData,self._FFTSubbandsAvgBuffs,self._bufferSizeInSamples,l,k,len(self._FFTSubbandsAvgBuffs),self._FFTSubbandsAvgVar[:,k])
                self._FFTSubbandsAvgVar[:,k] = movingVar(self._FFTSubbandsData,self._FFTSubbandsAvgBuffs,self._bufferSizeInSamples,l,k,len(self._FFTSubbandsAvgBuffs),self._FFTSubbandsAvgVar[:,k])
                # = [np.average((self._FFTSubbandsData[i+l:i+l+self._bufferSizeInSamples,k] - self._FFTSubbandsAvgBuffs[i,k])**2) for i in range(len(self._FFTSubbandsAvgBuffs))]
            max = np.max(self._FFTSubbandsAvgVar,axis=0)
            self._FFTSubbandsAvgVar = self._FFTSubbandsAvgVar / max * 200
            printi(f"Average Variance Energy for sample: {np.shape(self._FFTSubbandsAvgVar)}, time: {time.time()-t}")

            # t = time.time()
            # self._FFTSubbandsAvgVar = np.empty((len(self._FFTSubbandsAvgBuffs),self._FFTSubbands))
            # self._FFTSubbandsAvgVar[0] = np.average((self._FFTSubbandsData[:self._bufferSizeInSamples] - self._FFTSubbandsAvgBuffs[0])**2,axis=0)
            # for i in range(1,len(self._FFTSubbandsAvgBuffs)):
            #     if self._useAutoCFromVar :
            #         varhead = (self._FFTSubbandsData[i + self._bufferSizeInSamples//2] - self._FFTSubbandsAvgBuffs[i]) ** 2
            #         vartail = (self._FFTSubbandsData[i-1] - self._FFTSubbandsAvgBuffs[i-1])**2
            #         self._FFTSubbandsAvgVar[i] = self._FFTSubbandsAvgVar[-1] + (varhead - vartail)/self._bufferSizeInSamples
            # if self._useAutoCFromVar:
            #     varMax = np.max(self._FFTSubbandsAvgVar,axis=0)
            #     printi(f"varMax: {varMax}, varMin: {np.min(self._FFTSubbandsAvgVar)}")
            #     self._FFTSubbandsAvgVar = self._FFTSubbandsAvgVar/varMax * 200

            printi(f"Average Variance in Energy: {self._FFTSubbandsAvgVar.shape}, time: {time.time()-t}")

        if self._showPlots:
            for i in range(self._FFTSubbands):
                self._pltsBeatsPerSub.append([])
                self._pltsSubband.append([])
                self._pltsC.append([])
                self._pltsAvgVar.append([])

    def processFFTSample(self,i):
        '''
        i - is the middle index of avgEnergy (len(buf)//2)
        assumes i is in range
        '''
        threshold = 10e8
        C = self._C
        if self._useAutoCFromVar:
            C = (-0.0025714*4 * self._FFTSubbandsAvgVar[i]) + 1.5142857*4
        vals = self._FFTSubbandsData[i + self._bufferSizeInSamples//2] * self.filter()
        out = np.any((vals > C * self._FFTSubbandsAvgBuffs[i])
             & (self._FFTSubbandsAvgBuffs[i] > threshold))
        if self._showPlots:
            for k,band in enumerate( self._FFTSubbandsData[i + self._bufferSizeInSamples//2]):
                if self._useAutoCFromVar:
                    if band > C[k] * self._FFTSubbandsAvgBuffs[i,k]:
                        self._pltsBeatsPerSub[k].append(i)
                    self._pltsC[k].append(C[k])
                else:
                    if band > C * self._FFTSubbandsAvgBuffs[i,k]:
                        self._pltsBeatsPerSub[k].append(i)
                self._pltsSubband[k].append(self._FFTSubbandsAvgBuffs[i,k])
                self._pltsAvgVar[k].append(self._FFTSubbandsAvgVar[i,k])

        return (out,np.max(vals))

    def maxInDist(self,beats,i,dist):
        r=i+1
        while(r < len(beats) and beats[r,0] - beats[i,0] <= dist): r += 1
        return (i + np.argmax(beats[i:r,1]),r)

    def processAudio(self):
        beats = np.array([])

        data = self._FFTSubbandsAvgBuffs if self._useFFT else self._avgEnergySq
        f_procesSample = self.processFFTSample if self._useFFT else self.procesSample

        for i in range(len(data)):
            b,v = f_procesSample(i)
            if b: beats = (np.row_stack((beats,[i,v]))) if len(beats) > 0 else [i,v] 
        printi(f"Found: {len(beats)} beats, {beats.shape}")
        beats = beats*self._sampleSize/self._fs

        beatsFiltered = []
        i = 0; r = 0
        while(i <= r < len(beats)):
            i, r = self.maxInDist(beats,i,self._minLengthInSec)
            beatsFiltered.append(beats[i][0])
            i = r

        printi(f"Remaining beats after first min len select: {len(beatsFiltered)}")
        beatsFiltered2 = []
        prev = 0
        for b in beatsFiltered:
            if b - prev >= self._minLengthInSec:
                beatsFiltered2.append(b)
                prev = b
        printi(f"Remaining beats after second min len select: {len(beatsFiltered2)}")

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
                    v = 3 if self._useAutoCFromVar else 2
                    fig, axis = plt.subplots(v, 2)

                    axis[0,0].plot(self._pltsSubband[i][:limit])
                    for j in self._pltsBeatsPerSub[i]:
                        if j > limit :break
                        axis[0,0].axvline(x=j)
                    axis[0,1].plot(self._pltsSubband[i + 1][:limit])
                    for j in self._pltsBeatsPerSub[i + 1]:
                        if j > limit :break
                        axis[0,1].axvline(x=j)
                    if not self._useAutoCFromVar:
                        axis[1,0].plot(self._pltsSubband[i + 2][:limit])
                        for j in self._pltsBeatsPerSub[i + 2]:
                            if j > limit :break
                            axis[1,0].axvline(x=j)
                        axis[1,1].plot(self._pltsSubband[i + 3][:limit])
                        for j in self._pltsBeatsPerSub[i + 3]:
                            if j > limit :break
                            axis[1,1].axvline(x=j)
                    else:
                        axis[1,0].plot(self._pltsAvgVar[i][:limit])
                        axis[1,1].plot(self._pltsAvgVar[i+1][:limit])
                        axis[2,0].plot(self._pltsC[i][:limit])
                        axis[2,1].plot(self._pltsC[i + 1][:limit])
                    #plt.figure()


        return beatsFiltered2

def progres_bar(progress, total):
	precent = 100 * (progress / float(total))
	bar = '█' * int(precent) + '-' * int(100 - precent)
	print(f"\r|{bar}| {precent:.2f}%", end="\r")

if __name__ == "__main__":
    #musicFile = "E:\Filmy\Generator\Auto_PMV_Generator\Resources\TempMusic\Music Compilation Video - Sally.mp4"
    musicFile = "E:\Filmy\Generator\Auto_PMV_Generator\Resources\TempMusic\\'Bass Slut' Teen Fucking PMV.mp4"
    #musicFile = "tests.wav"
    # ap = AudioProcessor(musicFile, 1.3, useAutoConst=False, useFFT=False, buffSizeInSec=0.8, sampleSize= 1024, debug=True,showPlots=False)
    ap = AudioProcessor(musicFile,10,useAutoConst=False, useFFT=True,useLowPassFilter=True,minLengthInSec=0.15, buffSizeInSec=0.8, sampleSize=1024,FFTSubbands=32,filterFuncPrams=(2,1), debug=True,showPlots=False)
    #ap = AudioProcessor(musicFile,minLengthInSec=0.2, buffSizeInSec=0.8,debug=True)
    '''
    (a,b) 2,1 dla 1024 i 32
    (a,b) 2,33 dla 2048 i 32
    '''
    b = ap.processAudio()
    print(b[:10])
    print(b[-10:])
    print(len(b))

    plt.show()