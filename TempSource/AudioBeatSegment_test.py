import math
import pandas as pd
import numpy as np
from moviepy.audio.io.AudioFileClip import AudioFileClip
from pydub import AudioSegment
#import datetime

#2 channels, data in pars <channelA[0], channelB[0]> <channelA[1],channelB[1]> ...
def meanInBatches(data, batchSize):
	reshaped = np.mean(data[:(len(data)//batchSize)*batchSize].reshape(-1,batchSize), axis=1)
	return reshaped

def getRawData(sound):
	signed_data = np.frombuffer(sound._data, dtype=np.int16)
	raw_data = np.absolute(signed_data)
	return raw_data


def inputMusic(mp3_dir, granularity=0.05, nSections = 1, requiredDiff=20, minSections=8, mvAv=5):
	#audioclip = AudioFileClip(mp3_dir)
	sound = AudioSegment.from_file(mp3_dir, 'mp4')
	#sound = AudioSegment.from_file(mp3_dir, 'wav')
	bitrate = sound.channels * sound.frame_rate
	ratio = int(round(bitrate * granularity, 0))

	raw_data = getRawData(sound)
	batchsMean = meanInBatches(raw_data, ratio)

	#print(mvAv)
	selectedDiffsIndex = getMediaSectionsSplit(batchsMean, nSections, mvAv, granularity, False, requiredDiff, minSections)

	# plt.plot(np.frombuffer(sound._data,dtype=np.int16))
	# for i in selectedDiffsIndex:
	# 	plt.axvline(x=i)
	# plt.show()

	return selectedDiffsIndex, batchsMean #first_data,# audioclip, ratio, bitrate 

def moving_average(x, w):
	avNumpy = np.convolve(x, np.ones(w), 'valid')/w #sum 'w' elements and devide by 'w'
	return np.concatenate((np.zeros(w-1), avNumpy))

def checkIfNearInArray(valueIndex, value, selectedDiffsIndex, selectedDiffs, tolerance):
	for iSelectedDiff, selectedDiff in enumerate(selectedDiffs):
		index = selectedDiffsIndex[iSelectedDiff]
		difference = abs(index-valueIndex)
		if difference<tolerance:
			if selectedDiff>value:
				outPosition = iSelectedDiff
				outIndex = index
				outDiff = selectedDiff
				return outPosition, outIndex, outDiff
			else:
				outPosition = iSelectedDiff
				outIndex = valueIndex
				outDiff = value
				return outPosition, outIndex, outDiff
		# else:
	outPosition = len(selectedDiffs)-1
	outIndex = valueIndex
	outDiff = value
	return outPosition, outIndex, outDiff

def getMusicSections(musicData, granularity, nSections = 6, requiredDiffSeconds=250):
	mvAv = 50
	musicDataDiffAv = moving_average(musicData, mvAv)
	musicDataDiff = musicData - musicDataDiffAv
	# musicDataDiff = np.diff(musicData)
	musicDataDiff2 = musicDataDiff*musicDataDiff

	selectedDiffs = [0]*nSections
	selectedDiffsIndex = [0]*nSections
	for iDiff, diff in enumerate(musicDataDiff2):
		if diff>min(selectedDiffs) and iDiff>mvAv:
			outPosition, outIndex, outDiff = checkIfNearInArray(iDiff, diff, selectedDiffsIndex, selectedDiffs, int(requiredDiffSeconds/granularity))
			# iDiff, diff, overwriteBool = checkIfNearInArray(iDiff, diff, selectedDiffsIndex, selectedDiffs, 50)
			selectedDiffs[outPosition] = outDiff
			selectedDiffs.sort(reverse=True)
			for iSelectedDiff, selectedDiff in enumerate(selectedDiffs):
				selection = np.where(musicDataDiff2 == selectedDiff)
				try:
					point = selection[0][0]
				except:
					# print(selection)
					continue
					# pass
				selectedDiffsIndex[iSelectedDiff] = point

	# selectedDiffsIndex = []
	# for selectedDiff in selectedDiffs:
	#	  point = np.where(musicDataDiff2 == selectedDiff)[0][0]
	#	  selectedDiffsIndex.append(point)

	points=[]
	for index in selectedDiffsIndex:
		points.append(musicData[index])


	return musicDataDiff2, points, selectedDiffs, selectedDiffsIndex

def getMediaSectionsSplit(batchsMean, nSections, mvAv, granularity, plotCharts, MinClipLenInSec, minSections):

	iSection = 0
	startIndex = 0
	sectionLen = int(len(batchsMean)/nSections)
	selectedDiffsIndex = []
	musicDataDiffAvDiff = []
	musicDataDiffAvBack = []
	musicDataDiffAvBackLonger = []
	musicDataDiffAvBackDeltas = []
	minLineList = []
	maxLineList = []
	while iSection < nSections:
		#print(iSection)
		batchsMeanSelect = batchsMean[startIndex:startIndex + sectionLen]
		selectedDiffsIndex_section, musicDataDiffAvDiff_section, musicDataDiffAvBack_section, musicDataDiffAvBackLonger_section,\
		sdDiff = getMediaSections(batchsMeanSelect, 3, mvAv, granularity, minSections)

		selectedDiffsIndex.extend([x+startIndex for x in selectedDiffsIndex_section]) #np.array() + startIndex

		# musicDataDiffAvDiff += musicDataDiffAvDiff_section.tolist()
		# musicDataDiffAvBack += musicDataDiffAvBack_section.tolist()
		# musicDataDiffAvBackLonger += musicDataDiffAvBackLonger_section.tolist()
		# musicDataDiffAvBackDeltas += musicDataDiffAvBackDeltas_section.tolist()

		# minLineList += [meanDiff - sdDiff for i in musicDataDiffAvDiff_section.tolist()]
		# maxLineList += [meanDiff + sdDiff for i in musicDataDiffAvDiff_section.tolist()]

		startIndex += sectionLen
		iSection += 1


	#remove Duplicates
	newSectionsIndex = list(dict.fromkeys(selectedDiffsIndex))
	newSectionsIndexFiltered = []
	for i,section in enumerate(newSectionsIndex):
		if i == 0 or section-newSectionsIndexFiltered[-1] > int(MinClipLenInSec/granularity):
			newSectionsIndexFiltered.append(section)

	# if plotCharts:
	# 	plt.plot(musicDataDiffAvDiff)
	# 	plt.plot(musicDataDiffAvBack)
	# 	plt.plot(musicDataDiffAvBackLonger)
	# 	plt.plot(musicDataDiffAvBackDeltas)
	# 	for i in newSectionsIndexFiltered:
	# 		plt.axvline(x=i)
	# 	plt.plot(minLineList)
	# 	plt.plot(maxLineList)
	# 	# plt.axhline(y=meanDiff + sdDiff)
	# 	# plt.axhline(y=meanDiff - sdDiff)
	# 	plt.show()



	return newSectionsIndexFiltered


def getMediaSections(batchsMean, minLength, mvAv, granularity, minSections):
	# print(mvAv, minLength)
	mvAv = int(mvAv / granularity)
	minLength = minLength / granularity

	dataMovAvg = moving_average(batchsMean, mvAv)
	#dataMovAvgDeltas = np.concatenate((np.zeros(1), np.diff(dataMovAvg)))
	# musicDataDiffAvBackDeltasAv = moving_average(musicDataDiffAvBackDeltas, 3)
	dataMovAvgOfMovAvgSmall = moving_average(dataMovAvg, int(mvAv/2))
	dataMovAvgOfMovAvgBig = moving_average(dataMovAvg, int(mvAv*2))

	dataDiffSurraoundBig = dataMovAvg - dataMovAvgOfMovAvgBig
	dataDiffSurraoundSmal = dataMovAvg - dataMovAvgOfMovAvgSmall

	sdDiff = dataDiffSurraoundBig.std()*3

	newSectionsIndex = []

	# print(meanDiff, sdDiff, len(musicDataDiffAvDiff), 0)
	maxSection = 0
	iMaxSections = 0
	while len(newSectionsIndex)<minSections and sdDiff > 0.1:
		newSectionsIndex = []

		for i,diff in enumerate(dataDiffSurraoundSmal):
			if not( -sdDiff < diff < sdDiff):
				newSectionsIndex.append(goBackToCrossingPoint(dataDiffSurraoundSmal, i, granularity))

		if len(newSectionsIndex) > maxSection:
			maxSection = len(newSectionsIndex)
			iMaxSections = sdDiff
		print(len(newSectionsIndex), sdDiff)
		sdDiff *= 0.9
	print(newSectionsIndex)
	if len(newSectionsIndex) < minSections:
		sdDiff = iMaxSections
		for i,diff in enumerate(dataDiffSurraoundSmal):
			if  -sdDiff < diff < sdDiff:
				newSectionsIndex.append(goBackToCrossingPoint(dataDiffSurraoundSmal, i, granularity))

	#return newSectionsIndex, dataDiffSurraoundSmal, dataMovAvg, dataMovAvgOfMovAvgSmall, dataMovAvgDeltas, meanDiff, sdDiff
	return newSectionsIndex, dataDiffSurraoundSmal, dataMovAvg, dataMovAvgOfMovAvgSmall, sdDiff

def sign(x):
	return math.copysign(1,x)

def goBackToCrossingPoint(musicDataDiffAvDiff, iDiff, granularity):
	backSecondsSearch = 30
	dataValueStartSign = sign(musicDataDiffAvDiff[iDiff])
	for i in range(iDiff,max(0,iDiff-int(backSecondsSearch/granularity)),-1):
		dataValueSign = sign(musicDataDiffAvDiff[i])
		if (dataValueStartSign != dataValueSign): return i
	return iDiff



if __name__ == "__main__":
	import matplotlib.pyplot as plt
	musicFile = "E:\Filmy\Generator\Auto_PMV_Generator\Resources\TempMusic\Music Compilation Video - Sally.mp4"
	#musicFile = "test.wav"
	musicData = inputMusic(musicFile, granularity=0.010, nSections = 9, requiredDiff=0.1, minSections=8)
	
	#reshaped_data, first_data, audioclip, ratio, bitrate, songStart, songEnd, selectedDiffsIndex
	print(len(musicData[0]))
	print(musicData[0])


#
# time1=datetime.datetime.now()
# chroma, a, b, _ = pychorus.create_chroma(musicFile)
# plt.imshow(chroma)
# plt.show()
# time_time_similarity = compute_similarity_matrix_slow(chroma)
# time2=datetime.datetime.now()
# print(time2-time1)
#
# plt.imshow(time_time_similarity)
# plt.show()