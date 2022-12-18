from cgi import test
import os
import random
import shutil
import subprocess
import time
from sqlite3 import connect

from subprocess import Popen
from multiprocessing.pool import Pool

import pandas as pd
import bisect 
import copy

from AudioBeatDetect import AudioProcessor 
from MyLogger import printe,printw,printi,printv,setLevel

conn = connect("./Classified_Videos.sqlite")

_debug = False

# files = [
# "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4",
# "E:\Filmy\InTheCrack\InTheCrack 1497 Demi Sutra-40456.mp4",
# "E:\Filmy\InTheCrack\InTheCrack 1537 Demi Sutra-45361.mp4",
# "E:\Filmy\InTheCrack\InTheCrack 1674 Demi Sutra-62912.mp4",
# "E:\Filmy\InTheCrack\InTheCrack 1702 Demi Sutra-66098.mp4",
# "E:\Filmy\InTheCrack\InTheCrack 1113 Skin Diamond-60622.mp4",
# "E:\Filmy\InTheCrack\InTheCrack 1135 Skin Diamond-61328.mp4",
# "E:\Filmy\InTheCrack\InTheCrack 1180 Skin Diamond-62537.mp4",
# ]

folders = ['E:\Filmy\.Janice Griffith\Select']
excludeFiles = []
files = []

# music = "E:\Filmy\Generator\Auto_PMV_Generator\Resources\TempMusic\Music Compilation Video - Sally.mp4"
#music = "E:\Filmy\Generator\Auto_PMV_Generator\Resources\TempMusic\\'Bass Slut' Teen Fucking PMV.mp4"
#music = "E:\Filmy\Generator\Auto_PMV_Generator\Resources\TempMusic\Ida Corr vs Fedde Le Grand Let Me Think About It (Extended).mp3"
music = "E:\Filmy\Generator\Auto_PMV_Generator\Resources\TempMusic\Fedde Le Grand and Ida Corr - Let Me Think About It (Celebration Mix) [Official Music Video].mp3"

catsToLoad = [{'cat': 'other', 'tresh': '0.1'},
		{'cat': 'cunnilingus', 'tresh': '0.4'},
		{'cat': 'blowjob_handjob', 'tresh': '0.1'},
		{'cat': 'sex_back', 'tresh': '0.1'},
		{'cat': 'sex_front', 'tresh': '0.1'},
		{'cat': 'titfuck', 'tresh': '0.1'}
		]

#desc = {'cunnilingus': 1}
desc = {'cunnilingus': 10, 'blowjob_handjob': 10, 'sex_front': 5, 'titfuck': 1}
desc = {'cunnilingus': 10}

resolution = "1920x1080"
crf = 20
compresion_preset = "fast"
#fast, medium, slow, FOR GPU
#ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow FOR CPU

output_file = "janice-pussy2-pmv.mp4"

res_comm = resolution.replace('x',':')
#vid_parts = []
#{'v':0,'s':60,'t':10},

maxThreads = 24

vid_names = []
vid_res = []
musicData = None

vid_data = [] # ['id':id, 'name':name, 'cats' [{'cat':name,'tresh':val,'info': dataFrame}]]
last_vid_used = []
cat_segments = {}

def init():
	global vid_names
	global vid_res
	global files, musicData
	
	for dir in folders:
		for dirpath, dirnames, filenames in os.walk(dir):
			for f in filenames:
				if f not in excludeFiles:
					files.append(os.path.join(dir,f))
			break 
	printv(files)
	get_size_command = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "
	for file in files:
		res = os.popen(f'{get_size_command}"{file}"')
		vid_res.append(res.read())
	
	shutil.rmtree("./tmp")
	os.makedirs("./tmp")
	
	# Ap = AudioProcessor(music, 1.6, useAutoConst=True, useFFT=False, buffSizeInSec=4, sampleSize= 1024, debug=True)
	# Ap = AudioProcessor(music,5,useAutoConst=True, useFFT=True,useLowPassFilter=True, minLengthInSec=0.2 ,buffSizeInSec=0.6, sampleSize=1024,FFTSubbands=32,filterFuncPrams=(2,1), debug=True)
	Ap = AudioProcessor(music,4,minLengthInSec=0.3, buffSizeInSec=2,debug=True)
	musicData = Ap.processAudio()

###=====================Clip Cutting=========================###
#------------------------processes-----------------------------# vidPart{'v':0,'s':60,'d':10,'i': 0},
def getRunningProcesses(proc):
	count = 0
	for p in proc:
		if p.poll() is None:
			count += 1
	return count

def areRunningProcesses(proc):
	for p in proc:
		if p.poll() is None:
			return True
	return False

def startGPUProcess(vidPart):
	#comm = f'ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -ss {tstart} -i "{files[vid_id]}"'
	comm = f'ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -ss {vidPart["s"]} -i "{files[vidPart["v"]]}"'
	if resolution not in vid_res[vidPart["v"]]:
		printi(f"scaling {vidPart['v']}")
		comm += f' -vf scale_npp={res_comm}:force_original_aspect_ratio=decrease,pad={res_comm}:(ow-iw)/2:(oh-ih)/2'
	comm += f' -threads 1 -crf {crf} -t {vidPart["d"]} -c:v h264_nvenc -preset {compresion_preset} tmp/v{vidPart["i"]}.mp4'
	print(comm)
	subprocess.run(comm,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def CPUProcess(vidPart):
	global vid_res
	comm = f'ffmpeg -y -ss {vidPart["s"]} -i "{files[vidPart["v"]]}"'
	if resolution not in vid_res[vidPart["v"]]:
		comm += f' -vf "scale={res_comm}:force_original_aspect_ratio=decrease,pad={res_comm}:(ow-iw)/2:(oh-ih)/2"'
	comm += f' -crf {crf} -t {vidPart["d"]} -preset {compresion_preset} -r 30 tmp/v{vidPart["i"]}.mp4'
	print(comm)
	return comm

def taskCPUProcess(vidPart):
	comm = CPUProcess(vidPart)
	subprocess.run(comm,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def startCPUProcess(vidPart, processes):
	comm = CPUProcess(vidPart)
	processes.append(Popen(comm, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
#-----------------------vid cutters----------------------------#
def get_vid_partsGPU(vidParts):	
	for i,p in enumerate(vidParts):
		startGPUProcess(p)
		progres_bar(i ,len(vidParts))

def get_vid_partsCPU(vidParts):	
	compleatedThreads = 0
	processes = []
	i = 0
	while( i < len(vidParts)):
		if(compleatedThreads + maxThreads > i):
			startCPUProcess(vidParts[i],processes)
			progres_bar(i ,len(vidParts))
			i += 1
		else:
			for p in processes: p.wait()
			compleatedThreads += len(processes)
			processes = []		
	for p in processes: p.wait()

def get_vid_partsCPUPool(vidParts):	
	with Pool(maxThreads) as pool:
		result = pool.map(taskCPUProcess, vidParts)
		#pool.wait()
		pool.close()
		pool.join()
		print("pool finished")

def get_vid_parts(vidParts):	
	compleatedThreads = 0
	processes = []
	i = 0
	j = len(vidParts) - 1
	while( i < j):
		if(compleatedThreads + maxThreads > i):
			startCPUProcess(vidParts[i],processes)
			progres_bar(i + len(vidParts) - 1 - j,len(vidParts))
			i += 1
		else:
			if areRunningProcesses(processes):
				startGPUProcess(vidParts[j])
				progres_bar(i + len(vidParts) - 1 - j,len(vidParts))
				j -=1
			else:
				compleatedThreads += len(processes)
				processes = []		
	print("i: ", i, " j: ",len(vidParts) - 1 - j)
#------------------------join video----------------------------#
def join_vids():
	if os.path.exists(output_file): os.remove(output_file)
	if _debug:
		subprocess.run('ffmpeg -f concat -i ./tmp/list.txt -c copy ./tmp/out.mp4')
	else:
		subprocess.run('ffmpeg -f concat -i ./tmp/list.txt -c copy ./tmp/out.mp4',stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

	if _debug:
		subprocess.run(f'ffmpeg -i ./tmp/out.mp4 -i "{music}" -c copy -shortest -map 0:v:0 -map 1:a:0 {output_file}')
	else:
		subprocess.run(f'ffmpeg -i ./tmp/out.mp4 -i "{music}" -c copy -shortest -map 0:v:0 -map 1:a:0 {output_file}',stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
	# -stream_loop -1
	progres_bar(10,10)
	print('\n')

###==================Get Video SQL Data======================###
def calSimilarBlocks(df):
	firstRowGard = pd.DataFrame({'second':-10},index =[0])
	lastRowGard = pd.Series([df.second.iloc[-1]])
	df = pd.concat([firstRowGard, df[:]]).reset_index(drop=True) 
	df['dif'] = df.diff(1)
	df = df[df['dif'] > 1]
	df['lastSec'] = 0
	df.iloc[:-1,-1] = df[1:].second - df[1:].dif
	df.iloc[-1,-1] = lastRowGard
	df['len'] = df.lastSec - df.second + 1
	df = df[['second','lastSec','len']]
	return df

def findId(name):
	global files
	for i,f in enumerate(files):
		if name in f: return i
	return -1 

def get_Sql_data(categorys):
	global files, vid_data
	sqlFiles = ""
	for f in files:
		fName = os.path.splitext(os.path.basename(f))
		sqlFiles += f"'{fName[0]}'," 
	sqlFiles = sqlFiles[:-1]
	keys = pd.read_sql(f'SELECT classifier_id,key FROM Keys WHERE key in ({sqlFiles})', conn)
	keys.columns = ['id', 'name']
	vid_data = keys.to_dict(orient='records')
	for v in vid_data:
		v['fID'] = findId(v['name'])

	printv(vid_data)

	for vid in vid_data:
		cats = copy.deepcopy(categorys)
		for cat in cats:
			df = pd.read_sql(f'SELECT second FROM Data WHERE classifier_id = {vid["id"]} AND {cat["cat"]} >= {cat["tresh"]} ORDER BY second ASC', conn)
			cat['info'] = calSimilarBlocks(df)
		vid['cats'] = cats

		sqlRequest = f'SELECT second FROM Data WHERE classifier_id = {vid["id"]} '
		for cat in cats:
			sqlRequest += f'AND {cat["cat"]} <= {cat["tresh"]} '
		sqlRequest += 'ORDER BY second ASC'
		df = pd.read_sql(sqlRequest, conn)
		if(len(df) > 0):
			vid['cats'].append({'cat': 'none', 'tresh': '0', 'info': calSimilarBlocks(df)})

###==================Gen Video sequenc=======================###
def generate_simple_vide_parts(minStartSkip):# how much start of the vid to skip  
	global files, musicData
	parts = []
	vid_id = 0
	prevs = 0

	for i, s in enumerate(musicData):
		parts.append({'v':vid_id, 's':minStartSkip + s,'d':s-prevs,'i':i})
		prevs = s
		vid_id += 1
		vid_id %= len(files)

	vlist = open("tmp/list.txt","w")
	for j in range(len(musicData)): vlist.write(f'file v{j}.mp4 \n')
	vlist.close()

	return parts

def generate_category_sequenc(desc): #{'<catname0>':<len>} define pattern to repeat till end 
	# to comply with beats it will change patter rounding len of cat segment to beat len
	global musicData, cat_segments
	#l = desc['totalLen']
	mIndex = 0
	prev = 0
	vidseq = []
	while(mIndex < len(musicData)):
		for cat in desc.keys():
			#if(cat == 'totalLen'): continue
			catLen = desc[cat]
			while(catLen > 0 and mIndex < len(musicData)):
				s = musicData[mIndex] # beatTime
				mIndex += 1
				l = s - prev
				prev = s
				catLen -= l
				vidseq.append({'cat':cat, 'len':l})
	return vidseq

def get_vid_id_seq(len):
	pass

def init_select_vid_random(cats):
	global cat_segments, vid_data #['id':id, 'name':name, 'cats' [{'cat':name,'tresh':val,'info': dataFrame}]]
	for cat in cats:
		cat_segments[cat['cat']] = pd.DataFrame(columns=['second', 'lastSec', 'len', 'id'])
	cat_segments['none'] = pd.DataFrame(columns=['second', 'lastSec', 'len', 'id'])

	for vid in vid_data:
		for cat in vid['cats']:
			df = cat['info']
			df['id'] = vid['fID']
			cat_segments[cat['cat']] = pd.concat([cat_segments[cat['cat']],cat['info']],ignore_index=True)

	for catInfo in cat_segments.values():
		catInfo.sort_values(by=['len'], inplace=True, ascending=False)
		catInfo.reset_index(drop=True)
	
	keys = list(cat_segments.keys())
	for key in keys:
		cat_segments[key+'Len'] = cat_segments[key]['len'].sum()
	
def select_vid_random(minStartSkip,cat,l):# wszystkie id spelnia wymagania kategorii plus random
	global cat_segments
	parts=[]
	# per category array sorted by len
	df = cat_segments[cat]
	while l > 0:
		index = (df['len'][(df['len'] >= l) & (df['second'] > minStartSkip)]).count()
		if index == 0: # nic nie spelnia wymagań
			if df.iloc[0]['len'] > 0: # choose first
				rng = 0
				d = df.iloc[0]['len']
			else:
				print(f"error: not enough segments in category:{cat}")
				parts.append({'v':id,'s':stime,'d':l})
				return parts
		else: #choose random
			rng = random.randrange(0,index)
			d = l

		id = df.iloc[rng]['id']
		stime = df.iloc[rng]['second']
		parts.append({'v':id,'s':stime,'d':d})
		df.iloc[rng]['len'] -= d
		l -= d
		cat_segments[cat+'Len'] -= d
		df.sort_values(by=['len'],inplace=True,ascending=False, ignore_index=True, kind='mergesort')
		
	return parts

def select_vid_last_used(minStartSkip,cat,l): # id najdawniej uzyte i spelnia wymagania kategorii
	global last_vid_used
	id =0; l = 0
	# id najdawniej uzyte i spelnia wymagania kategorii
	#vidIdSeq = get_vid_id_seq(len(vidseq))
	return id, l

def select_vid_clip_from_seq(minStartSkip, vidseq, vidSelector): # vidseq = {cat, len}
	global vid_data,last_vid_used
	parts = []
	i = 0
	for seq in vidseq:
		selected = vidSelector(minStartSkip,seq['cat'],seq['len'])
		for s in selected:
			parts.append({'v':s['v'],'s':s['s'],'d':s['d'],'i': i})
			i += 1

	vlist = open("tmp/list.txt","w")
	for p in parts: vlist.write(f'file v{p["i"]}.mp4 \n')
	vlist.close()
	return parts


###=========================Other============================###
def progres_bar(progress, total):
	precent = 100 * (progress / float(total))
	bar = '█' * int(precent) + '-' * int(100 - precent)
	print(f"\r|{bar}| {precent:.2f}%", end="\r")

if __name__ == "__main__":
	init()
	startTotal = time.time()
	start = time.time()
	get_Sql_data(catsToLoad)
	end = time.time()
	printi(f'Sql Time: {end-start}')
	printv(vid_data)

	start = time.time()
	init_select_vid_random(catsToLoad)
	end = time.time()
	printi(f'Init select vid Time: {end-start}')
	printv(cat_segments)

	start = time.time()
	seq = generate_category_sequenc(desc)
	end = time.time()
	printi(f'Generate category sequenc Time: {end-start}')
	printv(seq)

	start = time.time()
	parts = select_vid_clip_from_seq(10,seq,select_vid_random)
	end = time.time()
	printi(f'Select vid clip from seq Time: {end-start}')
	printv(parts)
	#parts = generate_simple_vide_parts(60) # how much start of the vids to skip  
	#get_vid_parts(parts)
	#get_vid_partsGPU(parts)
	get_vid_partsCPU(parts)
	#get_vid_partsCPUPool(parts)
	join_vids()
	end = time.time()
	print(f'Time: {end-startTotal}')

		
		
		
		
		
	# seq = [
	# 	{'cat': 'other','len':0.3},
	# 	{'cat': 'other','len':0.3},
	# 	{'cat': 'other','len':1.5},
	# 	{'cat': 'other','len':0.3},
	# 	{'cat': 'other','len':0.3},
	# 	{'cat': 'other','len':0.3},
	# 	{'cat': 'cunnilingus','len':5},
	# 	{'cat': 'cunnilingus','len':1},
	# 	{'cat': 'cunnilingus','len':1},
	# ]
		
		
		
		
		