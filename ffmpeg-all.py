from cgi import test
import os
import random
import shutil
import subprocess
import time
from sqlite3 import connect

import pandas as pd

from AudioBeatDetect import AudioProcessor 

conn = connect("./Classified_Videos.sqlite")

files = [
"E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4",
"E:\Filmy\InTheCrack\InTheCrack 1674 Demi Sutra-62912.mp4",
"E:\Filmy\InTheCrack\InTheCrack 1702 Demi Sutra-66098.mp4"
]

# music = "E:\Filmy\Generator\Auto_PMV_Generator\Resources\TempMusic\Music Compilation Video - Sally.mp4"
music = "E:\Filmy\Generator\Auto_PMV_Generator\Resources\TempMusic\\'Bass Slut' Teen Fucking PMV.mp4"


resolution = "1920x1080"
crf = 20
compresion_preset = "fast"
#fast, medium, slow, FOR GPU
#ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow FOR CPU

output_file = "out-myFFTBass-alt-var-filt.mp4"

res_comm = resolution.replace('x',':')
vid_parts = []
#{'v':0,'s':60,'t':10},

			
vid_names = []
vid_res = []
vid_proces = []
musicData = None

def init():
	global vid_names
	global vid_res
	global files, musicData
	
	get_size_command = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "
	for file in files:
		res = os.popen(f'{get_size_command}"{file}"')
		vid_res.append(res.read())
	
	shutil.rmtree("./tmp")
	os.makedirs("./tmp")
	
	# Ap = AudioProcessor(music, 1.6, useAutoConst=True, useFFT=False, buffSizeInSec=4, sampleSize= 1024, debug=True)
	# Ap = AudioProcessor(music,5,useAutoConst=True, useFFT=True,useLowPassFilter=True, minLengthInSec=0.2 ,buffSizeInSec=0.6, sampleSize=1024,FFTSubbands=32,filterFuncPrams=(2,1), debug=True)
	Ap = AudioProcessor(music,minLengthInSec=0.2, buffSizeInSec=0.8,debug=True)
	musicData = Ap.processAudio()

def get_vid_parts(crf, preset):
	global vid_parts
	global files
	global vid_proces, musicData
	
	parts = []
	
	# for i,f in enumerate(files):
		# f_name = os.path.splitext(os.path.basename(f))
		# df_key = pd.read_sql(f'SELECT classifier_id FROM Keys WHERE key = "{f_name[0]}"', conn)
		# if len(df_key.classifier_id) == 0: continue
		
		# id = df_key.classifier_id[0]
		# df = pd.read_sql(f'SELECT second FROM Data WHERE classifier_id = {id} AND other >= 0.6 ', conn)
		# for s in df.second:
			# parts.append((i,s))
	
	vlist = open("tmp/list.txt","w")
	# # Test 100 1 s long parts
	vid_id = 0
	prevs = 0
	tstart = 60.0
	for i, s in enumerate(musicData):
		
		comm = f'ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -ss {tstart} -i "{files[vid_id]}"'
		if resolution not in vid_res[vid_id]:
			comm += f' -vf scale_npp={res_comm}:force_original_aspect_ratio=decrease,pad={res_comm}:(ow-iw)/2:(oh-ih)/2'
		comm += f' -threads 1 -crf {crf} -t {s - prevs} -c:v h264_nvenc -preset {compresion_preset} tmp/v{i}.mp4'
		tstart += s - prevs
		prevs = s
		print(comm)
		#vid_proces.append(subprocess.Popen(comm,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
		progres_bar(i,len(musicData))
		subprocess.run(comm,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
		vlist.write(f'file v{i}.mp4 \n')
		
		vid_id += 1
		vid_id %= len(files)
	vlist.close()
	
def join_vids():
	#for i,p in enumerate(vid_proces):		
	#	progres_bar(i,len(vid_proces))
	#	p.wait()
	
	if os.path.exists(output_file): os.remove(output_file)
	subprocess.run('ffmpeg -f concat -i ./tmp/list.txt -c copy ./tmp/out.mp4',stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

	subprocess.run(f'ffmpeg -i ./tmp/out.mp4 -i "{music}" -c copy -shortest -map 0:v:0 -map 1:a:0 {output_file}',stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
	# -stream_loop -1
	progres_bar(10,10)
	print('\n')
	


def progres_bar(progress, total):
	precent = 100 * (progress / float(total))
	bar = '█' * int(precent) + '-' * int(100 - precent)
	print(f"\r|{bar}| {precent:.2f}%", end="\r")

if __name__ == "__main__":
	init()
	start = time.time()
	get_vid_parts(crf,compresion_preset)
	join_vids()
	end = time.time()
	print(f'Time: {end-start}')

		
		
		
		
		
		
		
		
		
		