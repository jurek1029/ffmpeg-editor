import os
import shutil
import random
import pandas as pd
from sqlite3 import connect
import subprocess
import time 
from AudioBeatSegment import inputMusic

conn = connect("./Classified_Videos.sqlite")

files = [
"E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4",
"E:\Filmy\InTheCrack\InTheCrack 1674 Demi Sutra-62912.mp4",
"E:\Filmy\InTheCrack\InTheCrack 1702 Demi Sutra-66098.mp4"
]

music = "E:\Filmy\Generator\Auto_PMV_Generator\Resources\TempMusic\Music Compilation Video - Sally.mp4"

resolution = "1920x1080"
crf = 20
compresion_preset = "veryfast"
#ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow

output_file = "out.mp4"

res_comm = resolution.replace('x',':')
vid_parts = []
#{'v':0,'s':60,'t':10},

			
vid_names = []
vid_res = []
vid_proces = []


def init():
	global vid_names
	global vid_res
	global files
	
	get_size_command = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "
	for file in files:
		res = os.popen(f'{get_size_command}"{file}"')
		vid_res.append(res.read())
	
	shutil.rmtree("./tmp")
	os.makedirs("./tmp")
	

def get_vid_parts(crf, preset):
	global vid_parts
	global files
	global vid_proces
	
	parts = []
	
	for i,f in enumerate(files):
		f_name = os.path.splitext(os.path.basename(f))
		df_key = pd.read_sql(f'SELECT classifier_id FROM Keys WHERE key = "{f_name[0]}"', conn)
		if len(df_key.classifier_id) == 0: continue
		
		id = df_key.classifier_id[0]
		df = pd.read_sql(f'SELECT second FROM Data WHERE classifier_id = {id} AND other >= 0.6 ', conn)
		for s in df.second:
			parts.append((i,s))
	
	vlist = open("tmp/list.txt","w")
	# Test 100 1 s long parts
	elements = 100
	min_len = 100
	for i in range(elements):
		r = random.randint(0,len(parts))
		rmili = min_len + random.randint(0,1000 - min_len)
		comm = f'ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -ss {parts[r][1]} -i "{files[parts[r][0]]}"'
		if resolution not in vid_res[parts[r][0]]:
			comm += f' -vf scale_npp={res_comm}:force_original_aspect_ratio=decrease,pad={res_comm}:(ow-iw)/2:(oh-ih)/2'
		comm += f' -threads 1 -crf {crf} -t 00:00.{rmili:03} -c:v h264_nvenc -preset fast tmp/v{i}.mp4'
		
		print(comm)
		#vid_proces.append(subprocess.Popen(comm,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
		progres_bar(i,elements)
		subprocess.run(comm,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
		vlist.write(f'file v{i}.mp4 \n')
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
	precent = int(100 * (progress / float(total)))
	bar = '█' * precent + '-' * (100 - precent)
	print(f"\r|{bar}| {precent:.2f}%", end="\r")

if __name__ == "__main__":
	init()
	start = time.time()
	get_vid_parts(crf,compresion_preset)
	join_vids()
	end = time.time()
	print(f'Time: {end-start}')

		
		
		
		
		
		
		
		
		
		