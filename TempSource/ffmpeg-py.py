import os
import pandas as pd
from sqlite3 import connect
import random

conn = connect("./Classified_Videos.sqlite")

files = [
"E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4",
"E:\Filmy\InTheCrack\InTheCrack 1674 Demi Sutra-62912.mp4",
"E:\Filmy\InTheCrack\InTheCrack 1702 Demi Sutra-66098.mp4"
]

resolution = "1920x1080"
output_file = "out.mp4"

res_comm = resolution.replace('x',':')
vid_parts = []
#{'v':0,'s':60,'t':10},
#{'v':0,'s':20,'t':10},
#{'v':1,'s':20,'t':10},
#{'v':2,'s':20,'t':10}
			
vid_names = []
vid_res = []


def init():
	global vid_names
	global vid_res
	global files
	#for i in range(len(files)):
	#	vid_names[i] = i
	
	get_size_command = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "
	for file in files:
		res = os.popen(f'{get_size_command}"{file}"')
		vid_res.append(res.read())

	
def input_files(command):
	global files
	for f in files:
		command += f' -i "{f}"'
	return command

def get_vid_parts():
	global vid_parts
	global files
	
	parts = []
	
	for i,f in enumerate(files):
		f_name = os.path.splitext(os.path.basename(f))
		df_key = pd.read_sql(f'SELECT classifier_id FROM Keys WHERE key = "{f_name[0]}"', conn)
		if len(df_key.classifier_id) == 0: continue
		
		id = df_key.classifier_id[0]
		df = pd.read_sql(f'SELECT second FROM Data WHERE classifier_id = {id} AND other >= 0.6 ', conn)
		for s in df.second:
			parts.append((i,s))
		# other_parts = []
		# prev = df.second[0]
		# start = df.second[0]
		# for j in range(1,len(df.second)):
			# if df.second[j] - 1 == prev :
				# prev += 1
				# continue
			# else:
				# vid_parts
				# other_parts.append([start,prev])
				# prev = df.second[j]
				# start = prev
		# for op in other_parts:
			# vid_parts.append({'v':i,'s':op[0],'t':op[1]-op[0] + 1})
	
	for i in range(100):
		r = random.randint(0,len(parts))
		vid_parts.append({'v':parts[r][0],'s':parts[r][1],'t':1})

		

def add_scaling_to_res(command, res):
	command += f',scale={res}:force_original_aspect_ratio=decrease'
	command += f',pad={res}:(ow-iw)/2:(oh-ih)/2'
	return command

def edit_vid(command):
	global vid_names
	global vid_parts
	
	command += ' -filter_complex "'

	for i,part in enumerate(vid_parts):
		command += f'[{part["v"]}]trim=start={part["s"]}:duration={part["t"]},setpts=PTS-STARTPTS'
		if resolution not in vid_res[part["v"]]: command = add_scaling_to_res(command,res_comm)
		command += ",setsar=1"
		command += f'[v{i}],'
		vid_names.append(f'v{i}')
		
	for vid in vid_names:
		command += f'[{vid}]'
	command += f"concat=n={len(vid_names)}[outv]"
	command += '" -map [outv]'
	
	# for (i,file) in enumerate(files):
		
		# command += f'[{vid_names[i]}]'
		# if resolution not in res_v : command = add_scaling_to_res(command, res)
		# command += "setsar=1"	
		# command += f'[v{vid_names[i]}],'
		
		# vid_names[i] = f'v{vid_names[i]}'

	# for (i,vid) in enumerate(vid_names):
		# command += f'[{vid_names[i]}][{i}:a]'
	# command += f"concat=n={len(files)}:v=1:a=1[outv][outa]"
	# command += '"'
	
	return command


def outparams(command):
	#command += ' -vcodec libx264 -crf 27'
	command += ' -crf 20'
	command += " -preset ultrafast -threads 6 "
	command += f' {output_file}'
	return command

def create_command():
	command = "ffmpeg -hwaccel cuda"
	command = input_files(command)
	command = edit_vid(command)
	command = outparams(command)
	return command


if __name__ == "__main__":
	init()
	get_vid_parts()
	command = create_command()
	print(command)
	res = os.system(command)
		
		
		
		
		
		
		
		
		
		