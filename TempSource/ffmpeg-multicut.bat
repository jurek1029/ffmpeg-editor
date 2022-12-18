@REM ffmpeg -i "E:\Filmy\InTheCrack\InTheCrack 1674 Demi Sutra-62912.mp4" -vf "select='between(t,4,6.5)+between(t,17,26)+between(t,74,91)',setpts=N/FRAME_RATE/TB" tmp2/out.mp4

@REM ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -ss 376.6 -i "E:\Filmy\InTheCrack\InTheCrack 1674 Demi Sutra-62912.mp4" -threads 1 -crf 20 -t 0.34^
@REM     -ss 366.6080000000004 -i "E:\Filmy\InTheCrack\InTheCrack 1674 Demi Sutra-62912.mp4" -threads 1 -crf 20 -t 1.341^
@REM     -filter_complex ^"[0][1]concat=2:v=1[vout]^"^
@REM     -map [vout] tmp2/output.mp4

@REM ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -ss 376.6080000000004 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -threads 1 -crf 20 -t 0.4693333333333385 -c:v h264_nvenc -preset fast tmp2/v-0.mp4
@REM ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -ss 386.6080000000004 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -threads 1 -crf 20 -t 2.4693333333333385 -c:v h264_nvenc -preset fast tmp2/v-1.mp4
@REM ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -ss 396.6080000000004 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -threads 1 -crf 20 -t 2.4693333333333385 -c:v h264_nvenc -preset fast tmp2/v-2.mp4
@REM ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -ss 406.6080000000004 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -threads 1 -crf 20 -t 2.4693333333333385 -c:v h264_nvenc -preset fast tmp2/v-3.mp4
@REM ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -ss 416.6080000000004 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -threads 1 -crf 20 -t 2.4693333333333385 -c:v h264_nvenc -preset fast tmp2/v-4.mp4

@REM ffmpeg -y  -ss 376.6080000000004 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -t 0.4693333333333385 tmp2/v-0.mp4
@REM ffmpeg -y  -ss 386.6080000000004 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -t 2.4693333333333385 tmp2/v-1.mp4
@REM ffmpeg -y  -ss 396.6080000000004 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -t 2.4693333333333385 tmp2/v-2.mp4
@REM ffmpeg -y  -ss 406.6080000000004 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -t 2.4693333333333385 tmp2/v-3.mp4
@REM ffmpeg -y  -ss 416.6080000000004 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -t 2.4693333333333385 tmp2/v-4.mp4



@REM ffmpeg -y -ss 386.95466666666704 -i "E:\Filmy\InTheCrack\InTheCrack 1674 Demi Sutra-62912.mp4" -threads 1 -crf 20 -t 0.4693333333333385 tmp2/v-0.mp4^
@REM  -ss 10 -t 2.4693333333333385  tmp2/v-1.mp4^
@REM  -ss 20 -t 2.4693333333333385  tmp2/v-2.mp4^
@REM  -ss 30 -t 2.4693333333333385  tmp2/v-3.mp4^
@REM  -ss 40 -t 2.4693333333333385  tmp2/v-4.mp4^


@REM ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -ss 300.6080000000004 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -threads 1 -crf 20 -t 10 -c:v h264_nvenc -preset fast tmp2/v-0.mp4^
@REM  -ss 10 -t 10 -c:v h264_nvenc -preset fast tmp2/v-1.mp4^
@REM  -ss 20 -t 10 -c:v h264_nvenc -preset fast tmp2/v-2.mp4

@REM ffmpeg -y -ss 300.6080000000004 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -crf 20 -t 10 tmp2/v-0.mp4^
@REM  -ss 10 -t 10  tmp2/v-1.mp4^
@REM  -ss 20 -t 10  tmp2/v-2.mp4

@REM ffmpeg -y -ss 300 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -crf 20  ^
@REM  -filter_complex ^"[0]trim=0:10[v1];[0]trim=10:20[v2];[0]trim=20:30[v3]^"^
@REM  -map [v1] tmp2/v-0.mp4 -map [v2] tmp2/v-1.mp4 -map [v3] tmp2/v-2.mp4

 ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -ss 300 -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -crf 20  ^
 -filter_complex ^"[0]trim=0:10,setpts=PTS-STARTPTS[v0];^
 [0]trim=10:20,setpts=PTS-STARTPTS[v1];^
 [0]trim=20:30,setpts=PTS-STARTPTS[v2];^
 [0]trim=30:40,setpts=PTS-STARTPTS[v3]^"^
 -map [v0] -c:v h264_nvenc -preset fast tmp2/gpuv-0.mp4^
 -map [v1] -c:v h264_nvenc -preset fast tmp2/gpuv-1.mp4^
 -map [v2] -c:v h264_nvenc -preset fast tmp2/gpuv-2.mp4

@REM ffmpeg -i "E:\Filmy\InTheCrack\InTheCrack 1541 Sarah Banks-45823.mp4" -vf select='between(t,1,5)+between(t,11,15)' -vsync 0 tmp2/out%d.mp4