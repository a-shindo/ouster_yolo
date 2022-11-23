from glob import glob
import subprocess as sp

videos=sorted(glob("/home/hayashide/catkin_ws/src/object_detector/scripts/temp/sources/*"))

for videoPath in videos:
    if videoPath[-4:]==".MOV" or videoPath[-4:]==".mov":
        cmd_list=['ffmpeg','-i',videoPath,videoPath[:-4]+".mp4"]
        cmd=' '.join(cmd_list)
        sp.call(cmd,shell=True)

