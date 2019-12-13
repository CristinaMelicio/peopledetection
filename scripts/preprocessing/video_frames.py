"""
Create the train, eval and test image set
"""

import os
from os import listdir
from os.path import isfile, join


#folder = "/media/nas/PeopleDetection/tensorflow/datasets/sharp_dataset_v0/images_v0/"
#Get names of the videos
label_path = "/media/nas/PeopleDetection/tensorflow/datasets/sharp_dataset_v0/annotations/test_labels/"

# Read all the videos
path = "/media/NAS/PeopleDetection/up_realsense_frames/Data_Seq/AVI/"


path="/media/NAS/PeopleDetection/up_realsense_frames/Data_Pipe/AVI/385_cam_2.avi"
images="/home/cristina/Documents/peopledetectiondepth/scripts/preprocessing/teste"


def videos2frames(videos_input_path, frames_output_path, labels_path):
	onlyfiles = [f for f in listdir(labels_path) if isfile(join(labels_path, f))]
	for file in onlyfiles:
		out = file[:-4]
		os.system("ffmpeg -i %s.mp4 -start_number 0 -qscale:v 2 %s/%s_%%03d.jpg" % (videos_input_path+out, frames_output_path, out))

def video2frames(video_input_path, frames_output_path):
	video_input_path=video_input_path[:-4]
	out = video_input_path.split('/')[-1]
	os.system("ffmpeg -i %s.avi -start_number 0 -qscale:v 2 %s/%s_%%03d.jpg" % (video_input_path, frames_output_path, out))


if __name__ == "__main__":
	#videos2frames(path, images, label_path)
	video2frames(path, images)

