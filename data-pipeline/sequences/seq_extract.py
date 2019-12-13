import sys
import h5py as h5
import numpy as np
import cv2
sys.path.append('../')

import seq_extract_utils as sq


camera_height = 4500
mask_low_pass_filter = 700
min_object_area = 3500

NAS_Path = "/media/nas/PeopleDetection/up_realsense_frames/"
FPS = 15

if len(sys.argv) > 1:
    f_id = int(sys.argv[1])

if len(sys.argv) > 2:
    cam_id = int(sys.argv[2])
else:
    print("Arguments not provided. Please provide the id from the bag file \
          you want to convert")
    exit()

print("Processing video", f_id)
print("Processing camera", cam_id)
video_path = NAS_Path+"Data_Pipe/AVI/"
video_file = video_path+str(f_id)+"_cam_"+str(cam_id)+".avi"

print("HDF5 file opening")
depth_path = NAS_Path+"Data_Pipe/hdf5/"
depth_path = depth_path+str(f_id)+"_cam_"+str(cam_id)+".h5"
depth_file = h5.File(depth_path, "r")
depth_frames = np.array(depth_file.get(
               "camera"+str(cam_id)+"/depth"),
                dtype=float)

print("Background generation")
background = sq.interpolate_frame(np.median(depth_frames[2*FPS:-2*FPS],
                                  axis=0),
                                  method='constant')

# plt.figure()
# plt.imshow(background, cmap="bone")
# plt.show()

# sub = _subtract(depth_frames[310], background, "constant")
# plt.figure()
# plt.imshow(sub, cmap="bone")
# plt.show()

print("START sequence extraction")
sq.sequence_extraction(f_id,
                       cam_id,
                       background,
                       video_file,
                       depth_frames,
                       seq_record_file=NAS_Path+"Data_Seq/sequence_labeling_record.csv",
                       video_seq_path=NAS_Path+"Data_Seq/AVI/",
                       depht_seq_path=NAS_Path+"Data_Seq/hdf5/")

print("FINISHED sequence extraction")
