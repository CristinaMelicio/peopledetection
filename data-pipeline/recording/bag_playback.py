"""

The purpose of this script is to extract depth and color frames from the
bag file and process them (saving as HDF5 and video AVI as well as
alligning the frames)

"""

import pyrealsense2 as rs
import numpy as np
import sys
import os
import h5py_cache as h5c
import cv2
import pandas as pd

#__import__('tables')

NAS_PATH = "/media/NAS/PeopleDetection/up_realsense_frames/Data_Pipe/"
READ_PATH = "Data/bag/"
WRITE_PATH_DEPTH = "hdf5/"
WRITE_PATH_VIDEO = "AVI/"

def setup_camera(f_id,
                 read_path,
                 cam_id=1,
                 res_h=480,
                 res_w=848,
                 fps=15,
                 align_flag=False):
    filename = ("%i_cam_%i.bag" % (f_id, cam_id))

    pip = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config,
                                      read_path + filename,
                                      repeat_playback=False)

    config.enable_stream(rs.stream.depth, res_w, res_h, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, res_w, res_h, rs.format.rgb8, fps)

    if align_flag:
        align_to = rs.stream.color
        align = rs.align(align_to)
    else:
        align = None

    print("Setting up camera %i " % cam_id)

    return pip, config, align


def retrive_frames(pip,
                   align,
                   config,
                   record_time,
                   f_id,
                   write_path_depth,
                   write_path_video,
                   cam_id=1,
                   fps=15,
                   res_h=480,
                   res_w=848):
    """
    Extract frames from the bag file to a numpy array

    """
    depth_shape = (record_time * fps, res_h, res_w)
    # color_shape = (record_time*fps,res_h,res_w,3)

    depth_frames = np.zeros((fps, res_h, res_w), dtype=np.uint16)
    # color_frames = np.zeros(color_shape,dtype = np.uint8)

    # Save in AVI
    out = cv2.VideoWriter(
        write_path_video + str(f_id) + '_cam_' + str(cam_id) + '.avi',
        cv2.VideoWriter_fourcc(*'MJPG'), fps, (res_w, res_h))

    hf = h5c.File(write_path_depth + str(f_id) + '_cam_' + str(cam_id) + '.h5',
                  'w',
                  libver='latest',
                  chunk_cache_mem_size=1024**3)
    c1 = hf.create_group('camera' + str(cam_id))
    c1_depth = c1.create_dataset('depth',
                                 shape=depth_shape,
                                 chunks=True,
                                 dtype=np.uint16,
                                 compression="gzip",
                                 compression_opts=4)

    profile = pip.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)
    
    playback.resume()
    for i in range(fps):
        # Dropping the first second of recording as the camera is
        # warming up
        frames = pip.wait_for_frames()

    for j in range(record_time):
        print("Progress: %i" % int(j * 100 / record_time))
        for i in range(fps):
            try:
                frames = pip.wait_for_frames()
            except:
                continue
            if frames is not None:
                if align is not None:
                    aligned_frames = align.process(frames)
                else:
                    aligned_frames = frames  # Frames are not aligned here

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
            else:
                print("None")
                continue

            if not depth_frame or not color_frame:
                print("No Frame")
                continue
            # Testing
            # print(np.mean(np.asanyarray(depth_frame.get_data())))
            depth_frames[i] = np.asanyarray(depth_frame.get_data())

            color_frame = np.asanyarray(color_frame.get_data())
            out.write(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))

        c1_depth[j * fps:(j + 1) * fps] = depth_frames

    out.release()
    hf.close()
    # pip.stop() #Uncomenting this may cause trouble
    print("Camera %i is processed" % cam_id)
    return 1

def bag_playback(rec_id, n_cameras):

    # Read metadata
    metadata_path = READ_PATH + str(rec_id) + ".csv"
    metadata = pd.read_csv(metadata_path)
    record_time = metadata["Record time"][0]
    fps = metadata["FPS"][0]
    res = metadata["RES"]

    # Write path to NAS
    write_path_depth = NAS_PATH + WRITE_PATH_DEPTH
    write_path_video = NAS_PATH + WRITE_PATH_VIDEO
    pip, config, align = setup_camera(rec_id, READ_PATH, cam_id=1)
    succ = retrive_frames(pip,
                              align,
                              config,
                              record_time,
                              rec_id,
                              write_path_depth,
                              write_path_video,
                              cam_id=1,
                              fps=fps,
                              res_h=res[1],
                              res_w=res[0])


    if n_cameras == 2:
        pip2, config2, align2 = setup_camera(rec_id, READ_PATH, cam_id=2)
        succ2 = retrive_frames(pip2,
                                   align2,
                                   config2,
                                   record_time,
                                   rec_id,
                                   write_path_depth,
                                   write_path_video,
                                   cam_id=2,
                                   fps=fps,
                                   res_h=res[1],
                                   res_w=res[0])

    if succ == 1:
        bag_file_cam1 = READ_PATH + ("%i_cam_1.bag" % rec_id)
        # del pip
        # gc.collect
        os.remove(bag_file_cam1)
        os.remove(metadata_path)
        print("Removing locally", bag_file_cam1)

    
    here = False
    if n_cameras == 2 and succ2 == 1:
        bag_file_cam2 = READ_PATH + ("%i_cam_2.bag" % rec_id)
        os.remove(bag_file_cam2)
        # del pip2
        # gc.collect
        here = True
        print("Removing locally", bag_file_cam2)
        print("Here", here)
        os._exit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        rec_id = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_cameras = int(sys.argv[2])
    
    bag_playback(rec_id, n_cameras)
