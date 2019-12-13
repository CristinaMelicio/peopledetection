import time
import bag_playback as bag
import os
import pandas as pd
import sys
import numpy as np
import pyrealsense2 as rs
import gc
import getopt



NAS_PATH = "/media/NAS/PeopleDetection/up_realsense_frames/Data_Pipe/"
READ_PATH = "Data/bag/"
WRITE_PATH_DEPTH = "hdf5/"
WRITE_PATH_VIDEO = "AVI/"
ALIGN_bool = False
FPS = 15
RES = (848, 480)
RECORD_TIME = 10
N_CAMERAS = 2


HELP_STR = '''record.py -b <bool> -t <int> -n <int>
      -b: flag to record bag and save locally (False) OR to record bag followed by bag playback saved in NAS. Default False
      -t: record time in seconds. Default 30s 
      -n: number of cameras to record. Default 2
      -h: Display this messag'''


def start_record(rec_id, path, cam_id=1, res_w=848, res_h=480, fps=15):

    pipeline = rs.pipeline()
    config = rs.config()

    path_name = path + str(rec_id) + "_cam_" + str(cam_id) + ".bag"
    config.enable_stream(rs.stream.depth, res_w, res_h, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, res_w, res_h, rs.format.rgb8, fps)
    config.enable_record_to_file(path_name)

    pipeline.start(config)

    ## EXPOSURE STUFF

    # dev = profile.get_device()

    # for i in range(len(dev.sensors)):
    #     if dev.sensors[i].is_depth_sensor() is False:
    #         sensor_color = dev.sensors[i]
    #         break

    # print("Trying to set Exposure")
    # exp = sensor_color.get_option(rs.option.exposure)
    # print("exposure = %d" % exp)
    # print("Setting exposure to new value")
    # exp = sensor_color.set_option(rs.option.enable_auto_exposure, 10000)
    # exp = sensor_color.get_option(rs.option.exposure)
    # print("New exposure = %d" % exp)

    return pipeline, path_name

def record_bag(n_cameras, record_time):

    print("Record Time:", record_time)
    if os.path.isfile('rec_id.npy'):
        rec_id = np.load('rec_id.npy', allow_pickle=True) + 1
    else:
        rec_id = 0

    flag = True
    while(flag):
        try:
            pip, _ = start_record(rec_id, READ_PATH)
            if n_cameras == 2:
                pip2, _ = start_record(rec_id, READ_PATH, 2)
            flag = False

        except RuntimeError:
            print("RuntimeError")
            try:
                pip.stop()
                pip2.stop()
            except:
                print("StopError in pip")

    print("Start recording")
    time0 = time.time()

    while (time.time() - time0) < record_time:
        continue

    if n_cameras == 2:
        pip.stop()
        pip2.stop()
    else:
        pip.stop()

    np.save('rec_id.npy', rec_id)

    # Write Metadata
    print("Saving metadata")
    metadata_path = READ_PATH + str(rec_id) + ".csv"
    metadata = pd.DataFrame({'Record time': record_time,
                             'FPS': FPS,
                             'RES': RES})
    metadata.to_csv(metadata_path)

    print("Saved sucesfully with rec_id: %i" % rec_id)

    return rec_id

def rec_bag_playback(n_cameras, record_time):

    rec_id = record_bag(n_cameras, record_time)
    bag.bag_playback(rec_id, n_cameras)

def main(argv):
    flag = False
    record_time = RECORD_TIME
    n_cameras = N_CAMERAS

    try:
        opts, args = getopt.getopt(argv, "hb:t:n:")
    except getopt.GetoptError:
        print(HELP_STR)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
          print(HELP_STR)
          sys.exit()
        elif opt == "-b":
          flag = bool(arg)
        elif opt == "-t":
          record_time = int(arg)
        elif opt == "-n":
          n_cameras = int(arg)

    # record bag and extract AVI+depth(hdf5) files to NAS
    if flag:
      rec_bag_playback(n_cameras, record_time)
    # record bag and save locally
    else:
      record_bag(n_cameras, record_time)

if __name__ == "__main__":
    main(sys.argv[1:])