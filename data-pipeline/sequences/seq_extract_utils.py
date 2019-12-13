"""

The purpose of this script is to extract subclips only with people
from avi videos and depth HDF5

"""


import numpy as np
from scipy.interpolate import griddata
from sklearn.impute import SimpleImputer
from skimage.restoration.inpaint import inpaint_biharmonic
from skimage.morphology import remove_small_holes
import cv2
import more_itertools as mit
from moviepy.video.io.VideoFileClip import VideoFileClip
import h5py_cache as h5c
import matplotlib.pyplot as plt
import pandas as pd
import os

from moviepy.tools import subprocess_call
from moviepy.config import get_setting

camera_height = 4500
mask_low_pass_filter = 700
min_object_area = 4500


def ffmpeg_extract_subclip(filename, t1, t2, targetname=None):
    """ Makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
    name, ext = os.path.splitext(filename)
    if not targetname:
        T1, T2 = [int(1000*t) for t in [t1, t2]]
        targetname = "%sSUB%d_%d.%s" % (name, T1, T2, ext)
    
    cmd = [get_setting("FFMPEG_BINARY"), "-y",
           "-ss", "%0.8f" % t1,
           "-i", filename,
           "-t", "%0.8f" % (t2-t1),
           "-vcodec", "copy", "-acodec", "copy", targetname]
    
    subprocess_call(cmd)


def interpolate_frame(frame, max_height=4500, method="constant"):
    '''Interpolates a frame to remove bad points.
    Three methods available: imputer, inpaint and grid
    '''

    if method == "imput":
        shape = frame.shape
        frame_flat = np.reshape(frame, (-1, 1))
        invalid_mins = np.argwhere(frame_flat <= 0)
        invalid_maxs = np.argwhere(frame_flat >= max_height)
        frame_flat[invalid_mins] = np.nan
        frame_flat[invalid_maxs] = np.nan
        reshape_frame = np.reshape(frame_flat, shape)
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        result = imputer.fit_transform(reshape_frame)

        return result

    elif method == "constant":
        shape = frame.shape
        frame_flat = np.reshape(frame, (-1, 1))
        invalid_mins = np.argwhere(frame_flat <= 0)
        invalid_maxs = np.argwhere(frame_flat >= max_height)
        frame_flat[invalid_mins] = np.nan
        frame_flat[invalid_maxs] = np.nan
        reshape_frame = np.reshape(frame_flat, shape)
        imputer = SimpleImputer(missing_values=np.nan, strategy="constant",
                                fill_value=0.)
        result = imputer.fit_transform(reshape_frame)

        return result

    elif method == "inpaint":
        mask = np.zeros(frame.shape)
        mask[frame <= 0] = 1
        mask[frame >= max_height] = 1
        result = inpaint_biharmonic(frame, mask)

        return result

    elif method == "grid":
        points = np.nonzero(frame)
        values = frame[points]
        shape = frame.shape

        grid_x, grid_y = np.mgrid[0:shape[0]:complex(0, shape[0]),
                                  0:shape[1]:complex(0, shape[1])]

        result = griddata(points, values, (grid_x, grid_y), method="nearest")

        return result

    else:
        print("Not a method")

    return frame


def remove_artifacts(frame,
                     min_area=4500,
                     low_pass=700):
    '''Removes artifacts from a previously subtracted image.
    Only allows "objects" bigger than a threshold value
    '''

    markers = np.zeros_like(frame)
    markers[frame < low_pass] = 1
    bools = markers.astype(np.bool)
    mask_without_holes = remove_small_holes(bools,
                                            area_threshold=min_area,
                                            connectivity=2)
    subtracted = frame.copy()
    subtracted[mask_without_holes] = 0
    return subtracted


def _subtract(frame, background, method="constant"):
    interpolated = interpolate_frame(frame, method=method)
    dif = cv2.absdiff(background, interpolated)
    subtracted = remove_artifacts(dif)
    return subtracted


def _frame_has_object(frame, background, method="constant"):
    subtracted = _subtract(frame, background)
    has_obj = subtracted.max() > 0
    return has_obj


def frame_has_object(background, frame, method="constant"):

    interpolated = interpolate_frame(frame, method=method)

    if not interpolated.shape == frame.shape:
        print(interpolated.shape, frame.shape)
        print("Error")
        return False
    else:
        dif = cv2.absdiff(background, interpolated)
        subtracted = remove_artifacts(dif)
        has_obj = subtracted.max() > 0
        return has_obj
        # return np.array([has_obj, subtracted])


def sequence_extraction(f_id,
                        cam_id,
                        background,
                        video_file,
                        depth_frames,
                        seq_record_file,
                        video_seq_path,
                        depht_seq_path,
                        method="constant",
                        fps=15):

    idx = []
    res_h = depth_frames[0].shape[0]
    res_w = depth_frames[0].shape[1]

    for i in range(2*fps, len(depth_frames)-2*fps):
        if frame_has_object(background, depth_frames[i], method):
            idx.append(i)
            print(i)

    list_seq_idx = []
    for group in mit.consecutive_groups(idx):
        list_seq_idx.append(list(group))

    if len(list_seq_idx) == 0:
        print("Complete video without people")
        return -1
    else:

        # Migth have incorence in deth videos and rgb videos
        # Splitting in frames and times
        i = 0
        for seq_idx in list_seq_idx:
            seq_name = str(f_id)+"_cam_"+str(cam_id)+"_"+str(i)
            video_seq = video_seq_path+seq_name+".mp4"
            depht_seq = depht_seq_path+seq_name+".h5"

            if (seq_idx[-1]-seq_idx[0]) > 1:
                i = i+1
                t1 = np.maximum(seq_idx[0]/fps, 0.0)
                t2 = np.minimum(seq_idx[-1]/fps, len(depth_frames)/fps)
                print("Sequence", i)
                print("t1:", t1, "seq_i:", seq_idx[0])
                print("t2:", t2, "seq_f:", seq_idx[-1])

                df = pd.DataFrame({'name': [seq_name],
                                   'sent': [False],
                                   'labeled': [False]})

                # if file does not exist write header
                if not os.path.isfile(seq_record_file):
                    df.to_csv(seq_record_file,
                              header=['name', 'sent', 'labeled'],
                              index=False)
                # else it exists so append without writing the header
                else:
                    df.to_csv(seq_record_file, mode='a',
                              header=False,
                              index=False)

                # Save HDF5
                hf = h5c.File(depht_seq,
                              'w',
                              libver='latest',
                              chunk_cache_mem_size=1024**3)

                depth_shape = (len(seq_idx), res_h, res_w)

                c1_depth = hf.create_dataset(
                           'depth',
                           data=np.asanyarray(depth_frames[seq_idx]),
                           shape=depth_shape,
                           chunks=True,
                           dtype=np.uint16,
                           compression="gzip",
                           compression_opts=9)
                hf.close()

                ffmpeg_extract_subclip(video_file,
                                       t1,
                                       t2,
                                       targetname=video_seq)


