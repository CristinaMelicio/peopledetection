#https://www.kaggle.com/lehomme/overhead-depth-images-people-detection-gotpd1


import sys
import numpy as np
# import struct
# import os
# import array
# import cv2
import matplotlib.pylab as plt
from scipy.ndimage.filters import gaussian_filter
# from sklearn.preprocessing import Imputer
# from scipy.ndimage import gaussian_filter
# from functools import reduce
import h5py as h5
import gc
np.set_printoptions(threshold=sys.maxsize)
sys.path.append("../")
import seq_extract_utils as sq


def localNeigborhood(di, v, r, c):

    L = []
    delta_r = {1: -1, 2: 0, 3: 1, 4: 0, 5: -1, 6: 1, 7: 1, 8: -1}
    delta_c = {1:  0, 2: 1, 3: 0, 4: -1, 5: 1, 6: 1, 7: -1, 8: -1}

    # vertical and horizontal regions
    if di == 1 or di == 2 or di == 3 or di == 4:
        for i in range(-(v-1), v):
            if delta_c[di] == 0:
                c_ = c + i
                r_ = r + delta_r[di]*v
                if c_ < 0 or c_ >= Nc or r_ < 0 or r_ >= Nr:
                    continue
            elif delta_r[di] == 0:
                r_ = r + i
                c_ = c + delta_c[di]*v
                if c_ < 0 or c_ >= Nc or r_ < 0 or r_ >= Nr:
                    continue
            L.append((r_, c_))

    # diagonal regions
    else:
        r_ = r + delta_r[di]*v
        c_ = c + delta_c[di]*v
        if not (c_ < 0 or c_ >= Nc or r_ < 0 or r_ >= Nr):
            L.append((r_, c_))

    if L == []:
        return []
    else:
        return L


def v1Neigborhood(r, c):
    delta_r = {1: -1, 2: 0, 3: 1, 4: 0, 5: -1, 6: 1, 7: 1, 8: -1}
    delta_c = {1:  0, 2: 1, 3: 0, 4: -1, 5: 1, 6: 1, 7: -1, 8: -1}

    v1 = []

    for i in range(1, 9, 1):
        r_ = r+delta_r[i]
        c_ = c+delta_c[i]
        if not (c_ < 0 or c_ >= Nc or r_ < 0 or r_ >= Nr):
            v1.append((r_, c_))

    return v1


NAS_Path = "/media/nas/PeopleDetection/up_realsense_frames/"

print("Background generation")

depth_path = NAS_Path+"Data_Pipe/hdf5/13_cam_1.h5"
depth_file = h5.File(depth_path, "r")
depth_frames = np.array(depth_file.get(
               "camera1/depth"),
                dtype=float)
background = sq.interpolate_frame(np.median(depth_frames[60:-60],
                                  axis=0),
                                  method='constant')

del depth_frames
del depth_file
gc.collect()

seq_path = NAS_Path+"Data_Seq/hdf5/"+"13_cam_1_0.h5"
seq_file = h5.File(seq_path, "r")
seq_frames = np.array(seq_file.get("depth"),
                      dtype=float)

print("seq number of frames", seq_frames.shape)

for seq_idx, depthimage in enumerate(seq_frames):

    plt.imshow(depthimage, cmap='bone')
    plt.colorbar()
    plt.show()

    subtracted = sq._subtract(depthimage, background, method='constant')
    h_filtered = gaussian_filter(subtracted, sigma=9)


    plt.imshow(h_filtered, cmap='bone')
    plt.colorbar()
    plt.show()

    plt.imshow(subtracted, cmap='bone')
    plt.colorbar()
    plt.show()

## 3RD ROI'S ESTIMATION
M = seq_frames[0].shape[0]
N = seq_frames[0].shape[1]
hpmax = 2200
hpmin = 1100
hcmax = 4500
hinterest=1400
l = 130 # minimum area of top head l*l
# aux = 1933/1.4
aux = 1200
# aux = 365.77
D = round(aux*l/(hcmax-hpmin))

Nr = int(M / D)
Nc = int(N / D)

h_rc = np.zeros((Nr, Nc))


for i in range(Nr):
    for j in range(Nc):
        h_rc[i][j] = np.amax(h_filtered[np.ix_(range(i*D, i*D+D-1), range(j*D, j*D+D-1))])
h_rc = np.reshape(h_rc, (Nr, Nc))

candidates = []
count = 0
for r in range(Nr):
    for c in range(Nc):
        if (h_rc[r][c] > hpmin):
            for v1 in v1Neigborhood(r, c):
                if h_rc[r][c] > h_rc[v1[0]][v1[1]]:
                    count = count + 1
                else:
                    count = 0
                    break
            if count == len(v1Neigborhood(r, c)):
                candidates.append((r, c))
                count = 0
print("Candidates", candidates)

delta_r = {1: -1, 2: 0, 3: 1, 4: 0, 5: -1, 6: 1, 7: 1, 8: -1}
delta_c = {1:  0, 2: 1, 3: 0, 4: -1, 5: 1, 6: 1, 7: -1, 8: -1}
Nv = 5
ROIs = []

for sr in candidates:
    ROI = [sr]
    r = sr[0]
    c = sr[1]
    for v1 in v1Neigborhood(r, c):
        if h_rc[v1[0]][v1[1]] >= h_rc[r][c] - hinterest:
            ROI.append(v1)

    brk = False
    for i in range(1, 9, 1):
        for v in range(2, Nv, 1):
            for sr_ in localNeigborhood(i, v, r, c):
                cadinality = len(localNeigborhood(i, v, r, c)) + len(ROI)
                r_n = sr_[0] - delta_r[i]
                r_p = sr_[0] + delta_r[i]
                c_n = sr_[1] - delta_c[i]
                c_p = sr_[1] + delta_c[i]

                #print("rp: ", r_p, " rn: ", r_n, " cp: ", c_p, " cn: ", c_n)
                if (r_p < Nr and c_p < Nc and c_n >= 0 and r_n >= 0 and r_p >=0 and c_p >= 0 and c_n < Nc and r_n < Nr):
                    if h_rc[sr_[0]][sr_[1]] >= h_rc[r][c] - hinterest:
                        if (i == 1 or i == 2 or i == 3 or i == 4) and cadinality >= v-1:
                            if h_rc[r_n][c_n] >= h_rc[r][c] - hinterest:
                                if h_rc[r_n][c_n] >= h_rc[sr_[0]][sr_[1]] and h_rc[sr_[0]][sr_[1]] >= h_rc[r_p][c_p]:
                                    if (sr_[0]==14 and sr_[1] ==9):
                                        print("AQUI", i)
                                        print("rp: ", r_p, " rn: ", r_n, " cp: ", c_p, " cn: ", c_n)
                                        print("h_rc[sr_[0]][sr_[1]]: ", h_rc[sr_[0]][sr_[1]], " h_rc[r_n][c_n]: ", h_rc[r_n][c_n], " h_rc[r_p][c_p]: ", h_rc[r_p][c_p])

                                    ROI.append(sr_)
                                else:
                                    print("break")
                                    brk = True
                                    break
                            else:
                                brk = True
                                break

                        if (i == 5 or i == 6 or i == 7 or i == 8) and cadinality == 1:
                            if h_rc[r_n][c_n] >= h_rc[r][c] - hinterest:
                                if h_rc[r_n][c_n] >= h_rc[sr_[0]][sr_[1]] and h_rc[sr_[0]][sr_[1]] >= h_rc[r_p][c_p]:
                                    if (sr_[0] == 14 and sr_[1] == 9):
                                        print("AQUI", i)
                                        print("rp: ", r_p, " rn: ", r_n, " cp: ", c_p, " cn: ", c_n)
                                        print("h_rc[r_n][c_n]: ", h_rc[r_n][c_n], " h_rc[r_p][c_p]: ", h_rc[r_p][c_p])

                                    ROI.append(sr_)
                                else:
                                    print("break")
                                    brk = True
                                    break
                            else:
                                brk = True
                                break
            if brk == True:
                brk = False
                break

    ROI = list(set(ROI))
    ROIs.append(ROI)
    #print(ROI)
print(ROIs)