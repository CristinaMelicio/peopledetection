import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image
import h5py as h5


# Read RGB

# Read Depth
depth_file = h5.File('2_cam_1_0.h5', "r")
depth_frames = np.array(depth_file.get(
               "/depth"),
                dtype=float)

# Read CSV
labels = pd.read_csv("2_cam_1_0_labels.csv")

# Create a Rectangle patch
x, y = (labels['x1']+labels['x2'])/2, (labels['y1']+labels['y2'])/2
x, y = labels['x1'], labels['y1']
width, height = (labels['x2'] - labels['x1'], labels['y2']-labels['y1'])

for i in range(len(labels)):
    frame_id = int(labels['frame'][i][-2:])

    if frame_id < 9:
        im = np.array(Image.open('frames/2_cam_1_0_0%d.png' % (frame_id+1)), dtype=np.uint8)
    else:
        im = np.array(Image.open('frames/2_cam_1_0_%d.png' % (frame_id+1)), dtype=np.uint8)

    fig, ax = plt.subplots(2, figsize=(50, 50))

    # Display the image
    ax[0].imshow(im)
    ax[1].imshow(depth_frames[frame_id])

    rect = patches.Rectangle((x[i], y[i]), width[i], height[i], linewidth=1, edgecolor='r',
                             facecolor='none')
    rect2 = patches.Rectangle((x[i] + 50*(np.abs(x[i]-424)/424), y[i] + 75*(np.abs(y[i]-240)/240)), width[i] / 1.25, height[i] / 1.36, linewidth=1, edgecolor='r',
                             facecolor='none')

    # Add the patch to the Axes
    ax[0].add_patch(rect)
    ax[1].add_patch(rect2)
    plt.title("Frame %i" %frame_id)
    plt.show()
