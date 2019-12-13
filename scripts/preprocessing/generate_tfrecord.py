import tensorflow as tf
import re
import pandas as pd
from os.path import isfile, join
import os
from os import listdir
import getopt
import glob
import numpy as np
import sys
from PIL import Image 

sys.path.append("../../models/research/")
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
sys.path.remove("../../models/research/")

flags = tf.app.flags
flags.DEFINE_string('csv_path_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('img_path', '', 'Path to images')
flags.DEFINE_string('label_map', '', 'Path to the label map of the tfrecord')
FLAGS = flags.FLAGS



HELP_STR = '''generate_tfrecord.py -o <path> -i <path>
      -o: path to output tfrecord file 
      -i: path to images 
      -c: path to the csv input folder
      -l: path to the label map of the tfrecord
      -h: Display this messag'''

def create_tf_example(example, img_path, csv_path, label_map_dict):
    filename = example.encode()  # Filename of the image. Empty if image is not from file
    with tf.gfile.GFile(img_path+example, 'rb') as fid:
        encoded_image_data = fid.read()
    image_format = 'jpeg'.encode()  # b'jpeg' or b'png'

    if (csv_path == None and label_map_dict == None):
        with Image.open(img_path+example) as img: 
            width, height = img.size 

        feature = {
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(image_format),
        }

    else:         
        # TODO(user): Populate the following variables from your example.
        frame_label = read_frame_label(example[:-4], csv_path)
        if frame_label.empty:
            return 0

        height = frame_label['height'].values[0]  # Image height
        width = frame_label['width'].values[0]  # Image width
        xmins = (frame_label['x1']/width).tolist()
        # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs = (frame_label['x2']/width).tolist()  # List of normalized right x coordinates in bounding box
        # (1 per box)
        ymins = (frame_label['y1']/height).tolist()
        # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs = (frame_label['y2']/height).tolist()  # List of normalized bottom y coordinates in bounding box
        # (1 per box)
        #classes_text = frame_label.apply(lambda x: x.encode(), columnns=['class']).tolist()
        classes_text = frame_label['class'].apply(lambda x: x.encode()).tolist()  # List of string class name of bounding box (1 per box)
        classes = [label_map_dict[k.decode("utf-8")] for k in classes_text]
        feature = {
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_example


def read_frame_label(frame_name, csv_path):  # frame_name should be without the jpg extension
    # Extract the sequence name & open the csv

    num = re.findall('\d+', frame_name)
    video_id = num[0]
    cam_id = num[1]
    seq_id = num[2]
    frame_id = int(num[3])
    seq_name = "%s_cam_%s_%s.csv" % (video_id, cam_id, seq_id)
    frame_name = "%s_cam_%s_%s_%d" % (video_id, cam_id, seq_id, frame_id)
    try:
        frame_label = pd.read_csv(csv_path + seq_name)
    except:
        return pd.DataFrame()
    
    frame_label = frame_label.loc[frame_label['frame'] == frame_name]
    df = frame_label.drop(frame_label[frame_label['class'] != 'Person'].index)
    #print(df)
    return df


def main(argv):
    output_path = None
    img_path = None
    csv_path = None
    label_map_path = None

    try:
        opts, args = getopt.getopt(argv, "ho:c:i:l:")
    except getopt.GetoptError:
        print(HELP_STR)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
          print(HELP_STR)
          sys.exit()
        elif opt == "-o":
          output_path = arg
        elif opt == "-c":
          csv_path = arg
        elif opt == "-i":
          img_path = arg
        elif opt == "-l":
          label_map_path = arg

    if img_path == None or output_path == None:
        print(HELP_STR)
        sys.exit(2)

    tfrecord_generator(output_path, img_path, csv_path=csv_path, label_map_path=label_map_path)

def tfrecord_generator(output_path, img_path, csv_path=None, label_map_path=None):

    with tf.python_io.TFRecordWriter(output_path) as writer:
        images = [f for f in sorted(listdir(img_path)) if isfile(join(img_path, f))]
        
        if label_map_path == None:
            label_map_dict = None
        else:
            label_map_dict = label_map_util.get_label_map_dict(label_map_path, use_display_name=False)

        len_examples = len(images)
        counter = 0
        for image in images:
            tf_example = create_tf_example(image, img_path, csv_path, label_map_dict)
            if tf_example == 0:
                print("True")
            if tf_example != 0:
                writer.write(tf_example.SerializeToString())
            if counter % 100 == 0:
                print("Percent done", (counter / len_examples) * 100)
            counter += 1
    #writer.close()

if __name__ == '__main__':
    main(sys.argv[1:])
