import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import getopt
import os.path
sys.path.append("../../")
from models.research.object_detection.utils import label_map_util
sys.path.remove("../../")

HELP_STR = '''read_tfrecord.py -g <boolean> -p <path> -i <bolean> -l <path>
      -g: show ground truth boxes. Default False
      -p: path to the tfrecord. Default "detections.tfrecord"
      -i: if the tf record has images. If not it will read from a "frames" dir in the same directory as path. Default False
      -l: path to the label map of the tfrecord. Default "label_map.pbtxt"
      -b: max bounding boxes per image
      -v: path to create video
      -h: Display this messag'''

def add_ground_truth(image, example):
    height = int(example['image/height'].numpy())
    width = int(example['image/width'].numpy())

    xmin = example['image/object/bbox/xmin'].values.numpy()
    xmax = example['image/object/bbox/xmax'].values.numpy()
    ymin = example['image/object/bbox/ymin'].values.numpy()
    ymax = example['image/object/bbox/ymax'].values.numpy()

    xmin = xmin * width
    xmax = xmax * width
    ymin = ymin * height
    ymax = ymax * height

    bboxes = np.column_stack((xmin, ymin, xmax, ymax)).astype(dtype=int)
    label_list = example['image/object/class/text'].values.numpy()

    label = np.vectorize(lambda y: y.decode())(label_list)
    print(label_list)

    for idx, bbox in enumerate(bboxes):
        y = bbox[1] - 10 if bbox[1] > 10 else bbox[1] + 10
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), [255, 255, 255], 2)
        cv2.putText(image, label[idx], (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 0], 1)

def add_detection(image, example, label_dict, box_limit):
    height = int(example['image/height'].numpy())
    width = int(example['image/width'].numpy())

    xmin_d = example['image/detection/bbox/xmin'].values.numpy()
    xmax_d = example['image/detection/bbox/xmax'].values.numpy()
    ymin_d = example['image/detection/bbox/ymin'].values.numpy()
    ymax_d = example['image/detection/bbox/ymax'].values.numpy()

    xmin_d = xmin_d * width
    xmax_d = xmax_d * width
    ymin_d = ymin_d * height
    ymax_d = ymax_d * height

    labels_d = example['image/detection/label'].values.numpy()
    scores_d = example['image/detection/score'].values.numpy()

    idx_d = np.argwhere(scores_d>0.98)
    bboxes_d = np.column_stack((xmin_d[idx_d], ymin_d[idx_d], xmax_d[idx_d], ymax_d[idx_d], labels_d[idx_d], scores_d[idx_d]))

    ind = np.argsort(bboxes_d[:,-1])
    bboxes_d = bboxes_d[ind][:box_limit]

    for ix in range(len(bboxes_d)):
        bbox = bboxes_d[ix]
        label_d = bbox[4]
        score_d = bbox[5]
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        y = ymin - 10 if ymin > 10 else ymin + 10
        text = "%s: %.2f"%(label_dict[int(label_d)], score_d)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), [0, 0, 255], 2)
        cv2.putText(image, text, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0, 0, 0], 1)

def main(argv):
    ground_truth = False
    data_path = 'detections.tfrecord'
    has_images = True
    label_map_path = 'label_map.pbtxt'
    box_limit = None
    video_path = None
    try:
        opts, args = getopt.getopt(argv, "hg:p:i:l:b:v:")
    except getopt.GetoptError:
        print(HELP_STR)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(HELP_STR)
            sys.exit()
        elif opt == "-g":
            ground_truth = bool(arg)
        elif opt == "-p":
            data_path = arg
        elif opt == "-i":
            has_images = bool(arg)
        elif opt == "-l":
            label_map_path = arg
        elif opt == "-b":
            box_limit = int(arg)
        elif opt == "-v":
            video_path = arg

    tfrecord_read(data_path, label_map_path, video_path, ground_truth, box_limit, has_images)

def tfrecord_read(data_path, label_map_path, video_path, ground_truth=False, box_limit=None, has_images=True):
    tf.enable_eager_execution()
    raw_image_dataset = tf.data.TFRecordDataset(data_path)
    feature_description = {
        'image/height': tf.FixedLenFeature((), tf.int64),
        'image/width': tf.FixedLenFeature((), tf.int64),
        'image/filename': tf.FixedLenFeature((), tf.string),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/text': tf.VarLenFeature(tf.string),
        'image/detection/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/detection/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/detection/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/detection/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/detection/label': tf.VarLenFeature(tf.int64),
        'image/detection/score': tf.VarLenFeature(tf.float32),
    }
    
    if has_images:
        feature_description['image/encoded'] = tf.FixedLenFeature((), tf.string)
 
    label_map_dict_inv = label_map_util.get_label_map_dict(label_map_path, use_display_name=True)
    label_map_dict = {v: k for k, v in label_map_dict_inv.items()}

    # width = sample['image/width']
    # height = sample['image/height']
    # print(width, height)
    parsed_image_dataset = raw_image_dataset.map(lambda record: tf.parse_single_example(record, feature_description))

    #parsed_image_dataset.shapes['image/width'].bytes_list.value[0]
    #print(parsed_image_dataset)
    if video_path != None:
        width = 848
        height =480
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(video_path,fourcc, 15.0,(width,height))
  
    counter = 0
    for example in parsed_image_dataset: 
        if has_images:
            image_bytes = example['image/encoded']
            image_nparr = np.fromstring(image_bytes.numpy(), np.uint8)
            img = cv2.imdecode(image_nparr, cv2.COLOR_RGB2BGR)
        else:
            data_parent_dir = os.path.dirname(data_path)
            filename = example['image/filename'].numpy()
            filename = np.fromstring(filename, np.uint8)
            img_path = os.path.join(data_parent_dir, 'frames', filename)
            img = cv2.imread(img_path)

        #add_detection(img, example, label_map_dict, box_limit)
        if ground_truth:
            add_ground_truth(img, example)
        
        cv2.imshow("Output", img)
        k = cv2.waitKey()
        if k == 27:
            break

        if video_path != None: 
            video.write(img)

    if video_path != None:
        video.release()

if __name__ == "__main__":
   main(sys.argv[1:])