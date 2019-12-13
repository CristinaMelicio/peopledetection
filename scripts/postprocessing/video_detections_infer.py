import cv2
import numpy as np
import sys
import tensorflow as tf
import getopt
import time
import datetime
import math
import pandas as pd

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("../../models/research/")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
sys.path.remove("../../models/research/")
import tensorflow.contrib.tensorrt as trt
from sort import *
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]


HELP_STR = '''video_detections_infer.py -i <path> -g <path> -l <path> 
      -i: path to input video
      -g: path to the frozen graph (optimized or not .pb)
      -l: path to label map of graph
      -o: path to the output video. Default None
      -h: Display this messag'''


def run_inference_for_single_image(image, tf_graph, tensor_dict, sess):     
    """
    Computes inference for a single image
    """
    
    image_tensor = tf_graph.get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
   
    return output_dict

def main(argv):
    input_video_path="/media/NAS/PeopleDetection/up_realsense_frames/Data_Pipe/AVI/385_cam_2.avi"
    #input_video_path="/media/NAS/PeopleDetection/up_realsense_frames/Data_Pipe/AVI/376_cam_1.avi"

    frozengraph_path="/media/NAS/PeopleDetection/tensorflow/datasets/sharp_blurry_dataset/trainings/faster_rcnn_inception_v2_train/export_trained/frozen_inference_graph.pb"
    #frozengraph_path="/media/NAS/PeopleDetection/tensorflow/datasets/blurry_dataset/trainings/faster_rcnn_inception_v2_train/export_trained/frozen_inference_graph.pb"
    output_video_dir="/media/NAS/PeopleDetection/tensorflow/datasets/sharp_blurry_dataset/trainings/faster_rcnn_inception_v2_train/inferences/"
    label_map_path="/media/NAS/PeopleDetection/tensorflow/datasets/sharp_blurry_dataset/trainings/label_map_v0.pbtxt"
    csv_detections = "./detections.csv"
    try:
        opts, args = getopt.getopt(argv, "hi:g:l:o:")
    except getopt.GetoptError:
        print(HELP_STR)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(HELP_STR)
            sys.exit()
        elif opt == "-o":
            output_video_dir = arg
        elif opt == "-i":
            input_video_path = arg
        elif opt == "-g":
            frozengraph_path = arg
        elif opt == "-l":
            label_map_path = arg
    if input_video_path == None or frozengraph_path == None or label_map_path==None:
        print(HELP_STR)
        sys.exit(2)

    frozen_graph = tf.GraphDef()
    with open(frozengraph_path, 'rb') as f:
        frozen_graph.ParseFromString(f.read())

    category_index = label_map_util.create_category_index_from_labelmap(label_map_path, 
                                                                        use_display_name=True)

    cap = cv2.VideoCapture(input_video_path)
    ret, frame = cap.read()
    im_width = frame.shape[1]
    im_height = frame.shape[0]

    # Sort tracking
    mot_tracker = Sort() 
   
    if output_video_dir != None:
        video_name =input_video_path.split('/')[-1]
        video_name = video_name[:-4]
        vid_path=output_video_dir+video_name+"_det.mp4"
        print("Video Size", im_width, im_height)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        outvideo = cv2.VideoWriter(vid_path, fourcc, 15.0,(im_width, im_height))

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.Session(config=tf_config)
    
    df = pd.DataFrame(columns = ['frame','x1', 'y1','box_h','box_w'])
    with tf.Graph().as_default()  as tf_graph:
        with tf.Session() as tf_sess:
            tf.import_graph_def(frozen_graph, name='')
            ops = tf_graph.get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
              ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf_graph.get_tensor_by_name(tensor_name)
            
            cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Stream', (im_width, im_height))
            frames = 0
            startime=time.time()
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break
                frames += 1
 
                # Actual detection.
                outdet_dict = run_inference_for_single_image(frame, tf_graph, tensor_dict, tf_sess)

                scores = outdet_dict['detection_scores']
                classes = outdet_dict['detection_classes']
                dets = outdet_dict['detection_boxes']
                N = scores.shape[0]

                # Score threshold
                idx = np.argwhere(scores > 0.8)

                dets = dets[idx]
                scores = scores[idx]
                classes = classes[idx]

                dets = np.reshape(dets, (-1, 4))
                scores = np.reshape(scores, (-1, 1))
                classes = np.reshape(classes, (-1, 1))

                # convert to detection notation to [x1,y1,x2,y2]
                ids = np.zeros((dets.shape[0], 1))
                i = np.array([1,0,3,2])
                dets = dets[:, i]

                detections = np.concatenate((dets, scores, ids, classes), axis=1)

                # change detection bboox to image scale
                detections[:,0] = detections[:,0]*im_width
                detections[:,2] = detections[:,2]*im_width
                detections[:,1] = detections[:,1]*im_height
                detections[:,3] = detections[:,3]*im_height        
                

                # update tracking
                tracked_objects = mot_tracker.update(detections)

                print('Frame {}'.format(frames))
                #print(detections)
                
                # Draw tracking retangles
                for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                    box_h = math.floor(y2 - y1)
                    box_w = math.floor(x2 - x1)
                    y1 = math.floor(y1)
                    x1 = math.floor(x1) 
                    color = colors[int(obj_id) % len(colors)]
                    cls = category_index[int(cls_pred)]['name']
                    cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 3)
                    cv2.rectangle(frame, (x1, y1+20), (x1+len(cls)*19+30, y1), color, -1)
                    cv2.putText(frame, cls + ":" + str(int(obj_id)), (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    
                    print("Contagem", obj_id)

                # Draw detection retangles                
                for i in range(0, detections.shape[0]):
                    x1 = math.floor(detections[i,0])
                    y1 = math.floor(detections[i,1])
                    box_h = math.floor(detections[i,3] - y1)
                    box_w = math.floor(detections[i,2] - x1)
                    df = df.append({'frame': frames, 'x1': x1, 'y1': y1, 'box_h':box_h, 'box_w':box_w}, ignore_index=True)
                    cv2.rectangle(frame, (x1, y1),(x1+box_w, y1+box_h), (255,255,255), 4)
                    #print(df)
                # # Visualization of the results of a detection.
                # vis_util.visualize_boxes_and_labels_on_image_array(
                #    frame,
                #    outdet_dict['detection_boxes'],
                #    outdet_dict['detection_classes'],
                #    outdet_dict['detection_scores'],
                #    category_index,
                #    instance_masks=outdet_dict.get('detection_masks'),
                #    use_normalized_coordinates=True,
                #    line_thickness=4,
                #    skip_scores=True,
                #    skip_labels=True)
                  
                cv2.imshow('Stream', frame)
                #cv2.waitKey(0)
                if output_video_dir != None: 
                    outvideo.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            totaltime = time.time()-startime
            print(frames, "frames", frames/totaltime, "fps")
            df.to_csv(r'385_cam_2_detection.csv')
            if output_video_dir != None:
                outvideo.release()
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
   main(sys.argv[1:])

