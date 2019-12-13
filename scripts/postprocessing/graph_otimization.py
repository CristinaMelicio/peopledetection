import tensorflow as tf
from object_detections import optimize_model

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.67)

frozen_graph = tf.GraphDef()
with tf.gfile.GFile("/media/NAS/PeopleDetection/tensorflow/datasets/blurry_dataset/trainings/faster_rcnn_inception_v2_train/export_trained/frozen_inference_graph.pb", 'rb') as f:
    frozen_graph.ParseFromString(f.read())
    frozen_graph_opt = optimize_model(
        frozen_graph,
        use_trt=True,
        precision_mode="FP16",
        calib_images_dir="/media/NAS/PeopleDetection/tensorflow/datasets/sharp_dataset_v0/images/test",
        num_calib_images=8,
        calib_batch_size=1,
        calib_image_shape=[480, 848],
        max_workspace_size_bytes=1320000000,
        output_path="/media/NAS/PeopleDetection/tensorflow/datasets/blurry_dataset/trainings/faster_rcnn_inception_v2_train/export_trained/frozen_graph_opt.pb"
    )



## Load and convert a frozen graph
#graph_def = tf.GraphDef()
#with tf.gfile.GFile("/media/NAS/PeopleDetection/tensorflow/datasets/blurry_dataset/trainings/faster_rcnn_inception_v2_train/export_trained/frozen_inference_graph.pb", 'rb') as f:
#  graph_def.ParseFromString(f.read())
#  print('Check out the input placeholders:')
#  nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
#  for node in nodes:
#    print(node)
#  
#
#converter = trt.TrtGraphConverter(input_graph_def=graph_def, nodes_blacklist=['image_tensor'])
#graph_def = converter.convert()
#converter.save('/media/NAS/PeopleDetection/tensorflow/datasets/blurry_dataset/trainings/faster_rcnn_inception_v2_train/export_trained/')
#
