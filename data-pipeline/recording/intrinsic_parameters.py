import pyrealsense2 as rs
import numpy as np
import cv2

FPS = 15
RES = (848, 480)


ctx = rs.context()
pipelines = []

for idx,d in enumerate(ctx.query_devices()):
	print(d)
	pipe = rs.pipeline(ctx)
	cfg = rs.config()
	cfg.enable_device(d.get_info(rs.camera_info.serial_number))
	cfg.enable_stream(rs.stream.color, RES[0], RES[1], rs.format.rgb8, FPS)
	prof = pipe.start(cfg)
	for i in range(FPS):
		# Dropping the first second of recording as the camera is
		# warming up
		frames = pipe.wait_for_frames()

	color_frame = frames.get_color_frame()
	color_intrin = color_frame.profile.as_video_stream_profile().intrinsics


	color_frame = np.asanyarray(color_frame.get_data())
	# Using cv2.imwrite() method 
	# Saving the image 
	filename="camera"+str(idx)+".png"
	cv2.imwrite(filename, cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)) 

	print(filename, color_intrin)
	pipe.stop()


