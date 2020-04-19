import numpy as np
import argparse
import cv2
from os import listdir
from os.path import isfile, join
import pandas as pd
import shutil
import os, sys

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path_folder", required=True,
	help="path to image folder")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it

predictions_imgs = []
predictions_scores = []

path = args["path_folder"]
images_in_folder = [f for f in listdir(path) if isfile(join(path, f))]
print(len(images_in_folder))

output_df = pd.DataFrame()

# load the tf Graph 
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	tf.import_graph_def(graph_def, name='')
	

for img in images_in_folder:
	# print(img)
	# print(type(img))
	one_pic_pred = 0.0
	os.mkdir("./temp_crop")

	image = cv2.imread(path+"/"+img)
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	#print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	temp_pred = []
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	 
			# draw the bounding box of the face along with the associated
			# probability
			# text = "{:.2f}%".format(confidence * 100)
			# y = startY - 10 if startY - 10 > 10 else startY + 10
			# cv2.rectangle(image, (startX, startY), (endX, endY),
			# 	(0, 0, 255), 2)
			# cv2.putText(image, text, (startX, y),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			roi_color = image[startY:endY, startX:endX]
			#print("[INFO] Object found. Saving locally.")
			print(np.shape(roi_color))
			if np.shape(roi_color)[0] != 0 and np.shape(roi_color)[1] != 0:
				cv2.imwrite("./temp_crop/"+ str(startX) + str(startY) + str(endY) + '_faces.jpg', roi_color)

				# now let's predict :
				image_path = "./temp_crop/"+ str(startX) + str(startY) + str(endY) + '_faces.jpg'

				# Read in the image_data
				image_data = tf.gfile.FastGFile(image_path, 'rb').read()

				with tf.Session() as sess:
				    # Feed the image_data as input to the graph and get first prediction
				    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
				    
				    predictions = sess.run(softmax_tensor, \
				             {'DecodeJpeg/contents:0': image_data})
				    
				    # Sort to show labels of first prediction in order of confidence
				    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
				    for node_id in top_k:
				        human_string = label_lines[node_id]
				        if human_string == '1' :
				        	score = predictions[0][node_id]
				        	temp_pred.append(score)
				        	print('%s (precision = %.5f)' % (human_string, score))
	if temp_pred != []:	
		one_pic_pred = max(temp_pred)
	else:
		one_pic_pred = 0.0

	if one_pic_pred>0.9 : 
		one_pic_pred = 1.0
	elif one_pic_pred<0.1 :
		one_pic_pred = 0.0

	predictions_imgs.append(img)
	predictions_scores.append(one_pic_pred)


	shutil.rmtree("./temp_crop/")

output_df["img"] = predictions_imgs
output_df["score"] = predictions_scores
output_df.to_csv("./output_scores.csv", index=False)





				



