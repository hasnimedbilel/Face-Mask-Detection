import numpy as np
import argparse
import cv2
from os import listdir
from os.path import isfile, join

# # construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path_folder", required=True,
                help="path to image folder")

ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it

path = args["path_folder"]
images_in_folder = [f for f in listdir(path) if isfile(join(path, f))]
print(len(images_in_folder))

for img in images_in_folder:
    # print(img)
    # print(type(img))
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
            print("[INFO] Object found. Saving locally.")
            print(np.shape(roi_color))

            # Check if the image is not empty :
            if np.shape(roi_color)[0] != 0 and np.shape(roi_color)[1] != 0:
                cv2.imwrite(path+'/test_crop/' + str(startX) + str(startY) + '_faces.jpg', roi_color)
