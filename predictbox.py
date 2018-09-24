import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
from PIL import Image
import math

# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1] 
filename = dir_path +'/' +image_path
image_size=32
num_channels=3
images = []


##Prediction
def Pred(image):

	# Reading the image using OpenCV
	#image = cv2.imread(filename)
	# Resizing the image to our desired size and preprocessing will be done exactly as done during training
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = []

	image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
	images.append(image)
	images = np.array(images, dtype=np.uint8)
	images = images.astype('float32')
	images = np.multiply(images, 1.0/255.0) 

##CV OPEN


	
	
	# labels
	directory='training_data/'
	
	labels=[]
	
	for root, dirs, files in os.walk(directory):
	    for currentclass in dirs:
	        labels.append(currentclass)
	
	
	#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
	x_batch = images.reshape(1, image_size,image_size,num_channels)
	
	## Let us restore the saved model 
	sess = tf.Session()
	
	# Step-1: Recreate the network graph. At this step only graph is created.
	saver = tf.train.import_meta_graph('my-model.meta')
	
	# Step-2: Now let's load the weights saved using the restore method.
	saver.restore(sess, tf.train.latest_checkpoint('./'))
	
	# Accessing the default graph which we have restored
	graph = tf.get_default_graph()
	
	# Now, let's get hold of the op that we can be processed to get the output.
	# In the original network y_pred is the tensor that is the prediction of the network
	y_pred = graph.get_tensor_by_name("y_pred:0")
	
	## Let's feed the images to the input placeholders
	x= graph.get_tensor_by_name("input_images:0") 
	y_true = graph.get_tensor_by_name("y_true:0") 
	y_test_images = np.zeros((1, len(labels))) 
	
	
	### Creating the feed_dict that is required to be fed to calculate y_pred 
	feed_dict_testing = {x: x_batch, y_true: y_test_images}
	result=sess.run(y_pred, feed_dict=feed_dict_testing)
	
	# result is of this format [probabiliy_of_rose probability_of_sunflower]
	
	
	result= result.reshape([-1])
	top_k = result.argsort()[-5:][::-1]
	
	for i in top_k:
	    print(labels[i], result[i])

	return labels[0]
	#print(labels)
	#print(result)
	


def nothing(x):
	pass

# Reading the image using OpenCV
#sourceImage = cv2.imread(filename)

#REad image from camera
videoFrame = cv2.VideoCapture(0)

#imagge preprocessing

# Create a window to display the camera feed
cv2.namedWindow('Camera Output',cv2.WINDOW_NORMAL)
cv2.namedWindow('Hand',cv2.WINDOW_NORMAL)
cv2.namedWindow('HandTrain',cv2.WINDOW_NORMAL)

# TrackBars for fixing skin color of the person
cv2.createTrackbar('B for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('G for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('R for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('B for max', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('G for max', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('R for max', 'Camera Output', 0, 255, nothing)

# Default skin color values in natural lighting
# cv2.setTrackbarPos('B for min','Camera Output',52)
# cv2.setTrackbarPos('G for min','Camera Output',128)
# cv2.setTrackbarPos('R for min','Camera Output',0)
# cv2.setTrackbarPos('B for max','Camera Output',255)
# cv2.setTrackbarPos('G for max','Camera Output',140)
# cv2.setTrackbarPos('R for max','Camera Output',146)

# Default skin color values in indoor lighting
cv2.setTrackbarPos('B for min', 'Camera Output', 0)
cv2.setTrackbarPos('G for min', 'Camera Output', 130)
cv2.setTrackbarPos('R for min', 'Camera Output', 103)
cv2.setTrackbarPos('B for max', 'Camera Output', 255)
cv2.setTrackbarPos('G for max', 'Camera Output', 182)
cv2.setTrackbarPos('R for max', 'Camera Output', 130)

# cascade xml file for detecting palm. Haar classifier
#palm_cascade = cv2.CascadeClassifier('palm.xml')

_, prevHandImage = videoFrame.read()

keyPressed = -1

while(keyPressed):
	
	# previous values of cropped variable
	x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0
	
	#while 1:
	
	min_YCrCb = np.array([cv2.getTrackbarPos('B for min', 'Camera Output'),
	                             cv2.getTrackbarPos('G for min', 'Camera Output'),
	                             cv2.getTrackbarPos('R for min', 'Camera Output')], np.uint8)
	max_YCrCb = np.array([cv2.getTrackbarPos('B for max', 'Camera Output'),
	                             cv2.getTrackbarPos('G for max', 'Camera Output'),
	                             cv2.getTrackbarPos('R for max', 'Camera Output')], np.uint8)

	read, sourceImage = videoFrame.read()
	sourceImage=cv2.flip(sourceImage,1)
	
	if read:
		x1, y1, x2, y2 = 100, 100, 300, 300
		croppedImage = sourceImage[y1:y2, x1:x2]
	
	# Convert image to YCrCb
	imageYCrCb = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2YCR_CB)
	imageYCrCb = cv2.GaussianBlur(imageYCrCb, (5, 5), 0)
	
	# Find region with skin tone in YCrCb image
	skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
	
	# Do contour detection on skin region
	contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	# sorting contours by area. Largest area first.
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	
	# get largest contour and compare with largest contour from previous frame.
	# set previous contour to this one after comparison.
	cnt = contours[0]
	prevcnt = contours[0]
	
	# once we get contour, extract it without background into a new window called handTrainImage
	stencil = np.zeros(croppedImage.shape).astype(croppedImage.dtype)
	color = [255, 255, 255]
	cv2.fillPoly(stencil, [cnt], color)
	handTrainImage = cv2.bitwise_and(croppedImage, stencil)
	
	
	# crop coordinates for hand.
	x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)
	
	# place a rectange around the hand.
	cv2.rectangle(croppedImage, (x_crop, y_crop), (x_crop + w_crop, y_crop + h_crop), (0, 255, 0), 2)
	
	# if the crop area has changed drastically form previous frame, update it.
	if (abs(x_crop - x_crop_prev) > 50 or abs(y_crop - y_crop_prev) > 50 or
	            abs(w_crop - w_crop_prev) > 50 or abs(h_crop - h_crop_prev) > 50):
	    x_crop_prev = x_crop
	    y_crop_prev = y_crop
	    h_crop_prev = h_crop
	    w_crop_prev = w_crop
	
	# create crop image
	handImage = croppedImage.copy()[max(0, y_crop_prev - 50):y_crop_prev + h_crop_prev + 50,
	            max(0, x_crop_prev - 50):x_crop_prev + w_crop_prev + 50]
	
	# Training image with black background
	handTrainImage = handTrainImage[max(0, y_crop_prev - 15):y_crop_prev + h_crop_prev + 15,
	                 max(0, x_crop_prev - 15):x_crop_prev + w_crop_prev + 15]
	
	
	letterDetected = Pred(handTrainImage) 
	
	
	#cv2.putText(croppedImage, letterDetected, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
	
	
	# Display the source image and cropped image
	cv2.rectangle(sourceImage, (x1, y1), (x2, y2), (255,0,0), 2)
	cv2.imshow('Camera Output', sourceImage)
	cv2.imshow('Hand', handImage)
	cv2.imshow('HandTrain', handTrainImage)
	keyPressed = cv2.waitKey(30) 
		
cv2.destroyAllWindows()	
	

