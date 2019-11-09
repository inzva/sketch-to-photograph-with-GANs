# USAGE
# python detect_edges_image.py --edge-detector hed_model --image images/guitar.jpg

# import the necessary packages
import gc
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--edge-detector", type=str, required=True,
	help="path to OpenCV's deep learning edge detector")
ap.add_argument("-s", "--source", type=str, required=True,
	help="path to source directory")
ap.add_argument("-d", "--destination", type=str, required=True,
				help="path to destination directory")
ap.add_argument("-ts", "--target-size", type=int, required=True,
				help="target image size (nxn)")
args = vars(ap.parse_args())

class CropLayer(object):
	def __init__(self, params, blobs):
		# initialize our starting and ending (x, y)-coordinates of
		# the crop
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0

	def getMemoryShapes(self, inputs):
		# the crop layer will receive two inputs -- we need to crop
		# the first input blob to match the shape of the second one,
		# keeping the batch size and number of channels
		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])

		# compute the starting and ending crop coordinates
		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H

		# return the shape of the volume (we'll perform the actual
		# crop during the forward pass
		return [[batchSize, numChannels, H, W]]

	def forward(self, inputs):
		# use the derived (x, y)-coordinates to perform the crop
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]

# load our serialized edge detector from disk
print("[INFO] loading edge detector...")
protoPath = os.path.sep.join([args["edge_detector"],
	"deploy.prototxt"])
modelPath = os.path.sep.join([args["edge_detector"],
	"hed_pretrained_bsds.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# register our new layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)

dir = args["source"]
file_names = os.listdir(dir)

image = None
print("[INFO] performing holistically-nested edge detection...")
for i, file_name in enumerate(file_names):



	full_path = os.path.join(dir, file_name)
	dest_path = os.path.join(args["destination"], file_name)

	if os.path.exists(dest_path):
		continue

	#print(full_path)
	# load the input image and grab its dimensions
	image = cv2.imread(full_path)
	if image is None:
		continue
	image = cv2.resize(image, (args["target_size"],args["target_size"]))
	(H, W) = image.shape[:2]


	# construct a blob out of the input image for the Holistically-Nested
	# Edge Detector
	blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
		mean=(104.00698793, 116.66876762, 122.67891434),
		swapRB=False, crop=False)

	# set the blob as the input to the network and perform a forward pass
	# to compute the edges
	net.setInput(blob)
	hed = net.forward()
	hed = cv2.resize(hed[0, 0], (W, H))
	hed = (255 * hed).astype("uint8")

	cv2.imwrite(os.path.join(args["destination"], file_name), hed)
    
    #memory leak 16kb/s
	del full_path
	del dest_path
	del image
	del H
	del W
	del blob
	del hed
	gc.collect()

# python holistically-nested-edge-detection/detect_edges_image.py --edge-detector holistically-nested-edge-detection/hed_model --image holistically-nested-edge-detection/images/cat.jpg
