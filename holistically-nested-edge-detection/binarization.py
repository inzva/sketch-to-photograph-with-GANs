import cv2
import os
import argparse
import numpy as np
import subprocess
from skimage.morphology import skeletonize, binary_erosion, binary_dilation

DEFAULT_THRESHOLD = 125

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", type=str, required=True,
	help="path to source directory")
ap.add_argument("-d", "--destination", type=str, required=True,
	help="path to destination directory")
ap.add_argument("-t", "--threshold", default=DEFAULT_THRESHOLD, type=int, required=False)
args = vars(ap.parse_args())

def skeleton_pipeline(image):
    image = image[:,:,0]/255
    skeleton = skeletonize(image).astype(int) * (-255) + 255
    skeleton = np.expand_dims(skeleton, axis=2)
    skeleton = np.concatenate([skeleton, skeleton, skeleton], axis=2)
    skeleton = np.uint8(skeleton)
    return skeleton

dir = args["source"]
file_names = os.listdir(dir)
threshold = args["threshold"]

for i, file_name in enumerate(file_names):
    
    if os.path.exists(os.path.join(args["destination"], file_name)):
        ##if os.path.exists(os.path.join(dir, file_name)):
            #move file to data if file already processed.
            ##bashCommand = "mv " + os.path.join(dir, file_name) + " ../../../data/team6/celeba/edges/" + file_name
            ##output = subprocess.check_output(['bash','-c', bashCommand])
        continue
    
    image = cv2.imread(os.path.join(dir, file_name))
    
    if image is None:
        continue
    im_binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    skeleton = skeleton_pipeline(im_binary[1])

  #  skeleton = binary_erosion(skeleton).astype(int)*255
  #  skeleton = binary_dilation(skeleton).astype(int)*255
                      
    #move file to data after file is processed.
    ##bashCommand = "mv " + os.path.join(dir, file_name) + " ../../../data/team6/celeba/edges/" + file_name
    ##output = subprocess.check_output(['bash','-c', bashCommand])

    cv2.imwrite(os.path.join(args["destination"], file_name), skeleton)
