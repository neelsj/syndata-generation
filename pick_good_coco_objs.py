import argparse
import glob
import sys
import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw 
import scipy

import signal
import time
import json
from datetime import datetime
import csv
import shutil 

import cv2
import matplotlib.pyplot as plt

from itertools import combinations

def get_list_of_images(root_dir):
    '''Gets the list of images of objects in the root directory. The expected format 
       is root_dir/<object>/<image>.jpg. Adds an image as many times you want it to 
       appear in dataset.

    Args:
        root_dir(string): Directory where images of objects are present
        N(int): Number of times an image would appear in dataset. Each image should have
                different data augmentation
    Returns:
        list: List of images(with paths) that will be put in the dataset
    '''
    img_list = glob.glob(os.path.join(root_dir, '*/*_mask.jpg'))

    if (len(img_list) == 0):
        img_list = glob.glob(os.path.join(root_dir, '*/*.jpg'))
    else:
        img_list = [img.replace("_mask", "") for img in img_list]
    
    return img_list

if __name__ == '__main__':

    root_dir = "E:\\Source\\EffortlessCVSystem\\Data\\coco_objects_good\\"

    images = get_list_of_images(root_dir)
    
    picked_images = []
    
    for image in images:
        img = Image.open(image)

        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
        plt.show()