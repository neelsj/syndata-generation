import argparse
import glob
import sys
import os
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw 
import scipy

from multiprocessing import Pool
from multiprocessing import get_context

from functools import partial
import signal
import time
import json
from datetime import datetime
import csv
import shutil 

from defaults import *
sys.path.insert(0, POISSON_BLENDING_DIR)
from pb import *
import math
from pyblur3 import *
from collections import namedtuple

from pycocotools.coco import COCO

from itertools import combinations

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def randomAngle(kerneldim):
    """Returns a random angle used to produce motion blurring

    Args:
        kerneldim (int): size of the kernel used in motion blurring

    Returns:
        int: Random angle
    """ 
    kernelCenter = int(math.floor(kerneldim/2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])

def LinearMotionBlur3C(img):
    """Performs motion blur on an image with 3 channels. Used to simulate 
       blurring caused due to motion of camera.

    Args:
        img(NumPy Array): Input image with 3 channels

    Returns:
        Image: Blurred image by applying a motion blur with random parameters
    """
    lineLengths = [3,5,7,9]
    lineTypes = ["right", "left", "full"]
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    lineTypeIdx = np.random.randint(0, len(lineTypes)) 
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)
    blurred_img = img
    for i in range(3):
        blurred_img[:,:,i] = PIL2array1C(LinearMotionBlur(img[:,:,i], lineLength, lineAngle, lineType))
    blurred_img = Image.fromarray(blurred_img, 'RGB')
    return blurred_img

def overlap(a, b, max_allowed_iou):
    '''Find if two bounding boxes are overlapping or not. This is determined by maximum allowed 
       IOU between bounding boxes. If IOU is less than the max allowed IOU then bounding boxes 
       don't overlap

    Args:
        a(Rectangle): Bounding box 1
        b(Rectangle): Bounding box 2
    Returns:
        bool: True if boxes overlap else False
    '''
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    
    if (dx>=0) and (dy>=0) and float(dx*dy) > max_allowed_iou*(a.xmax-a.xmin)*(a.ymax-a.ymin):
        return True
    else:
        return False

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

def get_mask_file(img_file):
    '''Takes an image file name and returns the corresponding mask file. The mask represents
       pixels that belong to the object. Default implentation assumes mask file has same path 
       as image file with different extension only. Write custom code for getting mask file here
       if this is not the case.

    Args:
        img_file(string): Image name
    Returns:
        string: Correpsonding mask file path
    '''
    mask_file = img_file.replace('.jpg','_mask.jpg')
    return mask_file

def get_labels(imgs):
    '''Get list of labels/object names. Assumes the images in the root directory follow root_dir/<object>/<image>
       structure. Directory name would be object name.

    Args:
        imgs(list): List of images being used for synthesis 
    Returns:
        list: List of labels/object names corresponding to each image
    '''
    labels = []
    for img_file in imgs:        
        img_file = img_file.replace("\\", "/")
        label = img_file.split('/')[-2]
        labels.append(label)
    return labels

def get_annotation_from_mask_file(mask_file, scale=1.0):
    '''Given a mask file and scale, return the bounding box annotations

    Args:
        mask_file(string): Path of the mask file
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    if os.path.exists(mask_file):
        mask = cv2.imread(mask_file)
        if INVERTED_MASK:
            mask = 255 - mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if len(np.where(rows)[0]) > 0:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return int(scale*xmin), int(scale*xmax), int(scale*ymin), int(scale*ymax)
        else:
            return -1, -1, -1, -1
    else:
        print ("%s not found. Using empty mask instead." % mask_file)
        return -1, -1, -1, -1

def get_annotation_from_mask(mask):
    '''Given a mask, this returns the bounding box annotations

    Args:
        mask(NumPy Array): Array with the mask
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if len(np.where(rows)[0]) > 0:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, xmax, ymin, ymax
    else:
        return -1, -1, -1, -1

def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])

def PIL2array3C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def create_image_anno_wrapper(objects, args):
   ''' Wrapper used to pass params to workers
   '''
   return create_image_anno(*objects, args)

def create_image_anno_spatial_pairs_wrapper(objects, args):
   ''' Wrapper used to pass params to workers
   '''
   return create_image_spatial_pairs_anno(*objects, args)

def create_image_anno_spatial_wrapper(objects, args):
   ''' Wrapper used to pass params to workers
   '''
   return create_image_spatial_anno(*objects, args)

def crop_resize(im, desired_size):
    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = max(desired_size)/min(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.Resampling.LANCZOS)

    im = im.resize(new_size, Image.Resampling.LANCZOS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (desired_size[0], desired_size[1]))
    new_im.paste(im, ((desired_size[0]-new_size[0])//2,
                        (desired_size[1]-new_size[1])//2))

    return new_im

def create_coco_annotation(xmin, ymin, xmax, ymax, image_id, obj, annotation_id):

    annotation = {
        'iscrowd': False,
        'image_id': image_id,
        'category_id':  obj,
        'id': annotation_id,
        'bbox': [xmin, ymin, xmax-xmin, ymax-ymin],
        'area': (xmax-xmin)*(ymax-ymin)
    }

    return annotation

def render_objects(backgrounds, all_objects, min_scale, max_scale, args, already_syn=[], annotations=None, image_id=None):

    blending_list = args.blending_list
    w = args.width
    h = args.height
    max_scale = min(max_scale,1.0)

    rendered = False
    annotation_id = 0
    for idx, obj in enumerate(all_objects):

        if (os.path.isfile(obj[0])):
            foreground = Image.open(obj[0])
        else:
            foreground = Image.open(obj[0].replace("jpg", "JPEG"))

        xmin, xmax, ymin, ymax = get_annotation_from_mask_file(get_mask_file(obj[0]))
        if xmin == -1 or ymin == -1 or xmax-xmin < args.min_width or ymax-ymin < args.min_height :
            continue
        foreground = foreground.crop((xmin, ymin, xmax, ymax))
        orig_w, orig_h = foreground.size
        mask_file =  get_mask_file(obj[0])
        mask = Image.open(mask_file)
        mask = mask.crop((xmin, ymin, xmax, ymax))
        if INVERTED_MASK:
            mask = Image.fromarray(255-PIL2array1C(mask)).convert('1')
        o_w, o_h = orig_w, orig_h

        additional_scale = min(w/o_w,h/o_h)

        if args.scale:
            attempt_scale = 0
            while True:
                attempt_scale +=1
                frot
                if (args.stats and image_id):                                  
                    scale = np.clip(1-np.random.lognormal(args.stats["s"]["mean"], args.stats["s"]["std"]), min_scale, max_scale)*additional_scale
                else:
                    scale = random.uniform(min_scale, max_scale)*additional_scale

                o_w, o_h = int(scale*orig_w), int(scale*orig_h)
                if  w-o_w > 0 and h-o_h > 0 and o_w > 0 and o_h > 0:
                    foreground = foreground.resize((o_w, o_h), Image.Resampling.LANCZOS)
                    mask = mask.resize((o_w, o_h), Image.Resampling.LANCZOS)
                    break
                if attempt_scale == MAX_ATTEMPTS_TO_SYNTHESIZE:
                    o_w = w
                    o_h = h
                    foreground = foreground.resize((o_w, o_h), Image.Resampling.LANCZOS)
                    mask = mask.resize((o_w, o_h), Image.Resampling.LANCZOS)
                    break           
                
        if args.rotation:
            max_degrees = args.max_degrees 
            attempt_rotation = 0
            while True:
                attempt_rotation +=1
                rot_degrees = random.randint(-max_degrees, max_degrees)
                foreground_tmp = foreground.rotate(rot_degrees, expand=True)
                mask_tmp = mask.rotate(rot_degrees, expand=True)
                o_w, o_h = foreground_tmp.size
                if  w-o_w > 0 and h-o_h > 0:
                    mask = mask_tmp
                    foreground = foreground_tmp
                    break
                if attempt_rotation == MAX_ATTEMPTS_TO_SYNTHESIZE:
                    o_w, o_h = foreground.size
                    break  
        xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
        
        attempt_place = 0
        while True:
            attempt_place +=1

            # gaussian perterb from placing centered
            if (args.stats and image_id):  
                x = int(w/2-(xmax-xmin)/2) + int(np.random.normal(args.stats["x"]["mean"]*w, args.stats["x"]["std"]*w))
                y = int(h/2-(ymax-ymin)/2) + int(np.random.normal(args.stats["y"]["mean"]*h, args.stats["y"]["std"]*h))
            elif (args.gaussian_trans):
                x = int(w/2-(xmax-xmin)/2) + int(np.random.normal(args.gaussian_trans_mean[0]*w, args.gaussian_trans_std[0]*w))
                y = int(h/2-(ymax-ymin)/2) + int(np.random.normal(args.gaussian_trans_mean[1]*h, args.gaussian_trans_std[1]*h))
            else:
                x = random.randint(int(-args.max_truncation_fraction*o_w), int(w-o_w+args.max_truncation_fraction*o_w))
                y = random.randint(int(-args.max_truncation_fraction*o_h), int(h-o_h+args.max_truncation_fraction*o_h))

            found = True
            if args.dontocclude:
                for prev in already_syn:
                    ra = Rectangle(prev[0], prev[2], prev[1], prev[3])
                    rb = Rectangle(x+xmin, y+ymin, x+xmax, y+ymax)
                    if overlap(ra, rb, args.max_allowed_iou):
                            found = False
                            break
                if found:
                    break
            else:
                break
            if attempt_place == MAX_ATTEMPTS_TO_SYNTHESIZE:
                break

        if (found):
            if args.dontocclude:
                already_syn.append([x+xmin, x+xmax, y+ymin, y+ymax])

            rendered = True

            for i in range(len(blending_list)):
                if blending_list[i] == 'none' or blending_list[i] == 'motion':
                    backgrounds[i].paste(foreground, (x, y), mask)
                elif blending_list[i] == 'poisson':
                    offset = (y, x)
                    img_mask = PIL2array1C(mask)
                    img_src = PIL2array3C(foreground).astype(np.float64)
                    img_target = PIL2array3C(backgrounds[i])
                    img_mask, img_src, offset_adj \
                        = create_mask(img_mask.astype(np.float64),
                            img_target, img_src, offset=offset)
                    background_array = poisson_blend(img_mask, img_src, img_target,
                                    method='normal', offset_adj=offset_adj)
                    backgrounds[i] = Image.fromarray(background_array, 'RGB') 
                elif blending_list[i] == 'gaussian':
                    backgrounds[i].paste(foreground, (x, y), Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask),(5,5),2)))
                elif blending_list[i] == 'box':
                    backgrounds[i].paste(foreground, (x, y), Image.fromarray(cv2.blur(PIL2array1C(mask),(3,3))))

            if (annotations is not None):  
                
                xmin = int(max(1,xmin))                       
                ymin = int(max(1,ymin))           
                xmax = int(min(w,xmax))           
                ymax = int(min(h,ymax))

                annotation = create_coco_annotation(xmin, ymin, xmax, ymax, image_id, str(obj[1]), annotation_id)

                annotations.append(annotation)
                annotation_id += 1

    return rendered, already_syn, annotations

def render_object_spatial(img, args):
    
    w = args.width
    h = args.height

    if (os.path.isfile(img)):
        foreground = Image.open(img)
    else:
        foreground = Image.open(img.replace("jpg", "JPEG"))

    mask_file =  get_mask_file(img)
    xmin, xmax, ymin, ymax = get_annotation_from_mask_file(mask_file)    
    mask = Image.open(mask_file)

    #foreground.show()
    #mask.show()

    foreground = foreground.crop((xmin, ymin, xmax, ymax))
    orig_w, orig_h = foreground.size

    mask = mask.crop((xmin, ymin, xmax, ymax))
    if INVERTED_MASK:
        mask = Image.fromarray(255-PIL2array1C(mask)).convert('1')
    o_w, o_h = orig_w, orig_h

    #foreground.show()
    #mask.show()

    additional_scale = min(0.4*w/o_w,0.4*h/o_h)

    if args.scale:
        scale = random.uniform(args.min_scale, args.max_scale)*additional_scale
    else:
        scale = additional_scale

    o_w, o_h = int(scale*orig_w), int(scale*orig_h)

    foreground = foreground.resize((o_w, o_h), Image.Resampling.LANCZOS)
    mask = mask.resize((o_w, o_h), Image.Resampling.LANCZOS)
                
    #foreground.show()
    #mask.show()

    if args.rotation:
        rot_degrees = random.randint(-args.max_degrees, args.max_degrees)
        foreground = foreground.rotate(rot_degrees, expand=True)
        mask = mask.rotate(rot_degrees, expand=True)

        xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)

        foreground = foreground.crop((xmin, ymin, xmax, ymax))
        mask = mask.crop((xmin, ymin, xmax, ymax))

    #foreground.show()
    #mask.show()

    return foreground, mask

def colorize_image(foregroundA, colorA):
    if (colorA == "red"):
        matrix = (2.5, 0, 0, 0,
                0, .25, 0, 0,
                0, 0, .25, 0)
        foregroundA = foregroundA.convert("RGB", matrix)
    elif (colorA == "green"):
        matrix = (.25, 0, 0, 0,
                0, 2.5, 0, 0,
                0, 0, .25, 0)
        foregroundA = foregroundA.convert("RGB", matrix)

    elif (colorA == "blue"):
        matrix = (.25, 0, 0, 0,
                0, .25, 0, 0,
                0, 0, 2.5, 0)

        foregroundA = foregroundA.convert("RGB", matrix)

    return foregroundA

def render_objects_spatial_pair(background, imgA, imgB, relation, colorA, colorB, args, image_id, annotations):

    w = args.width
    h = args.height

    foregroundA, maskA = render_object_spatial(imgA, args)
    foregroundB, maskB = render_object_spatial(imgB, args)

    if (relation == "left"):
        xA = w*.25
        yA = h*.5
        xB = w*.75
        yB = h*.5

    elif (relation == "right"):
        xA = w*.75
        yA = h*.5
        xB = w*.25
        yB = h*.5

    elif (relation == "above"):
        xA = w*.5
        yA = h*.25
        xB = w*.5
        yB = h*.75
    else:
        xA = w*.5
        yA = h*.75
        xB = w*.5
        yB = h*.25

    if args.translation:
        xA += random.uniform(-args.min_trans*w, args.min_trans*w)
        yA += random.uniform(-args.min_trans*h, args.min_trans*h)
        xB += random.uniform(-args.min_trans*w, args.min_trans*w)
        yB += random.uniform(-args.min_trans*h, args.min_trans*h)

    foregroundA = colorize_image(foregroundA, colorA)
    foregroundB = colorize_image(foregroundB, colorB)

    xAmin = int(xA-foregroundA.size[0]/2)
    yAmin = int(yA-foregroundA.size[1]/2)
    xAmax = xAmin+foregroundA.size[0]
    yAmax = yAmin+foregroundA.size[1]

    xBmin = int(xB-foregroundB.size[0]/2)
    yBmin = int(yB-foregroundB.size[1]/2)
    xBmax = xBmin+foregroundB.size[0]
    yBmax = yBmin+foregroundB.size[1]

    background.paste(foregroundA, (xAmin, yAmin), Image.fromarray(cv2.GaussianBlur(PIL2array1C(maskA),(5,5),2)))
    background.paste(foregroundB, (xBmin, yBmin), Image.fromarray(cv2.GaussianBlur(PIL2array1C(maskB),(5,5),2)))

    if (annotations is not None):
       
        xAmin = int(max(1,xAmin))                       
        yAmin = int(max(1,yAmin))           
        xAmax = int(min(w,xAmax))           
        yAmax = int(min(h,yAmax))

        classA = os.path.dirname(imgA).split("\\")[-1]

        annotation_id = 0
        annotation = create_coco_annotation(xAmin, yAmin, xAmax, yAmax, image_id, classA, annotation_id)
        annotations.append(annotation)
       
        xBmin = int(max(1,xBmin))                       
        yBmin = int(max(1,yBmin))           
        xBmax = int(min(w,xBmax))           
        yBmax = int(min(h,yBmax))

        classB = os.path.dirname(imgB).split("\\")[-1]

        annotation_id = 1
        annotation = create_coco_annotation(xBmin, yBmin, xBmax, yBmax, image_id, classB, annotation_id)
        annotations.append(annotation)
    
    if (args.draw_boxes):
        img = ImageDraw.Draw(background)

        lineWidth = 4
        xAmin = int(xA-foregroundA.size[0]/2)-lineWidth/2
        yAmin = int(yA-foregroundA.size[1]/2)-lineWidth/2
        xAmax = xAmin+foregroundA.size[0]+lineWidth
        yAmax = yAmin+foregroundA.size[1]+lineWidth

        img.rectangle([xAmin, yAmin, xAmax, yAmax], outline ="red", width=4)

        xBmin = int(xB-foregroundB.size[0]/2)-lineWidth/2
        yBmin = int(yB-foregroundB.size[1]/2)-lineWidth/2
        xBmax = xBmin+foregroundB.size[0]+lineWidth
        yBmax = yBmin+foregroundB.size[1]+lineWidth

        img.rectangle([xBmin, yBmin, xBmax, yBmax], outline ="yellow", width=4)

    #background.show()

    return

def render_objects_spatial(background, imgA, relation, args, image_id, annotations):

    w = args.width
    h = args.height

    foregroundA, maskA = render_object_spatial(imgA, args)
    
    if (relation == "left"):
        xA = w*.25
        yA = h*.5

    elif (relation == "right"):
        xA = w*.75
        yA = h*.5

    elif (relation == "top"):
        xA = w*.5
        yA = h*.25
    else:
        xA = w*.5
        yA = h*.75

    if args.translation:
        xA += random.uniform(-args.min_trans*w, args.min_trans*w)
        yA += random.uniform(-args.min_trans*h, args.min_trans*h)    

    xAmin = int(xA-foregroundA.size[0]/2)
    yAmin = int(yA-foregroundA.size[1]/2)
    xAmax = xAmin+foregroundA.size[0]
    yAmax = yAmin+foregroundA.size[1]

    background.paste(foregroundA, (xAmin, yAmin), Image.fromarray(cv2.GaussianBlur(PIL2array1C(maskA),(5,5),2)))
       
    xAmin = int(max(1,xAmin))                       
    yAmin = int(max(1,yAmin))           
    xAmax = int(min(w,xAmax))           
    yAmax = int(min(h,yAmax))

    classA = os.path.dirname(imgA).split("\\")[-1]

    annotation_id = 0
    annotation = create_coco_annotation(xAmin, yAmin, xAmax, yAmax, image_id, classA, annotation_id)
    annotations.append(annotation)

    if (args.draw_boxes):
        img = ImageDraw.Draw(background)
        
        lineWidth = 4
        xAmin = int(xA-foregroundA.size[0]/2)-lineWidth/2
        yAmin = int(yA-foregroundA.size[1]/2)-lineWidth/2
        xAmax = xAmin+foregroundA.size[0]+lineWidth
        yAmax = yAmin+foregroundA.size[1]+lineWidth

        img.rectangle([xAmin, yAmin, xAmax, yAmax], outline ="red", width=4)

    #background.show()

    return

def create_image_anno(objects, distractor_objects, img_file, bg_file, image_id, args):
    '''Add data augmentation, synthesizes images and generates annotations according to given parameters

    Args:
        objects(list): List of objects whose annotations are also important
        distractor_objects(list): List of distractor objects that will be synthesized but whose annotations are not required
        img_file(str): Image file name
        anno_file(str): Annotation file name
        bg_file(str): Background image path 
    '''
    #if 'none' not in img_file:
    #    return 
    
    if (args.one_type_per_image):
        img_file = os.path.join(objects[0][1], img_file)

    print ("Working on %s" % img_file)

    blending_list = args.blending_list
    w = args.width
    h = args.height

    assert len(objects) > 0

    annotations = []

    img_info = {}

    background = Image.open(bg_file)
    background = crop_resize(background, (w, h))

    backgrounds = []
    for i in range(len(blending_list)):
        backgrounds.append(background.copy())
       
    if(args.add_backgroud_distractors and len(distractor_objects) > 0):
        rendered, _, _ = render_objects(backgrounds, distractor_objects, args.min_distractor_scale, args.max_distractor_scale, args)

    already_syn = []
    rendered, already_syn, annotations = render_objects(backgrounds, objects, args.min_scale, args.max_scale, args, already_syn, annotations, image_id)

    if (args.add_distractors and len(distractor_objects) > 0):
        rendered, _, _ = render_objects(backgrounds, distractor_objects, args.min_distractor_scale, args.max_distractor_scale, args, already_syn)

    if (rendered):

        img_info["license"] = 0
        img_info["file_name"] = img_file
        img_info["width"] = w
        img_info["height"] = h
        img_info["id"] = image_id

        img_file = os.path.join(args.exp, img_file)
        for i in range(len(blending_list)):
            if blending_list[i] == 'motion':
                backgrounds[i] = LinearMotionBlur3C(PIL2array3C(backgrounds[i]))
            backgrounds[i].save(img_file.replace('none', blending_list[i]))

    return img_info, annotations

def create_image_spatial_pairs_anno(imgA, imgB, relation, backgroundImg, path, img_file, colorA, colorB, image_id, args):

    img_file = os.path.join(path, img_file)

    print ("Working on %s" % img_file)

    blending_list = args.blending_list
    w = args.width
    h = args.height

    annotations = []

    #background = Image.new(mode="RGB", size=(w, h), color=(128, 128, 128))
    background = Image.open(backgroundImg).resize((w, h))

    render_objects_spatial_pair(background, imgA, imgB, relation, colorA, colorB, args, image_id, annotations)

    os.makedirs(os.path.join(args.exp, path), exist_ok=True)
    img_file_full = os.path.join(args.exp, img_file)

    background.save(img_file_full)

    img_info = {}
    img_info["license"] = 0
    img_info["file_name"] = img_file
    img_info["width"] = w
    img_info["height"] = h
    img_info["id"] = image_id

    return img_info, annotations

def create_image_spatial_anno(imgA, relation, backgroundImg, path, img_file, image_id, args):

    img_file = os.path.join(path, img_file)

    print ("Working on %s" % img_file)

    blending_list = args.blending_list
    w = args.width
    h = args.height

    annotations = []

    img_info = {}

    #background = Image.new(mode="RGB", size=(w, h), color=(128, 128, 128))
    background = Image.open(backgroundImg).resize((w, h))

    render_objects_spatial(background, imgA, relation, args, image_id, annotations)

    os.makedirs(os.path.join(args.exp, path), exist_ok=True)
    img_file_full = os.path.join(args.exp, path, img_file)

    background.save(img_file_full)

    img_info = {}
    img_info["license"] = 0
    img_info["file_name"] = img_file
    img_info["width"] = w
    img_info["height"] = h
    img_info["id"] = image_id

    return img_info, annotations

def gen_syn_data(args):
    '''Creates list of objects and distrctor objects to be pasted on what images.
       Spawns worker processes and generates images according to given params

    Args:
        img_files(list): List of image files
        labels(list): List of labels for each image  
    '''

    w = args.width
    h = args.height   

    img_list = get_list_of_images(args.root) 
    labels = get_labels(img_list)
    unique_labels = sorted(set(labels))

    print ("Number of classes : %d" % len(unique_labels))
    print ("Number of images : %d" % len(img_list))

    img_files = list(zip(img_list, labels))

    N = int(max(np.ceil(args.max_objects*args.total_num/len(img_files)), 1))

    print("Objects will be used %d times" % N)

    if(args.one_type_per_image):
        img_by_labels = {}
        img_labels = []
        for l in unique_labels:
            img_by_labels[l] = []
            img_labels.append([])

            if not os.path.exists(os.path.join(args.exp, l)):
                os.makedirs(os.path.join(args.exp, l))

        for img_file in img_files:
            img_by_labels[img_file[1]].append(img_file)

        for j, key in enumerate(img_by_labels.keys()):
            for i in range(N):
                img_labels[j] = img_labels[j] + random.sample(img_by_labels[key], len(img_by_labels[key]))
    else:
        img_labels = []
        for i in range(N):
            img_labels = img_labels + random.sample(img_files, len(img_files))

    background_files = glob.glob(os.path.join(args.background_dir, '*/*.jpg')) 
    if (not background_files):
        background_files = glob.glob(os.path.join(args.background_dir, '*/*/*.jpg')) 

    print ("Number of background images : %d" % len(background_files))

    if (args.add_distractors or args.add_backgroud_distractors):
        distractor_list = get_list_of_images(args.distractor_dir) 
        distractor_labels = get_labels(distractor_list)
        distractor_files = list(zip(distractor_list, distractor_labels))

        print ("Number of distractor images : %d" % len(distractor_files))        
    else:
        distractor_files = []

    idx = 0
    img_files = []
    anno_files = []
    params_list = []

    while any(img_labels):
        # Get list of objects
        objects = []

        if (args.stats):
            n = int(np.round(max(np.random.lognormal(args.stats["n"]["mean"], args.stats["n"]["std"]), 0), 1))
        else:
            n = min(random.randint(args.min_objects, args.max_objects), len(img_labels))

        if (args.one_type_per_image):
            non_empty = [i for i in range(len(img_labels)) if len(img_labels[i])>0]
            nn = random.randint(0, len(non_empty)-1)
            nl = non_empty[nn]
            for i in range(n):
                objects.append(img_labels[nl].pop())
        else:
            for i in range(n):
                objects.append(img_labels.pop())

        # Get list of distractor objects 
        distractor_objects = []
        if (args.add_distractors or args.add_backgroud_distractors):
            n = min(random.randint(args.min_distractor_objects, args.max_distractor_objects), len(distractor_files))
            for i in range(n):
                distractor_objects.append(random.choice(distractor_files))
            #print ("Chosen distractor objects: %s" % distractor_objects)
        
        bg_file = random.choice(background_files)
        for blur in args.blending_list:
            img_file = '%i_%s-%s.jpg'%(idx,blur, os.path.splitext(os.path.basename(bg_file))[0])
            params = (objects, distractor_objects, img_file, bg_file, idx)
            params_list.append(params)
            img_files.append(img_file)
        idx += 1

        if (idx >= args.total_num):
            break    

    if (args.workers>1):

        partial_func = partial(create_image_anno_wrapper, args=args) 

        p = get_context("spawn").Pool(args.workers, init_worker)
        try:
            results = p.map(partial_func, params_list)
        except KeyboardInterrupt:
            print ("....\nCaught KeyboardInterrupt, terminating workers")
            p.terminate()
        else:
            p.close()
        p.join()
    else:
        results = []
        for object in params_list:
            img_info, annotations = create_image_anno(*object, args=args)
            results.append([img_info, annotations])

    return results, unique_labels

def get_article(a, color=None):

    if (color):
        return ("a " + color + " " + a)
    else:
        if (a[0] in ("a", "e", "i", "o", "u")):
            return "an " + a 
        else:
            return "a " + a

def get_relation(relation):
    if (relation in ("left", "right")):
        return " to the " + relation + " of "
    else:
        return " " + relation + " "

def mirror_relation(relation):
    if (relation == "left"):
        return "right"
    elif (relation == "right"):
        return "left"
    elif (relation == "above"):
        return "below"
    else:
        return "above"

def create_prompts(a, b, relation, background, colorA=None, colorB=None):

    background = background.replace("-", " ").replace("_", " ")

    prompta = get_article(a, colorA) + get_relation(relation) + get_article(b, colorB) + " in a " + background
    promptb = get_article(b, colorB) + get_relation(mirror_relation(relation)) + get_article(a, colorA) + " in a " + background

    prompts = [prompta, promptb]

    return prompts

def create_prompt(a, relation, background):

    background = background.replace("-", " ").replace("_", " ")

    prompta = get_article(a) + " on the " + relation + " in a " + background

    return prompta

def gen_syn_data_spatial_pairs(args):
    '''Creates list of objects and distrctor objects to be pasted on what images.
       Spawns worker processes and generates images according to given params

    Args:
        img_files(list): List of image files
        labels(list): List of labels for each image  
    '''

    w = args.width
    h = args.height   

    #img_list = get_list_of_images(args.root) 

    if (args.val):
        with open(os.path.join(args.root, "val.txt")) as f:
            img_list = f.readlines()
    else:
        with open(os.path.join(args.root, "train.txt")) as f:
            img_list = f.readlines()

    img_list = [os.path.join(args.root, b.strip()) for b in img_list]

    labels = get_labels(img_list)
    unique_labels = sorted(set(labels))

    #background_list = get_list_of_images(args.background_dir) 

    if (args.val):
        with open(args.background_dir + "val.txt") as f:
            background_list = f.readlines()
    else:
        with open(args.background_dir + "train.txt") as f:
            background_list = f.readlines()

    background_list = [os.path.join(args.background_dir, b.strip()) for b in background_list]

    background_labels = get_labels(background_list)
   
    unique_background_labels = sorted(set(background_labels))

    print ("Number of classes : %d" % len(unique_labels))
    print ("Number of input images : %d" % len(img_list))

    print ("Number of background classes : %d" % len(unique_background_labels))
    print ("Number of background images : %d" % len(background_list))

    img_files = list(zip(img_list, labels))
    background_files = list(zip(background_list, background_labels))

    labels_to_images = {}
    for label in labels:
        labels_to_images[label] = []

    for img_file in img_files:
        labels_to_images[img_file[1]].append(img_file[0])

    background_labels_to_images = {}
    for label in background_labels:
        background_labels_to_images[label] = []

    for img_file in background_files:
        background_labels_to_images[img_file[1]].append(img_file[0])

    #unique_labels = random.sample(unique_labels,10)
    #print ("Using number of classes : %d" % len(unique_labels))

    if (args.focus_objs):

        unique_labels = args.focus_objs.split(",")

        if (args.total_objs>0):
            unique_labels = random.sample(unique_labels,args.total_objs)

        pairs = list(combinations(unique_labels, 2))
        pairs = random.sample(pairs,args.total_objs)
        
        # pairs = []
        # for focus_obj in unique_labels:
        #     for obj in unique_labels:
        #         pairs.append((focus_obj, obj))
    else:
        
        if (args.total_objs>0):
            # unique_labelsA = random.sample(unique_labels,args.total_objs)
            # unique_labelsB = random.sample(list(set(unique_labels).difference(set(unique_labelsA))),args.total_objs)

            # pairs = []
            # for objA in unique_labelsA:
            #     for objB in unique_labelsB:

            #         pairs.append((objA, objB))

            pairs = list(combinations(unique_labels, 2))
            pairs = random.sample(pairs,args.total_objs)

        else:
            pairs = list(combinations(unique_labels, 2))

    relations = ["left", "right", "above", "below"]
    colors = ["red", "green", "blue"]

    obj_rels = []

    params_list = []

    tups = []
    for pair in pairs:
        #pick classes of backgrounds
        backgrounds = random.sample(unique_background_labels,args.total_backgrounds)        
        for background in backgrounds:
            for relation in relations:
                tups.append(pair + (background, relation))

    print ("Number of output images : %d" % (len(tups)*args.total_num))

    for idx, tup in enumerate(tups):

        #classes for A, B, and background
        A = tup[0]
        B = tup[1]
        background = tup[2]
        relation = tup[3]

        #print(labels_to_images[A])
        #print(labels_to_images[B])

        # pick instances of A and B
        if (len(labels_to_images[A]) >= args.total_num):
            imgAs = random.sample(labels_to_images[A], args.total_num)
        else:
            imgAs = random.choices(labels_to_images[A], k=args.total_num)
            
        if (len(labels_to_images[B]) >= args.total_num):
            imgBs = random.sample(labels_to_images[B], args.total_num)
        else:
            imgBs = random.choices(labels_to_images[B], k=args.total_num)

        #pick instances of background
        backgroundImgs = random.sample(background_labels_to_images[background],args.total_num)

        for i in range(args.total_num):

            imgA = imgAs[i]
            imgB = imgBs[i]
            backgroundImg =  backgroundImgs[i]

            path = ("val\%s_%s\%s\%s" if args.val else "train\%s_%s\%s\%s") % (A, B, relation, background)
            
            if (args.colors):

                colorA = random.sample(colors, 1)[0]
                colorB = random.sample(colors, 1)[0]

                img_file = '%s_%s_%s_%s_%s.jpg' % (colorA, os.path.splitext(os.path.basename(imgA))[0], colorB, os.path.splitext(os.path.basename(imgB))[0], os.path.splitext(os.path.basename(backgroundImg))[0])

                params = (imgA, imgB, relation, backgroundImg, path, img_file, colorA, colorB, idx)
                params_list.append(params)           

                prompts = create_prompts(A, B, relation, background, colorA, colorB)

            else:
                img_file = '%s_%s_%s.jpg' % (os.path.splitext(os.path.basename(imgA))[0], os.path.splitext(os.path.basename(imgB))[0], os.path.splitext(os.path.basename(backgroundImg))[0])

                params = (imgA, imgB, relation, backgroundImg, path, img_file, None, None, idx)
                params_list.append(params)           

                prompts = create_prompts(A, B, relation, background)

            output_file = os.path.join(path, img_file)
            obj_rels.append([output_file] + prompts)

    with open(os.path.join(args.exp, 'val.csv' if args.val else 'train.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for obj_rel in obj_rels:
            writer.writerow(obj_rel[0:2])
            writer.writerow([obj_rel[0], obj_rel[2]])

    if (args.workers>1):
        partial_func = partial(create_image_anno_spatial_pairs_wrapper, args=args) 

        p = get_context("spawn").Pool(args.workers, init_worker)
        try:
            results = p.map(partial_func, params_list)
        except KeyboardInterrupt:
            print ("....\nCaught KeyboardInterrupt, terminating workers")
            p.terminate()
        else:
            p.close()
        p.join()
    else:
        results = []
        for object in params_list:
            img_info, annotations = create_image_spatial_pairs_anno(*object, args=args)
            results.append([img_info, annotations])

    return results, unique_labels

def gen_syn_data_spatial(args):
    '''Creates list of objects and distrctor objects to be pasted on what images.
       Spawns worker processes and generates images according to given params

    Args:
        img_files(list): List of image files
        labels(list): List of labels for each image  
    '''

    w = args.width
    h = args.height   

    #img_list = get_list_of_images(args.root) 

    if (args.val):
        with open(os.path.join(args.root, "val.txt")) as f:
            img_list = f.readlines()
    else:
        with open(os.path.join(args.root, "train.txt")) as f:
            img_list = f.readlines()

    img_list = [os.path.join(args.root, b.strip()) for b in img_list]

    labels = get_labels(img_list)
    unique_labels = sorted(set(labels))

    #background_list = get_list_of_images(args.background_dir) 

    if (args.val):
        with open(args.background_dir + "val.txt") as f:
            background_list = f.readlines()
    else:
        with open(args.background_dir + "train.txt") as f:
            background_list = f.readlines()

    background_list = [os.path.join(args.background_dir, b.strip()) for b in background_list]

    background_labels = get_labels(background_list)
   
    unique_background_labels = sorted(set(background_labels))

    print ("Number of classes : %d" % len(unique_labels))
    print ("Number of input images : %d" % len(img_list))

    print ("Number of background classes : %d" % len(unique_background_labels))
    print ("Number of background images : %d" % len(background_list))

    img_files = list(zip(img_list, labels))
    background_files = list(zip(background_list, background_labels))

    labels_to_images = {}
    for label in labels:
        labels_to_images[label] = []

    for img_file in img_files:
        labels_to_images[img_file[1]].append(img_file[0])

    background_labels_to_images = {}
    for label in background_labels:
        background_labels_to_images[label] = []

    for img_file in background_files:
        background_labels_to_images[img_file[1]].append(img_file[0])

    #unique_labels = random.sample(unique_labels,10)
    #print ("Using number of classes : %d" % len(unique_labels))

    if (args.total_objs>0):
        unique_labels = random.sample(unique_labels,args.total_objs)

    if (args.focus_objs):
        pairs = args.focus_objs.split(",")
    else:
        pairs = unique_labels

    relations = ["left", "right", "top", "bottom"]

    obj_rels = []

    params_list = []

    #pick classes of 
 #   backgrounds = random.sample(unique_background_labels,args.total_backgrounds)

    tups = []
    for pair in pairs:
        backgrounds = random.sample(unique_background_labels,args.total_backgrounds)
        for background in backgrounds:
            for relation in relations:
                tups.append((pair, background, relation))

    print ("Number of output images : %d" % (len(tups)*args.total_num))

    for idx, tup in enumerate(tups):

        #classes for A, B, and background
        A = tup[0]
        background = tup[1]
        relation = tup[2]

        #print(labels_to_images[A])
        #print(labels_to_images[B])

        # pick instances of A and B
        if (len(labels_to_images[A]) >= args.total_num):
            imgAs = random.sample(labels_to_images[A], args.total_num)
        else:
            imgAs = random.choices(labels_to_images[A], k=args.total_num)
           
        #pick instances of background

        if (args.total_num > len(background_labels_to_images[background])):
            backgroundImgs = np.random.choice(background_labels_to_images[background],args.total_num, replace=True)
        else:
            backgroundImgs = random.sample(background_labels_to_images[background],args.total_num)

        for i in range(args.total_num):

            imgA = imgAs[i]
            backgroundImg =  backgroundImgs[i]

            path = ("val\%s\%s\%s" if args.val else "train\%s\%s\%s") % (A, relation, background)
            img_file = '%s_%s.jpg' % (os.path.splitext(os.path.basename(imgA))[0], os.path.splitext(os.path.basename(backgroundImg))[0])

            params = (imgA, relation, backgroundImg, path, img_file, idx)
            params_list.append(params)           

            prompt = create_prompt(A, relation, background)

            output_file = os.path.join(path, img_file)
            obj_rels.append([output_file, prompt])

    with open(os.path.join(args.exp, 'val.csv' if args.val else 'train.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for obj_rel in obj_rels:
            writer.writerow(obj_rel)

    if (args.workers>1):
        partial_func = partial(create_image_anno_spatial_wrapper, args=args) 

        p = get_context("spawn").Pool(args.workers, init_worker)
        try:
            p.map(partial_func, params_list)
        except KeyboardInterrupt:
            print ("....\nCaught KeyboardInterrupt, terminating workers")
            p.terminate()
        else:
            p.close()
        p.join()
    else:
        results = []
        for object in params_list:
            img_info, annotations = create_image_spatial_anno(*object, args=args)
            results.append([img_info, annotations])

    return results, unique_labels

def init_worker():
    '''
    Catch Ctrl+C signal to terminate workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)
 
def write_coco(results, unique_labels):
    ''' Generate synthetic dataset according to given args
    '''  
   
    categories = []
    label_to_id = {}

    for i, unique_label in enumerate(unique_labels):
        categories.append({"supercategory": "none", "id": i, "name": unique_label})
        label_to_id[unique_label] = i

    img_infos = []
    annotations = []
    annotation_id = 0
    for img_info, img_annotations in results:

        if (img_info):
            img_infos.append(img_info)

            for img_annotation in img_annotations:
                img_annotation["id"] = annotation_id
                img_annotation["category_id"] = label_to_id[img_annotation["category_id"]]
                annotation_id += 1
                annotations.append(img_annotation)

    print("saving annotations to coco as json ")
    ### create COCO JSON annotations
    my_dict = {}
    my_dict["info"] = COCO_INFO
    my_dict["licenses"] = COCO_LICENSES
    my_dict["images"] = img_infos
    my_dict["categories"] = categories
    my_dict["annotations"] = annotations

    # TODO: specify coco file locaiton 
    output_file_path = os.path.join(args.exp,"coco_instances.json")
    with open(output_file_path, 'w+') as json_file:
        json_file.write(json.dumps(my_dict))

    print(">> complete. find coco json here: ", output_file_path)

def generate_synthetic_dataset(args):
    ''' Generate synthetic dataset according to given args
    '''  
    
    results, unique_labels = gen_syn_data(args)  

    write_coco(results, unique_labels)

def coco_from_imagefolder(args):
    
    img_list = get_list_of_images(args.root) 
    labels = get_labels(img_list)
    class_list = sorted(set(labels))

    print ("Number of classes : %d" % len(class_list))
    print ("Number of images : %d" % len(img_list))

    coco_json = {"info":COCO_INFO, "licenses": COCO_LICENSES, "images": [], "categories": [], "annotations": []}

    label_to_id = {}

    for j, cls in enumerate(class_list):
        coco_json["categories"].append({"name": cls, "id": j})
        label_to_id[cls] = j

    for j, img_path_full in enumerate(img_list):
        img_dir, img = os.path.split(img_path_full)   
        cls = os.path.basename(img_dir)
        img_path = os.path.join(cls, img)
        cls_dir = os.path.join(args.root, cls)

        print(img_path)        
        w, h = Image.open(os.path.join(cls_dir, img)).size
        coco_json["images"].append({"id": j, "file_name": img_path, "width": w, "height": h, "classes": [label_to_id[cls]]})

    print("saving annotations to coco as json ")

    output_file_path = os.path.join(args.root,"annotations.json")
    with open(output_file_path, "w") as f:
        json.dump(coco_json, f)

    print(">> complete. find coco json here: ", output_file_path)

def center_crop(img):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]
    sz = min(width, height)

    # process crop width and height for max available dimension
    crop_width = sz
    crop_height = sz
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img
    
from PIL import Image, ImageDraw, ImageFont

def draw_image(image, label):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    font = ImageFont.truetype("arial.ttf", size=16)

    draw = ImageDraw.Draw(image)
    draw.text((5, 5), label, fill=(255,255,0), font=font)

    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

import random

def image_thumbnails(path, num_per_dir=1, thumbsize=128, max_w=8, display_label=False):
    
    thumbs_files = []

    dirs = os.listdir(path)

    if (os.path.isdir(os.path.join(path, dirs[0]))):

        num_dirs = len(dirs)

        if (num_per_dir*num_dirs < 8):
            num_per_dir = int(8 / num_dirs)
            print("num_per_dir %d" % num_per_dir)

        # This would print all the files and directories
        for dir in dirs:       
            img_list = glob.glob(os.path.join(path, dir, '*_mask.jpg'))
            img_list = [img.replace("_mask", "") for img in img_list]

            if (len(img_list) == 0):
                img_list = glob.glob(os.path.join(path, dir, '*.jpg'))

            if (len(img_list) >=1 ):

                for i in range(min(num_per_dir, len(img_list))):
                    file = os.path.join(path, dir, img_list[i])
                    thumbs_files.append(file)
       
    else:
        img_list = dirs

        ind = np.random.permutation(len(img_list))
        ind = ind[0:100]

        for i in ind:
            file = os.path.join(path, img_list[i])
            thumbs_files.append(file)

    if (display_label):
        random.shuffle(thumbs_files)
        
    print(thumbs_files)

    w = max_w
    h = int(np.ceil(len(thumbs_files)/w))
    collage = np.ones((thumbsize*h,thumbsize*w,3))

    i = 0
    while(thumbs_files):
        thumbs_file = thumbs_files.pop(0).replace("\\", "/")
        im = cv2.imread(thumbs_file)

        if (im is not None):
            im = center_crop(im)
            im = cv2.resize(im, (thumbsize,thumbsize))

            if (display_label):
                im = draw_image(im, os.path.split(thumbs_file)[0].split("/")[-1])

            r = i//w
            c = i - r*w
            x = r*thumbsize
            y = c*thumbsize   

            collage[x:x+thumbsize,y:y+thumbsize,:] = im/255

            #cv2.imshow("", collage)
            #cv2.waitKey(0)

            i += 1

    cv2.imwrite(os.path.join(path, "collage.jpg"), np.uint8(collage*255))

         
def change_fgvc_classes(annotations_file_src, annotations_file_dst, annotations_file_out):
 
    old_to_new_name = {}

    with open("E:/Research/Images/FineGrained/fgvc-aircraft-2013b/data/manufacturers_variants.txt", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')            
        for row in reader:
            old_to_new_name[row[1]] = row[0] + " " + row[1].replace("/","-")

    with open(annotations_file_src, 'rt', encoding='UTF-8') as annotations:
        coco_src = json.load(annotations)        
        images = coco_src['images']
        annotation_srcs = coco_src['annotations']
        categories_src = coco_src['categories']    

    class_to_id = {}
    for cls in categories_src:
        class_to_id[cls["name"]] = cls["id"]

    with open(annotations_file_dst, 'rt', encoding='UTF-8') as annotations:
        coco_dst = json.load(annotations)        
        images_dst = coco_dst['images']
        annotations_dst = coco_dst['annotations']
        categories_dst = coco_dst['categories']

    old_to_new_id = {}

    for class_old in categories_dst:
        class_id_old = class_old["id"]
        class_name_old = class_old["name"]
        class_name_new = old_to_new_name[class_name_old]
        class_id_new = class_to_id[class_name_new]

        old_to_new_id[class_id_old] = class_id_new

    for i in range(len(images_dst)):
        classes = images_dst[i]["classes"]
        classes_new = [old_to_new_id[class_id_old] for class_id_old in classes]
        images_dst[i]["classes"] = classes_new

    for i in range(len(annotations_dst)):
        class_id_old = annotations_dst[i]["category_id"]
        class_id_new = old_to_new_id[class_id_old]
        annotations_dst[i]["category_id"] = class_id_new

    coco_dst['images'] = images_dst
    coco_dst['annotations'] = annotations_dst
    coco_dst['categories'] = categories_src

    with open(annotations_file_out, "w") as f:
        json.dump(coco_dst, f)

    print(">> complete. find coco json here: ", annotations_file_out)

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

def showAnns(coco, file_name, anns):

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in anns:
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]

            [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4,2))
            polygons.append(Polygon(np_poly))
            color.append(c)

            category_id = ann['category_id']
            category_name = coco.cats[category_id]

            plt.text(bbox_x, bbox_y-10, file_name + " " + str(category_name))

        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)

def displayCoco(in_path, annFile):

    # Initialize the COCO api for instance annotations
    coco = COCO(os.path.join(in_path,  annFile))
   
    # Load the categories in a variable
    imgIds = coco.getImgIds()
    print("Number of images:", len(imgIds))

    # load and display a random image
    for i in range(len(imgIds)):
        img = coco.loadImgs(imgIds[i])[0]

        I = Image.open(in_path + "/" + img['file_name'])

        plt.imshow(I)
        plt.axis('off')

        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        showAnns(coco, img['file_name'], anns)

        plt.waitforbuttonpress()
        plt.clf()

def parse_args():
    '''Parse input arguments
    '''
    parser = argparse.ArgumentParser(description="Create dataset with different augmentations")

    parser.add_argument("root",
      help="The root directory which contains the images and annotations.")
    parser.add_argument("--exp",
      help="The directory where images and annotation lists will be created.")
    
    parser.add_argument("--scale",
      help="Add scale augmentation.Default is not to add scale augmentation.", action="store_true")
    parser.add_argument("--rotation",
      help="Add rotation augmentation.Default is not to add rotation augmentation.", action="store_true")
    parser.add_argument("--translation",
      help="Add rotation augmentation.Default is not to add translate augmentation.", action="store_true")
    parser.add_argument("--dontocclude",
      help="Add objects without occlusion. Default is to produce occlusions", action="store_true")
    parser.add_argument("--add_distractors",
      help="Add distractors objects. Default is to not use distractors", action="store_true")
    parser.add_argument("--add_backgroud_distractors",
      help="Add distractors in the background", action="store_true")

    parser.add_argument("--create_masks",
      help="Create object masks", action="store_true")

    parser.add_argument("--create_masks_coco",
      help="Create object masks", action="store_true")

    parser.add_argument("--background",
      help="Create object masks", action="store_true")

    parser.add_argument("--create_json",
      help="Create coco JSON", action="store_true")

    # Paths
    parser.add_argument('--background_dir', default='E:/Source/EffortlessCV/data/backgrounds/', type=str)
    parser.add_argument('--distractor_dir', default='E:/Source/EffortlessCV/data/distractors/', type=str)

    # Parameters for generator
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument("--blending_list", nargs="+", default=['gaussian']) # can be ['gaussian','poisson', 'none', 'box', 'motion']

    # Parameters for images
    parser.add_argument('--min_objects', default=1, type=int)
    parser.add_argument('--max_objects', default=1, type=int)
    parser.add_argument('--min_distractor_objects', default=0, type=int)
    parser.add_argument('--max_distractor_objects', default=2, type=int)
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--height', default=480, type=int)

    # Parameters for objects in images
    parser.add_argument('--min_scale', default=.5, type=float) # min scale for scale augmentation
    parser.add_argument('--max_scale', default=1.2, type=float) # max scale for scale augmentation
    parser.add_argument('--min_trans', default=.05, type=float) # min scale for scale augmentation
    parser.add_argument('--max_trans', default=.05, type=float) # max scale for scale augmentation
    parser.add_argument('--min_distractor_scale', default=.1, type=float) # min scale for scale augmentation
    parser.add_argument('--max_distractor_scale', default=.5, type=float) # max scale for scale augmentation
    parser.add_argument('--max_degrees', default=5, type=float) # max rotation allowed during rotation augmentation
    parser.add_argument('--max_truncation_fraction', default=0.1, type=float) # max fraction to be truncated = max_truncation_fraction*(width/height)
    parser.add_argument('--max_allowed_iou', default=.25, type=float) # IOU > max_allowed_iou is considered an occlusion
    parser.add_argument('--min_width', default=100, type=int) # Minimum width of object to use for data generation
    parser.add_argument('--min_height', default=100, type=int) # Minimum height of object to use for data generation

    parser.add_argument('--stats_file', default="", type=str)

    parser.add_argument("--gaussian_trans", action="store_true")
    parser.add_argument('--gaussian_trans_mean', nargs=2, default=(0,0), type=float) #in fraction of image dimension
    parser.add_argument('--gaussian_trans_std', nargs=2, default=(.01,.01), type=float) #in fraction of image dimension
    
    parser.add_argument("--one_type_per_image", action="store_true")

    parser.add_argument("--spatial_pairs", action="store_true")

    parser.add_argument("--spatial", action="store_true")

    parser.add_argument("--focus_objs", default="", type=str)

    parser.add_argument("--total_backgrounds", default=4, type=int)

    parser.add_argument("--total_num",
      help="Number of times each image will be in dataset", default=4, type=int)

    parser.add_argument("--total_objs", default=20, type=int)

    parser.add_argument("--colors", action="store_true")

    parser.add_argument("--draw_boxes", action="store_true")

    parser.add_argument("--val", action="store_true")

    args = parser.parse_args()
    return args

COCO_INFO = {
    "description": "",
    "url": "",
    "version": "1",
    "year": 2022,
    "contributor": "MSR CV Group",
    "date_created": datetime.now().strftime("%m/%d/%Y")
}

COCO_LICENSES = [{
    "url": "",
    "id": 0,
    "name": "License"
}]

if __name__ == '__main__':
    args = parse_args()
    args.stats = None

    if not os.path.exists(args.exp):
        os.makedirs(args.exp) 

    print("\ninput dir %s" % args.root)
    print("output dir %s\n" % args.exp)

    if (args.create_masks or args.create_masks_coco):
        from segment import generate_masks
        generate_masks(args.root, True, args.create_masks_coco)
    elif (args.create_json):
        coco_from_imagefolder(args)
    if (args.spatial_pairs):
        results, unique_labels = gen_syn_data_spatial_pairs(args)
        write_coco(results, unique_labels)
    elif (args.spatial):
        results, unique_labels = gen_syn_data_spatial(args) 
        write_coco(results, unique_labels)
    else:
        if (args.stats_file):
            print("using stats")
            
            with open(args.stats_file, 'rt', encoding='UTF-8') as stats:
                args.stats = json.load(stats)

            print(args.stats)

        generate_synthetic_dataset(args)

    #img_list = get_list_of_images(args.root) 

    #for img in img_list:
    #    print(img)
    #    mask_file =  get_mask_file(img)
    #    mask = Image.open(mask_file)
    #    mask = Image.fromarray(255-PIL2array1C(mask)).convert('1')

    #    w, h = mask.size

    #    mask = np.array(mask)
    #    avg = np.mean(mask)
    #    b = 4
    #    mw = int(w*.25)
    #    mh = int(h*.25)
    #    border = np.sum(mask[0:b,:]) + np.sum(mask[-b:,:]) + np.sum(mask[:,0:b]) + np.sum(mask[:,-b:])
    #    center = np.mean(mask[int(h/2)-mh:int(h/2)+mh,int(w/2)-mw:int(w/2)+mw])

    #    if not (avg < .1 or border > 0 or center < .1):

    #        img_target = img.replace(args.root, args.exp)
    #        mask_file_target = mask_file.replace(args.root, args.exp)

    #        if (not os.path.exists(os.path.dirname(img_target))):
    #            os.makedirs(os.path.dirname(img_target))

    #        shutil.copyfile(img, img_target)
    #        shutil.copyfile(mask_file, mask_file_target)
            
    #img_list = get_list_of_images(args.exp) 
    #labels = get_labels(img_list)
    #unique_labels = sorted(set(labels))

    #unique_labels_count = {}
    #unique_labels_files = {}
    #for label in unique_labels:
    #    unique_labels_count[label] = 0
    #    unique_labels_files[label] = []

    #for i, img in enumerate(img_list):
    #    label = labels[i]
    #    unique_labels_count[label] += 1
    #    unique_labels_files[label].append(img)
        
    #train = []
    #val = []

    #for label in unique_labels:
    #    print("%s:\t%d" % (label, unique_labels_count[label]))
    #    files = unique_labels_files[label]
    #    random.shuffle(files)

    #    s = max(int(len(files)*.2),2)
    #    v = sorted(files[:s])
    #    t = sorted(files[s:])

    #    val += v
    #    train += t

    #with open(os.path.join(args.exp, 'val.txt'), "w") as fd:
    #    for v in val:
    #        fd.write(v.replace(args.exp,""))
    #        fd.write("\n")
    
    #with open(os.path.join(args.exp, 'train.txt'), "w") as fd:
    #    for t in train:
    #        fd.write(t.replace(args.exp,""))
    #        fd.write("\n")

    #change_fgvc_classes("E:/Source/EffortlessCVData/planes/objects_benchmark/test.json", "E:/Source/EffortlessCVData/planes/annotations/trainval.json", "E:/Source/EffortlessCVData/planes/annotations/trainval_renumbered_classes.json")
    #change_fgvc_classes("E:/Source/EffortlessCVData/planes/objects_benchmark/test.json", "E:/Source/EffortlessCVData/planes/annotations/test.json", "E:/Source/EffortlessCVData/planes/annotations/test_renumbered_classes.json")
    #change_fgvc_classes("E:/Source/EffortlessCVData/planes/objects_benchmark/test.json", "E:/Source/EffortlessCVData/planes/train_bing50/annotations2.json", "E:/Source/EffortlessCVData/planes/train_bing50/annotations_renumbered_classes.json")

    #displayCoco("E:/Research/Images/FineGrained/fgvc-aircraft-2013b/data/images", "E:/Source/EffortlessCVData/planes/annotations/test.json")
    #displayCoco("E:/Research/Images/FineGrained/fgvc-aircraft-2013b/data/images", "E:/Source/EffortlessCVData/planes/annotations/test_renumbered_classes.json")
    #displayCoco("E:/Source/EffortlessCVData/planes/objects_benchmark", "E:/Source/EffortlessCVData/planes/objects_benchmark/test.json")
    
    #image_thumbnails("E:/Source/EffortlessCVData/planes/objects_benchmark", 1, max_w=6, display_label=True)