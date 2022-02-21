import argparse
import glob
import sys
import os
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
import cv2
import numpy as np
import random
from PIL import Image
import scipy
from multiprocessing import Pool
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
from pyblur import *
from collections import namedtuple

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

def overlap(a, b):
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
    
    if (dx>=0) and (dy>=0) and float(dx*dy) > MAX_ALLOWED_IOU*(a.xmax-a.xmin)*(a.ymax-a.ymin):
        return True
    else:
        return False

def get_list_of_images(root_dir, N=1):
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
    img_list = glob.glob(os.path.join(root_dir, '*_mask.jpg'))
    img_list = [img.replace("_mask", "") for img in img_list]

    img_list_f = []
    for i in range(N):
        img_list_f = img_list_f + random.sample(img_list, len(img_list))
    return img_list_f

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

def write_imageset_file(exp_dir, img_files, anno_files):
    '''Writes the imageset file which has the generated images and corresponding annotation files
       for a given experiment

    Args:
        exp_dir(string): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        img_files(list): List of image files that were generated
        anno_files(list): List of annotation files corresponding to each image file
    '''
    with open(os.path.join(exp_dir,'train.txt'),'w') as f:
        for i in range(len(img_files)):
            f.write('%s %s\n'%(img_files[i], anno_files[i]))

def write_labels_file(exp_dir, labels):
    '''Writes the labels file which has the name of an object on each line

    Args:
        exp_dir(string): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        labels(list): List of labels. This will be useful while training an object detector
    '''
    unique_labels = ['__background__'] + sorted(set(labels))
    with open(os.path.join(exp_dir,'labels.txt'),'w') as f:
        for i, label in enumerate(unique_labels):
            f.write('%s\t%s\n'%(i, label))

def keep_selected_labels(img_files, labels):
    '''Filters image files and labels to only retain those that are selected. Useful when one doesn't 
       want all objects to be used for synthesis

    Args:
        img_files(list): List of images in the root directory
        labels(list): List of labels corresponding to each image
    Returns:
        new_image_files(list): Selected list of images
        new_labels(list): Selected list of labels corresponidng to each imahe in above list
    '''
    with open(SELECTED_LIST_FILE) as f:
        selected_labels = [x.strip() for x in f.readlines()]
    new_img_files = []
    new_labels = []
    for i in range(len(img_files)):
        if labels[i] in selected_labels:
            new_img_files.append(img_files[i])
            new_labels.append(labels[i])
    return new_img_files, new_labels

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

def crop_resize(im, desired_size):
    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = max(desired_size)/min(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (desired_size[0], desired_size[1]))
    new_im.paste(im, ((desired_size[0]-new_size[0])//2,
                        (desired_size[1]-new_size[1])//2))

    return new_im

def create_image_anno(objects, distractor_objects, img_file, anno_file, bg_file, args):
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
    
    print ("Working on %s" % img_file)
    if os.path.exists(anno_file):
        return anno_file
    
    blending_list = args.BLENDING_LIST
    w = args.WIDTH
    h = args.HEIGHT

    all_objects = objects + distractor_objects
    assert len(all_objects) > 0
    attempt = 0
    while True:
        top = Element('annotation')
        background = Image.open(bg_file)
        background = crop_resize(background, (args.WIDTH, args.HEIGHT))

        backgrounds = []
        for i in range(len(blending_list)):
            backgrounds.append(background.copy())
        
        if args.dontocclude:
            already_syn = []

        rendered = False

        for idx, obj in enumerate(all_objects):
           foreground = Image.open(obj[0])
           xmin, xmax, ymin, ymax = get_annotation_from_mask_file(get_mask_file(obj[0]))
           if xmin == -1 or ymin == -1 or xmax-xmin < args.MIN_WIDTH or ymax-ymin < args.MIN_HEIGHT :
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
                while True:
                    scale = random.uniform(args.MIN_SCALE, args.MAX_SCALE)*additional_scale
                    o_w, o_h = int(scale*orig_w), int(scale*orig_h)
                    if  w-o_w > 0 and h-o_h > 0 and o_w > 0 and o_h > 0:
                        break
                foreground = foreground.resize((o_w, o_h), Image.ANTIALIAS)
                mask = mask.resize((o_w, o_h), Image.ANTIALIAS)
           if args.rotation:
               max_degrees = args.MAX_DEGREES 
               while True:
                   rot_degrees = random.randint(-max_degrees, max_degrees)
                   foreground_tmp = foreground.rotate(rot_degrees, expand=True)
                   mask_tmp = mask.rotate(rot_degrees, expand=True)
                   o_w, o_h = foreground_tmp.size
                   if  w-o_w > 0 and h-o_h > 0:
                        break
               mask = mask_tmp
               foreground = foreground_tmp
           xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
           attempt = 0
           while True:
               attempt +=1
               x = random.randint(int(-args.MAX_TRUNCATION_FRACTION*o_w), int(w-o_w+args.MAX_TRUNCATION_FRACTION*o_w))
               y = random.randint(int(-args.MAX_TRUNCATION_FRACTION*o_h), int(h-o_h+args.MAX_TRUNCATION_FRACTION*o_h))
               if args.dontocclude:
                   found = True
                   for prev in already_syn:
                       ra = Rectangle(prev[0], prev[2], prev[1], prev[3])
                       rb = Rectangle(x+xmin, y+ymin, x+xmax, y+ymax)
                       if overlap(ra, rb):
                             found = False
                             break
                   if found:
                      break
               else:
                   break
               if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
                   break
           if args.dontocclude:
               already_syn.append([x+xmin, x+xmax, y+ymin, y+ymax])

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
           if idx >= len(objects):
               continue 
           object_root = SubElement(top, 'object')
           object_type = obj[1]
           object_type_entry = SubElement(object_root, 'name')
           object_type_entry.text = str(object_type)
           object_bndbox_entry = SubElement(object_root, 'bndbox')
           x_min_entry = SubElement(object_bndbox_entry, 'xmin')
           x_min_entry.text = '%d'%(max(1,x+xmin))
           x_max_entry = SubElement(object_bndbox_entry, 'xmax')
           x_max_entry.text = '%d'%(min(w,x+xmax))
           y_min_entry = SubElement(object_bndbox_entry, 'ymin')
           y_min_entry.text = '%d'%(max(1,y+ymin))
           y_max_entry = SubElement(object_bndbox_entry, 'ymax')
           y_max_entry.text = '%d'%(min(h,y+ymax))
           difficult_entry = SubElement(object_root, 'difficult')
           difficult_entry.text = '0' # Add heuristic to estimate difficulty later on

           rendered = True

        if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
           continue
        else:
           break

    if (rendered):
        for i in range(len(blending_list)):
            if blending_list[i] == 'motion':
                backgrounds[i] = LinearMotionBlur3C(PIL2array3C(backgrounds[i]))
            backgrounds[i].save(img_file.replace('none', blending_list[i]))

        xmlstr = xml.dom.minidom.parseString(tostring(top)).toprettyxml(indent="    ")
        with open(anno_file, "w") as f:
            f.write(xmlstr)

def gen_syn_data(img_files, labels, img_dir, anno_dir, args):
    '''Creates list of objects and distrctor objects to be pasted on what images.
       Spawns worker processes and generates images according to given params

    Args:
        img_files(list): List of image files
        labels(list): List of labels for each image
        img_dir(str): Directory where synthesized images will be stored
        anno_dir(str): Directory where corresponding annotations will be stored
    '''

    w = args.WIDTH
    h = args.HEIGHT
    background_dir = args.BACKGROUND_DIR
    background_files = glob.glob(os.path.join(background_dir, BACKGROUND_GLOB_STRING))
   
    print ("Number of background images : %s" % len(background_files))
    img_labels = list(zip(img_files, labels))
    random.shuffle(img_labels)

    print ("Number of images : %s" % len(img_labels))

    if args.add_distractors:
        with open(args.DISTRACTOR_LIST_FILE) as f:
            distractor_labels = [x.strip() for x in f.readlines()]

        distractor_list = []
        for distractor_label in distractor_labels:
            distractor_list += glob.glob(os.path.join(args.DISTRACTOR_DIR, distractor_label, DISTRACTOR_GLOB_STRING))

        distractor_list = [img.replace("_mask", "") for img in distractor_list]

        distractor_files = list(zip(distractor_list, len(distractor_list)*[None]))
        random.shuffle(distractor_files)

        print ("List of distractor files collected: %s" % distractor_files)
    else:
        distractor_files = []

    idx = 0
    img_files = []
    anno_files = []
    params_list = []

    images = []
    image_id = 1    

    annotations = []

    while len(img_labels) > 0:
        # Get list of objects
        objects = []
        n = min(random.randint(args.MIN_NO_OF_OBJECTS, args.MAX_NO_OF_OBJECTS), len(img_labels))
        for i in range(n):
            objects.append(img_labels.pop())
        # Get list of distractor objects 
        distractor_objects = []
        if args.add_distractors:
            n = min(random.randint(args.MIN_NO_OF_DISTRACTOR_OBJECTS, args.MAX_NO_OF_DISTRACTOR_OBJECTS), len(distractor_files))
            for i in range(n):
                distractor_objects.append(random.choice(distractor_files))
            print ("Chosen distractor objects: %s" % distractor_objects)

        idx += 1
        bg_file = random.choice(background_files)
        for blur in args.BLENDING_LIST:
            img_file = os.path.join(img_dir, '%i_%s-%s.jpg'%(idx,blur, os.path.splitext(os.path.basename(bg_file))[0]))
            anno_file = os.path.join(anno_dir, '%i.xml'%idx)
            params = (objects, distractor_objects, img_file, anno_file, bg_file)
            params_list.append(params)
            img_files.append(img_file)
            anno_files.append(anno_file)

    partial_func = partial(create_image_anno_wrapper, args=args) 

    if (args.NUMBER_OF_WORKERS>1):
        p = Pool(args.NUMBER_OF_WORKERS, init_worker)
        try:
            p.map(partial_func, params_list)
        except KeyboardInterrupt:
            print ("....\nCaught KeyboardInterrupt, terminating workers")
            p.terminate()
        else:
            p.close()
        p.join()
    else:
        for object in params_list:
            create_image_anno_wrapper(object, args=args)

    return img_files, anno_files

def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)
 
def generate_synthetic_dataset(args):
    ''' Generate synthetic dataset according to given args
    '''
    img_files = get_list_of_images(args.root, args.num) 
    labels = get_labels(img_files)

    if args.selected:
       img_files, labels = keep_selected_labels(img_files, labels)

    if not os.path.exists(args.exp):
        os.makedirs(args.exp)
    
    write_labels_file(args.exp, labels)

    anno_dir = os.path.join(args.exp, 'annotations')
    img_dir = os.path.join(args.exp, 'images')
    if not os.path.exists(os.path.join(anno_dir)):
        os.makedirs(anno_dir)
    if not os.path.exists(os.path.join(img_dir)):
        os.makedirs(img_dir)
    
    syn_img_files, anno_files = gen_syn_data(img_files, labels, img_dir, anno_dir, args)
    write_imageset_file(args.exp, syn_img_files, anno_files)    

import xml.etree.ElementTree as ET

def createCocoJSONFromXML(path):

    in_path = os.path.join(path, "images")

    labels_file = os.path.join(path, "labels.txt")
    labels_to_id = {}

    with open(labels_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = "\t")
            
        class_list = []

        for row in reader:
            labels_to_id[row[1]] = row[0]

    print(labels_to_id)

    categories = list(labels_to_id.keys())

    fileNameToCocoID = {}

    file_list = [f for f in os.listdir(in_path) if os.path.splitext(f)[1] == ".jpg"]

    image_id = 1
    annotation_id = 1
    
    images = []
    annotations = []

    for f in file_list:

        #image
        file_path = os.path.join(in_path, f)    
        image = Image.open(file_path)
        print(file_path)

        img_size = image.size    
        new_img={}
        new_img["license"] = 0
        new_img["file_name"] = f
        new_img["width"] = img_size[0]
        new_img["height"] = img_size[1]
        new_img["id"] = image_id
        images.append(new_img)
        
        dir_path, file_name = os.path.split(file_path)
        dir_path = dir_path.replace("images", "annotations")
        file_name = file_name.split("_")[0] + ".xml"

        xml_path = os.path.join(dir_path, file_name)
       
        tree = ET.parse(xml_path)
        annotation = tree.getroot()

        box = {}
        for object in annotation:

            category = object[0].text
            category_id = labels_to_id[category]

            box[object[1][0].tag] = int(object[1][0].text)
            box[object[1][1].tag] = int(object[1][1].text)
            box[object[1][2].tag] = int(object[1][2].text)
            box[object[1][3].tag] = int(object[1][3].text)

            annotation = {
                'iscrowd': False,
                'image_id': image_id,
                'category_id': category_id,
                'id': annotation_id,
                'bbox': [box['xmin'], box['ymin'], box['xmax']-box['xmin'], box['ymax']-box['ymin']],
                'area': (box['xmax']-box['xmin'])*(box['ymax']-box['ymin'])
            }

            annotations.append(annotation)
            annotation_id += 1

        image_id += 1

    print("saving annotations to coco as json ")
    ### create COCO JSON annotations
    my_dict = {}
    my_dict["info"]= COCO_INFO
    my_dict["licenses"]= COCO_LICENSES
    my_dict["images"]=images
    my_dict["categories"]=categories
    my_dict["annotations"]=annotations

    # TODO: specify coco file locaiton 
    output_file_path = os.path.join(in_path,"coco_instances.json")
    with open(output_file_path, 'w+') as json_file:
        json_file.write(json.dumps(my_dict))

    print(">> complete. find coco json here: ", output_file_path)
    print("last annotation id: ", annotation_id)
    print("last image_id: ", image_id)

def makeDirsFromXML(path):

    in_path = os.path.join(path, "images")

    file_list = [f for f in os.listdir(in_path) if os.path.splitext(f)[1] == ".jpg"]

    for f in file_list:

        #image
        file_path = os.path.join(in_path, f)    
        image = Image.open(file_path)
        print(file_path)
        
        dir_path, file_name = os.path.split(file_path)
        dir_path = dir_path.replace("images", "annotations")
        xml_file_name = file_name.split("_")[0] + ".xml"

        xml_path = os.path.join(dir_path, xml_file_name)
       
        tree = ET.parse(xml_path)
        annotation = tree.getroot()

        for object in annotation:
            category = object[0].text
        
            path_category = os.path.join(path, "classes", category)
            print(path_category)
            os.makedirs(path_category, exist_ok=True)
            shutil.copyfile(file_path, os.path.join(path_category, file_name))

def parse_args():
    '''Parse input arguments
    '''
    parser = argparse.ArgumentParser(description="Create dataset with different augmentations")
    parser.add_argument("root",
      help="The root directory which contains the images and annotations.")
    parser.add_argument("exp",
      help="The directory where images and annotation lists will be created.")
    parser.add_argument("--selected",
      help="Keep only selected instances in the test dataset. Default is to keep all instances in the root directory", action="store_true")
    parser.add_argument("--scale",
      help="Add scale augmentation.Default is to add scale augmentation.", action="store_false")
    parser.add_argument("--rotation",
      help="Add rotation augmentation.Default is to add rotation augmentation.", action="store_false")
    parser.add_argument("--num",
      help="Number of times each image will be in dataset", default=1, type=int)
    parser.add_argument("--dontocclude",
      help="Add objects without occlusion. Default is to produce occlusions", action="store_true")
    parser.add_argument("--add_distractors",
      help="Add distractors objects. Default is to not use distractors", action="store_true")

    # Paths
    parser.add_argument('--BACKGROUND_DIR', default='E:/Source/EffortlessCV/data/office/', type=str)
    parser.add_argument('--SELECTED_LIST_FILE', default='demo_data_dir/selected.txt', type=str)
    parser.add_argument('--DISTRACTOR_LIST_FILE', default='demo_data_dir/neg_list.txt', type=str)
    parser.add_argument('--DISTRACTOR_DIR', default='E:/Source/EffortlessCV/data/objects/', type=str)

    # Parameters for generator
    parser.add_argument('--NUMBER_OF_WORKERS', default=10, type=int)
    parser.add_argument("--BLENDING_LIST", nargs="+", default=['gaussian']) # can be ['gaussian','poisson', 'none', 'box', 'motion']

    # Parameters for images
    parser.add_argument('--MIN_NO_OF_OBJECTS', default=1, type=int)
    parser.add_argument('--MAX_NO_OF_OBJECTS', default=1, type=int)
    parser.add_argument('--MIN_NO_OF_DISTRACTOR_OBJECTS', default=1, type=int)
    parser.add_argument('--MAX_NO_OF_DISTRACTOR_OBJECTS', default=1, type=int)
    parser.add_argument('--WIDTH', default=640, type=int)
    parser.add_argument('--HEIGHT', default=480, type=int)

    # Parameters for objects in images
    parser.add_argument('--MIN_SCALE', default=.8, type=float) # min scale for scale augmentation
    parser.add_argument('--MAX_SCALE', default=1.5, type=float) # max scale for scale augmentation
    parser.add_argument('--MAX_DEGREES', default=5, type=float) # max rotation allowed during rotation augmentation
    parser.add_argument('--MAX_TRUNCATION_FRACTION', default=0, type=float) # max fraction to be truncated = MAX_TRUNCACTION_FRACTION*(WIDTH/HEIGHT)
    parser.add_argument('--MAX_ALLOWED_IOU', default=.75, type=float) # IOU > MAX_ALLOWED_IOU is considered an occlusion
    parser.add_argument('--MIN_WIDTH', default=100, type=int) # Minimum width of object to use for data generation
    parser.add_argument('--MIN_HEIGHT', default=100, type=int) # Minimum height of object to use for data generation

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
        
    #from segment import generate_masks
    #generate_masks(args.root)

    generate_synthetic_dataset(args)

    ##createCocoJSONFromXML(train_path_exp)
    ##makeDirsFromXML(train_path_exp)