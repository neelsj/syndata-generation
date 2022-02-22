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

def render_objects(backgrounds, all_objects, min_scale, max_scale, args, already_syn=[], annotations=None, image_id=None):

    blending_list = args.blending_list
    w = args.width
    h = args.height

    rendered = False
    annotation_id = 0
    for idx, obj in enumerate(all_objects):
        foreground = Image.open(obj[0])
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
                scale = random.uniform(min_scale, max_scale)*additional_scale
                o_w, o_h = int(scale*orig_w), int(scale*orig_h)
                if  w-o_w > 0 and h-o_h > 0 and o_w > 0 and o_h > 0:
                    foreground = foreground.resize((o_w, o_h), Image.ANTIALIAS)
                    mask = mask.resize((o_w, o_h), Image.ANTIALIAS)
                    break
                if attempt_scale == MAX_ATTEMPTS_TO_SYNTHESIZE:
                    o_w = w
                    o_h = h
                    foreground = foreground.resize((o_w, o_h), Image.ANTIALIAS)
                    mask = mask.resize((o_w, o_h), Image.ANTIALIAS)
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
                xmin = int(max(1,x+xmin))           
                xmax = int(min(w,x+xmax))           
                ymin = int(max(1,y+ymin))           
                ymax = int(min(h,y+ymax))

                annotation = {
                    'iscrowd': False,
                    'image_id': image_id,
                    'category_id':  str(obj[1]),
                    'id': annotation_id,
                    'bbox': [xmin, ymin, xmax-xmin, ymax-ymin],
                    'area': (xmax-xmin)*(ymax-ymin)
                }

                annotations.append(annotation)
                annotation_id += 1

    return rendered, already_syn, annotations

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
    
    print ("Working on %s" % img_file)

    blending_list = args.blending_list
    w = args.width
    h = args.height

    assert len(objects) > 0

    annotations = []

    img_info = {}
    img_info["license"] = 0
    img_info["file_name"] = img_file
    img_info["width"] = w
    img_info["height"] = h
    img_info["id"] = image_id

    background = Image.open(bg_file)
    background = crop_resize(background, (w, h))

    backgrounds = []
    for i in range(len(blending_list)):
        backgrounds.append(background.copy())
       
    if(args.add_backgroud_distractors and len(distractor_objects) > 0):
        rendered, _, _ = render_objects(backgrounds, distractor_objects, args.min_distractor_scale, args.max_distractor_scale, args)

    if args.dontocclude:
        already_syn = []

    rendered, already_syn, annotations = render_objects(backgrounds, objects, args.min_scale, args.max_scale, args, already_syn, annotations, image_id)

    if (args.add_distractors and len(distractor_objects) > 0):
        attempt, rendered, already_syn = render_objects(backgrounds, distractor_objects, already_syn, args.min_distractor_scale, args.max_distractor_scale, args, already_syn)

    if (rendered):
        img_file = os.path.join(args.exp, img_file)
        for i in range(len(blending_list)):
            if blending_list[i] == 'motion':
                backgrounds[i] = LinearMotionBlur3C(PIL2array3C(backgrounds[i]))
            backgrounds[i].save(img_file.replace('none', blending_list[i]))

    return img_info, annotations

def gen_syn_data(img_files, labels, args):
    '''Creates list of objects and distrctor objects to be pasted on what images.
       Spawns worker processes and generates images according to given params

    Args:
        img_files(list): List of image files
        labels(list): List of labels for each image  
    '''

    w = args.width
    h = args.height   
    background_files = glob.glob(os.path.join(args.background_dir, '*/*.jpg'))    

    print ("Number of background images : %s" % len(background_files))

    img_labels = list(zip(img_files, labels))
    random.shuffle(img_labels)

    print ("Number of images : %s" % len(img_labels))

    if (args.add_distractors or args.add_backgroud_distractors):
        distractor_list = get_list_of_images(args.distractor_dir) 
        distractor_labels = get_labels(distractor_list)
        distractor_files = list(zip(distractor_list, distractor_labels))

        print ("Number of distractor images : %s" % len(distractor_files))        
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
        n = min(random.randint(args.min_objects, args.max_objects), len(img_labels))
        for i in range(n):
            objects.append(img_labels.pop())
        # Get list of distractor objects 
        distractor_objects = []
        if (args.add_distractors or args.add_backgroud_distractors):
            n = min(random.randint(args.min_distractor_objects, args.max_distractor_objects), len(distractor_files))
            for i in range(n):
                distractor_objects.append(random.choice(distractor_files))
            print ("Chosen distractor objects: %s" % distractor_objects)

        idx += 1
        bg_file = random.choice(background_files)
        for blur in args.blending_list:
            img_file = 'images/%i_%s-%s.jpg'%(idx,blur, os.path.splitext(os.path.basename(bg_file))[0])
            params = (objects, distractor_objects, img_file, bg_file, idx)
            params_list.append(params)
            img_files.append(img_file)

    partial_func = partial(create_image_anno_wrapper, args=args) 

    if (args.workers>1):
        p = Pool(args.workers, init_worker)
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
            img_info, annotations = create_image_anno_wrapper(object, args=args)
            results.append([img_info, annotations])

    return results

def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)
 
def generate_synthetic_dataset(args):
    ''' Generate synthetic dataset according to given args
    '''
    img_list = get_list_of_images(args.root) 

    if (args.total_num > 0):
        N = int(np.ceil(args.total_num/len(img_list)))
    else:
        N = 1

    print("Objects will be used %d times" % N)

    img_files = []
    for i in range(N):
        img_files = img_files + random.sample(img_list, len(img_list))

    labels = get_labels(img_files)

    if not os.path.exists(args.exp):
        os.makedirs(args.exp)   
    
    img_dir = os.path.join(args.exp, 'images')
    if not os.path.exists(os.path.join(img_dir)):
        os.makedirs(img_dir)
    
    results = gen_syn_data(img_files, labels, args)  

    unique_labels = sorted(set(labels))

    categories = []
    label_to_id = {}

    for i, unique_label in enumerate(unique_labels):
        categories.append({"supercategory": "none", "id": i, "name": unique_label})
        label_to_id[unique_label] = i

    img_infos = []
    annotations = []
    annotation_id = 0
    for img_info, img_annotations in results:
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

    parser.add_argument("--scale",
      help="Add scale augmentation.Default is to add scale augmentation.", action="store_false")
    parser.add_argument("--rotation",
      help="Add rotation augmentation.Default is to add rotation augmentation.", action="store_false")
    parser.add_argument("--total_num",
      help="Number of times each image will be in dataset", default=0, type=int)
    parser.add_argument("--dontocclude",
      help="Add objects without occlusion. Default is to produce occlusions", action="store_true")
    parser.add_argument("--add_distractors",
      help="Add distractors objects. Default is to not use distractors", action="store_true")
    parser.add_argument("--add_backgroud_distractors",
      help="Add distractors in the background", action="store_true")

    parser.add_argument("--create_masks",
      help="Create object masks", action="store_true")

    # Paths
    parser.add_argument('--background_dir', default='E:/Source/EffortlessCV/data/backgrounds/', type=str)
    parser.add_argument('--distractor_dir', default='E:/Source/EffortlessCV/data/distractors/', type=str)

    # Parameters for generator
    parser.add_argument('--workers', default=10, type=int)
    parser.add_argument("--blending_list", nargs="+", default=['gaussian']) # can be ['gaussian','poisson', 'none', 'box', 'motion']

    # Parameters for images
    parser.add_argument('--min_objects', default=5, type=int)
    parser.add_argument('--max_objects', default=5, type=int)
    parser.add_argument('--min_distractor_objects', default=0, type=int)
    parser.add_argument('--max_distractor_objects', default=2, type=int)
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--height', default=480, type=int)

    # Parameters for objects in images
    parser.add_argument('--min_scale', default=.8, type=float) # min scale for scale augmentation
    parser.add_argument('--max_scale', default=1.5, type=float) # max scale for scale augmentation
    parser.add_argument('--min_distractor_scale', default=.1, type=float) # min scale for scale augmentation
    parser.add_argument('--max_distractor_scale', default=.5, type=float) # max scale for scale augmentation
    parser.add_argument('--max_degrees', default=5, type=float) # max rotation allowed during rotation augmentation
    parser.add_argument('--max_truncation_fraction', default=0, type=float) # max fraction to be truncated = max_truncation_fraction*(width/height)
    parser.add_argument('--max_allowed_iou', default=.5, type=float) # IOU > max_allowed_iou is considered an occlusion
    parser.add_argument('--min_width', default=100, type=int) # Minimum width of object to use for data generation
    parser.add_argument('--min_height', default=100, type=int) # Minimum height of object to use for data generation

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
        
    if (args.create_masks):
        from segment import generate_masks
        generate_masks(args.root)

    generate_synthetic_dataset(args)

    ##createCocoJSONFromXML(train_path_exp)
    ##makeDirsFromXML(train_path_exp)