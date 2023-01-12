import os
import json
import shutil 
from tqdm import tqdm
from csv import reader
import cv2
import numpy as np
import dirtyjson as djson

from matplotlib import pyplot as plt
import scipy.io

def copydir_with_progress(src, dst, masks_dir=None, imagenet_folder_to_idx=None):
    os.makedirs(dst, exist_ok=True)

    if (masks_dir):
        masks = None
        if (os.path.isdir(masks_dir)):
            masks = os.listdir(masks_dir)
            masks = [mask.split(".")[0] for mask in masks]

        file_prefix_folder = os.path.split(src)[1]
        mask_mat = None
        mask_mat_file = os.path.join("E:\\Research\\Images\\ILSVRC2012\\imagenet_eccv12_500K_segmentations_v2\\", file_prefix_folder + ".mat")

        if (os.path.isfile(mask_mat_file)):
            print(mask_mat_file)
            mask_mat = scipy.io.loadmat(mask_mat_file)
            mask_mat_labels = mask_mat["seg"][0][0][0]
            mask_mat_masks = mask_mat["seg"][0][0][1]

            mask_mat_dict = {}

            for i, label in enumerate(mask_mat_labels):
               label = label[0][0]
               mask = mask_mat_masks[i][0]

               mask_mat_dict[label] = mask

               #plt.imshow(mask)
               #plt.show()

    files = os.listdir(dst)

    for file in files:

        if (masks_dir):
            file_prefix  = file.split(".")[0]            

            if (masks and file_prefix in masks):                   
                    mask_file = os.path.join(masks_dir, file_prefix + ".png")
                    segmentation = cv2.imread(mask_file)

                    segmentation_id = segmentation[:, :, 1] * 256 + segmentation[:, :, 2] # R+G*256                    

                    imagenet_folder = os.path.split(src)[1]

                    if (file_prefix_folder != imagenet_folder):
                        continue

                    idx = imagenet_folder_to_idx[imagenet_folder]

                    mask = 1 - np.equal(segmentation_id, idx).astype(int)

                    #if (objectnet_folder == "chair"):
                    #    plt.imshow(mask)
                    #    plt.show()

                    cv2.imwrite(os.path.join(dest, file_prefix + "_mask.jpg"), mask*255)
                    #shutil.copy(os.path.join(src, file), dest) 
            elif(mask_mat):
                print(dst)
                if (file_prefix in mask_mat_dict):                
                    mask = 1 - mask_mat_dict[file_prefix]
                    cv2.imwrite(os.path.join(dest, file_prefix + "_mask.jpg"), mask*255)
                    
        else:
            shutil.copy(os.path.join(src, file), dest) 

def convert_to_jpg(img_file):
    img = cv2.imread(img_file[0])
    #img = img[3:-3,3:-3,:]
    cv2.imwrite(img_file[1], img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

from multiprocessing import Pool

def convertdir_with_progress(src, dst, workers=16):
    os.makedirs(dst, exist_ok=True)

    files = os.listdir(src)

    params = []

    for file in files:
        file_prefix  = file.split(".")[0]
        img_file = os.path.join(src, file_prefix + ".JPEG")
        img_file_jpg = os.path.join(dst, file_prefix + ".jpg")
        
        if (workers > 1):
            params.append([img_file, img_file_jpg])
        else:
            convert_to_jpg(img_file, img_file_jpg) 

    with Pool(workers) as p:
        p.map(convert_to_jpg, params)

if __name__ == '__main__':

    # Opening JSON file 
    with open('E:/Research/Images/objectnet-1.0/mappings/folder_to_objectnet_label.json') as json_file: 
        folder_to_objectnet_label = json.load(json_file) 

    with open('E:/Research/Images/objectnet-1.0/mappings/objectnet_to_imagenet_1k.json') as json_file: 
        objectnet_to_imagenet_1k = json.load(json_file) 

    imagenet_folder_to_idx = {}
    with open('E:/Research/Images/objectnet-1.0/mappings/ImageNetS_categories_im919.txt', 'r') as read_obj:
        csv_reader = reader(read_obj, delimiter =":")
        for idx, label in enumerate(csv_reader):
            imagenet_folder_to_idx[label[0]] = idx+1
    
    # open file in read mode
    imagenet_label_to_folder = {}
    with open('E:/Research/Images/objectnet-1.0/mappings/imagenet_to_label_2012_v2.txt', 'r') as read_obj:
        csv_reader = reader(read_obj, delimiter =":")
        for row in csv_reader:
            folder, label = row
            label = label.strip()

            if (label in imagenet_label_to_folder):
                label += "2"

            imagenet_label_to_folder[label] = folder

    imagenet_folder_to_label = {}
    for key in imagenet_label_to_folder.keys():
        imagenet_folder_to_label[imagenet_label_to_folder[key]] = key

    imagenet_113_path = 'E:/Research/Images/objectnet-1.0/imagenet_107'
    imagenet_113_path_mask = 'E:/Research/Images/objectnet-1.0/masks_107'

    objectnet_path = 'E:/Research/Images/objectnet-1.0/images_107'
    objectnet_folders = os.listdir(objectnet_path)
    objectnet_folders = [objectnet_folder for objectnet_folder in objectnet_folders if os.path.isdir(os.path.join(objectnet_path, objectnet_folder))]

    imagenet_path = 'E:/Research/Images/ILSVRC2012/train'
    imagenet_path_masks = 'E:/Research/Images/ILSVRC2012/Imagenet-S//ImageNetS919/train-semi-segmentation'
    imagenet_folders = os.listdir(imagenet_path)
    imagenet_folders = [imagenet_folder for imagenet_folder in imagenet_folders if os.path.isdir(os.path.join(imagenet_path, imagenet_folder))]

    for objectnet_folder in tqdm(objectnet_folders):

        objectnet_label = folder_to_objectnet_label[objectnet_folder]
        imagenet_labels = objectnet_to_imagenet_1k[objectnet_label].split(";")

        for imagenet_label in imagenet_labels:
            imagenet_folder = imagenet_label_to_folder[imagenet_label.strip()]
            #print("\n%s:\t%s\t%s" % (objectnet_folder, imagenet_label.strip(), imagenet_folder))
            print(imagenet_folder)

            if (imagenet_folder in imagenet_folders):
                imagenet_folders.append(imagenet_folder)
                src = os.path.join(imagenet_path, imagenet_folder)
                dest = os.path.join(imagenet_113_path, objectnet_folder)
                masks_dir = os.path.join(imagenet_path_masks, imagenet_folder)
                copydir_with_progress(src, dest, masks_dir, imagenet_folder_to_idx) 

    #masks_path = 'E:/Research/Images/objectnet-1.0/masks_107'
    #folders = os.listdir(masks_path)

    #for folder in folders:
    #    src = os.path.join(masks_path.replace("masks_107", "images_113"), folder)
    #    dest = os.path.join(masks_path.replace("masks_107", "images_107"), folder)
    #    print(src)
    #    copydir_with_progress(src, dest) 

    #masks_path = 'E:/Research/Images/objectnet-1.0/imagenet_107'
    #folders = os.listdir(masks_path)

    #for folder in tqdm(folders):
    #    src = os.path.join(masks_path, folder)
    #    dest = os.path.join(masks_path.replace("imagenet_107", "jpeg_107"), folder)
    #    print(src)
    #    convertdir_with_progress(src, dest)

