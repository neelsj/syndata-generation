
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import numpy as np
from skimage import measure 
from shapely.geometry import Polygon, MultiPolygon

from PIL import Image
import cv2

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision import transforms

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

model = None

def create_mask(input_image):

    global model

    if (model is None):
        #model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
        model.eval()

    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # plot the semantic segmentation predictions of 21 classes in each color

    mask = np.uint8(255*(output_predictions.cpu().numpy() > 0))
    #mask = output_predictions.byte().cpu().numpy()

    return mask

def create_mask_coco(input_image, category, keepBiggest=True):

    global model
    global preprocess
    global weights

    if (model is None):
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        preprocess = weights.transforms()
    
        model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=False)
        model = model.eval()

    input_image = input_image.convert("RGB")

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)[0]

    mask = None
    
    labels = [weights.meta["categories"][label] for label in output['labels']]

    if (labels):

        #print(labels)

        mask = np.zeros((input_image.size[1],input_image.size[0]))

        masks = output["masks"].cpu().numpy()
        
        maskCatsMax = 0

        for i in range(masks.shape[0]):

            if (labels[i] == category):
                maskCat = (masks[i,:,:,:].squeeze() > .5).astype(float)

                if (keepBiggest):
                    maskCatsMean = np.mean(maskCat)

                    if (maskCatsMean > maskCatsMax):
                        mask = maskCat            
                        maskCatsMax = maskCatsMean        
                else:
                    mask = np.maximum(mask, maskCat)

    #plt.imshow(mask)
    #plt.waitforbuttonpress()

        mask = mask if np.mean(mask) > .1 else None

    return mask

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd, bbox=None):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    #contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    padded_binary_mask = np.pad(sub_mask, pad_width=1, mode='constant', constant_values=0)    
    contours = measure.find_contours(padded_binary_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = bbox if (bbox) else (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

def generate_masks(data_dir, background=False, coco=False):
    
    dirs = os.listdir(data_dir)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    
    prcThresh = 3

    images = []
    annotations = []

    image_id = 1
    category_id = 1
    annotation_id = 1

    categories = []

    for dir in tqdm(dirs):
        files_dir = os.path.join(data_dir, dir)

        if (not os.path.isdir(files_dir)):
            continue

        files = os.listdir(files_dir)
        files = [file for file in files if "_mask" not in file]

        category = {"supercategory": "object", "id": category_id, "name": dir}
        categories.append(category)

        for file in tqdm(files):

            filename = os.path.join(data_dir, dir, file)
            #print(filename)

            image = Image.open(filename)

            new_img={}
            new_img["license"] = 0
            new_img["file_name"] = os.path.join(dir, file)
            new_img["width"] = int(image.size[0])
            new_img["height"] = int(image.size[1])
            new_img["id"] = image_id
            images.append(new_img)

            if (coco):
                mask = create_mask_coco(image, dir)
            else:
                mask = create_mask(image)

            if (mask is not None):
                if (background):

                    maskname = os.path.splitext(filename)[0] + "_mask.jpg"
                    maskObj = np.uint8(255*(mask==0))
                    Image.fromarray(maskObj).save(maskname)

                #plt.imshow(np.array(image)[:,:,0]*mask)
                #plt.show()

                else:
                    nb_components, output, boxes, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                    box_sizes = [box[4] for box in boxes[1:]]        

                    for id in range(1, nb_components):
                
                        box = [int(b) for b in boxes[id][0:4]]

                        sub_mask = np.reshape(output==id, mask.shape).astype(np.double)

                        #plt.imshow(sub_mask)
                        #plt.show()

                        prc = 100*box_sizes[id-1]/(mask.shape[0]*mask.shape[1])
                
                        if (prc >= prcThresh):
                            try:
                                annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, False, bbox=box)
                                annotations.append(annotation)
                                annotation_id += 1
                            except Exception as e:
                                print(e)
                                pass


                    #print(nb_components)
                    #print(output)
                    #print(stats)
                    #print(centroids)

                    # save mask for dominant big object
                    if (box_sizes):
                        max_ind = np.argmax(box_sizes)
                        #print(max_ind)

                        prc = 100*box_sizes[max_ind]/(mask.shape[0]*mask.shape[1])
                        #print(prc)                

                        if (prc >= prcThresh):
                            maskname = os.path.splitext(filename)[0] + "_mask.jpg"
                            #print(maskname)

                            maskObj = np.uint8(255*np.reshape(1-(output==max_ind+1),  mask.shape))

                            #maskObjN = 255-maskObj
                            #edgeSum = np.sum(maskObjN[:,0]) + np.sum(maskObjN[:,-1]) + np.sum(maskObjN[0,:]) + np.sum(maskObjN[-1,:])
                    
                            #if (edgeSum == 0):
                            Image.fromarray(maskObj).save(maskname)

                            ##mask.putpalette(colors)
                            #plt.subplot(121)
                            #plt.imshow(image)                    
                            #plt.subplot(122)
                            #plt.imshow(maskObj)
                            #plt.show()

                image_id += 1

                #if (image_id > 3):
                #    break

            category_id += 1

            #if (category_id > 3):
            #    break

    print("saving annotations to coco as json ")
    ### create COCO JSON annotations
    coco = {}
    coco["info"] = COCO_INFO
    coco["licenses"] = COCO_LICENSES
    coco["images"] = images
    coco["categories"] = categories
    coco["annotations"] = annotations

    # TODO: specify coco file locaiton 
    output_file_path = os.path.join(data_dir,"../", "coco_instances.json")
    with open(output_file_path, 'w+') as json_file:
        json_file.write(json.dumps(coco))

    print(">> complete. find coco json here: ", output_file_path)
    print("last annotation id: ", annotation_id)
    print("last image_id: ", image_id)

    #from pycocotools.coco import COCO
    ## Initialize the COCO api for instance annotations
    #coco = COCO(output_file_path)
   
    ## Load the categories in a variable
    #imgIds = coco.getImgIds()
    #print("Number of images:", len(imgIds))

    ## load and display a random image
    #for i in range(len(imgIds)):
    #    img = coco.loadImgs(imgIds[i])[0]

    #    I = Image.open(data_dir + "/" + img['file_name'])

    #    plt.clf()
    #    plt.imshow(I)
    #    plt.axis('off')

    #    annIds = coco.getAnnIds(imgIds=img['id'])
    #    anns = coco.loadAnns(annIds)

    #    coco.showAnns(anns, True)
    #    plt.waitforbuttonpress()        

if __name__ == "__main__":

    data_dir = "E:/Research/Images/FineGrained/StanfordCars/train_bing/"