import json
from pycocotools.coco import COCO
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')

args = parser.parse_args()

def main(args):

    coco = COCO(args.annotations)
    ids = coco.getImgIds()
    images = coco.loadImgs(ids)

    ns = []
    xs = []
    ys = []
    ws = []
    hs = []

    for img in images:
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        ns.append(len(anns))

        w = img['width']
        h = img['height']

        for ann in anns:
            cx = ann["bbox"][0] + ann["bbox"][2]/2 - w/2
            cy = ann["bbox"][1] + ann["bbox"][3]/2 - h/2
            xs.append(cx/w)
            ys.append(cx/h)
            ws.append(ann["bbox"][2]/w)
            hs.append(ann["bbox"][3]/h)

    n_mean = np.mean(ns)
    n_std = np.std(ns)
    n_min = np.min(ns)
    n_max = np.max(ns)

    x_mean = np.mean(xs)
    x_std = np.std(xs)
    x_min = np.min(xs)
    x_max = np.max(xs)

    y_mean = np.mean(ys)
    y_std = np.std(ys)
    y_min = np.min(ys)
    y_max = np.max(ys)

    w_mean = np.mean(ws)
    w_std = np.std(ws)
    w_min = np.min(ws)
    w_max = np.max(ws)

    h_mean = np.mean(hs)
    h_std = np.std(hs)
    h_min = np.min(hs)
    h_max = np.max(hs)

    print("n_mean:\t%f\tn_std:\t%f\tn_min:\t%f\tn_max:\t%f" % (n_mean, n_std, n_min, n_max))
    print("x_mean:\t%f\tx_std:\t%f\tx_min:\t%f\tx_max:\t%f" % (x_mean, x_std, x_min, x_max))
    print("y_mean:\t%f\ty_std:\t%f\ty_min:\t%f\ty_max:\t%f" % (y_mean, y_std, y_min, y_max))
    print("w_mean:\t%f\tw_std:\t%f\tw_min:\t%f\tw_max:\t%f" % (w_mean, w_std, w_min, w_max))
    print("h_mean:\t%f\th_std:\t%f\th_min:\t%f\th_max:\t%f" % (h_mean, h_std, h_min, h_max))

    ## the histogram of the data
    #plt.figure()
    #n, bins, patches = plt.hist(xs, 50)
    #plt.title('xs')
    #plt.xlim(x_min, x_max)    

    #plt.figure()
    #n, bins, patches = plt.hist(ys, 50)
    #plt.title('ys')
    #plt.xlim(y_min, y_max)  

    #plt.show()

if __name__ == "__main__":
    main(args)
