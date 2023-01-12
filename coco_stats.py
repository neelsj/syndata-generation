import json
from pycocotools.coco import COCO
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument('annotations', type=str)

parser.add_argument('--annotations2', type=str)

args = parser.parse_args()

def compute_stats(annotations):

    coco = COCO(annotations)
    ids = coco.getImgIds()
    images = coco.loadImgs(ids)

    ns = []
    xs = []
    ys = []
    ws = []
    hs = []
    ss = []

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
            ys.append(cy/h)
            ws.append(ann["bbox"][2]/w)
            hs.append(ann["bbox"][3]/h)
            ss.append(max(ann["bbox"][2], ann["bbox"][3])/max(w,h))

    ns_orig = ns
    ws_orig = ws
    hs_orig = hs
    ss_orig = ss

    ns = [max(n, 1e-6) for n in ns]
    ws = [max(1-w, 1e-6) for w in ws]
    hs = [max(1-h, 1e-6) for h in hs]
    ss = [max(1-s, 1e-6) for s in ss]

    ns = np.log(ns)
    ws = np.log(ws)
    hs = np.log(hs)
    ss = np.log(ss)

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

    s_mean = np.mean(ss)
    s_std = np.std(ss)
    s_min = np.min(ss)
    s_max = np.max(ss)

    print("n_mean:\t%f\tn_std:\t%f\tn_min:\t%f\tn_max:\t%f" % (n_mean, n_std, n_min, n_max))
    print("x_mean:\t%f\tx_std:\t%f\tx_min:\t%f\tx_max:\t%f" % (x_mean, x_std, x_min, x_max))
    print("y_mean:\t%f\ty_std:\t%f\ty_min:\t%f\ty_max:\t%f" % (y_mean, y_std, y_min, y_max))
    print("w_mean:\t%f\tw_std:\t%f\tw_min:\t%f\tw_max:\t%f" % (w_mean, w_std, w_min, w_max))
    print("h_mean:\t%f\th_std:\t%f\th_min:\t%f\th_max:\t%f" % (h_mean, h_std, h_min, h_max))
    print("s_mean:\t%f\ts_std:\t%f\ts_min:\t%f\ts_max:\t%f" % (s_mean, s_std, s_min, s_max))

    stats = {}
    stats["n"] = {"mean": n_mean, "std": n_std, "min": n_min, "max": n_max, "type": "lognormal"}
    stats["x"] = {"mean": x_mean, "std": x_std, "min": x_min, "max": x_max, "type": "normal"}
    stats["y"] = {"mean": y_mean, "std": y_std, "min": y_min, "max": y_max, "type": "normal"}
    stats["w"] = {"mean": w_mean, "std": w_std, "min": w_min, "max": w_max, "type": "one_minus_lognormal"}
    stats["h"] = {"mean": h_mean, "std": h_std, "min": h_min, "max": h_max, "type": "one_minus_lognormal"}
    stats["s"] = {"mean": s_mean, "std": s_std, "min": s_min, "max": s_max, "type": "one_minus_lognormal"}

    return stats, ns_orig, xs, ys, ws_orig, hs_orig, ss_orig

def plot_dist(stat, xs, label, plot_fit=False, norm=True):

    plt.figure()
    
    n, bins, patches = plt.hist(xs, 50, norm=True)    
    plt.title(label)

    if (plot_fit):
        if (stat["type"] == "lognormal"):
           n, bins, patches = plt.hist(np.random.lognormal(stat["mean"], stat["std"], len(xs)), bins, histtype='step')
        elif (stat["type"] == "one_minus_lognormal"):
            n, bins, patches = plt.hist(1-np.random.lognormal(stat["mean"], stat["std"], len(xs)), bins, histtype='step')
        else:
            n, bins, patches = plt.hist(np.random.normal(stat["mean"], stat["std"], len(xs)), bins, histtype='step')

def plot_dist_compare(stat, xs, stat2, xs2, label, plot_fit=False, norm=True):

    plt.figure()
    
    n, bins, patches = plt.hist(xs, 50, density=norm)    
    n, bins, patches = plt.hist(xs2, bins, density=norm, histtype='step')
    plt.title(label)

def plot_all(stats, ns, xs, ys, ws, hs, ss, plot_fit=False):

    #plot_dist(stats["n"], ns, "ns", plot_fit)
    plot_dist(stats["x"], xs, "xs", plot_fit)
    plot_dist(stats["y"], ys, "ys", plot_fit)
    #plot_dist(stats["w"], ws, "ws", plot_fit)
    #plot_dist(stats["h"], hs, "hs", plot_fit)
    plot_dist(stats["s"], ss, "ss", plot_fit)


def plot_all_compare(stats, ns, xs, ys, ws, hs, ss, stats2, ns2, xs2, ys2, ws2, hs2, ss2):

    plot_dist_compare(stats["x"], xs, stats2["x"], xs2, "xs")
    plot_dist_compare(stats["y"], ys, stats2["y"], ys2, "ys")
    plot_dist_compare(stats["s"], ss, stats2["s"], ss2, "ss")

def main(args):
    plot = True

    stats, ns, xs, ys, ws, hs, ss = compute_stats(args.annotations)
    stats2, ns2, xs2, ys2, ws2, hs2, ss2 = compute_stats(args.annotations2)

    if (plot):
        #plot_all(stats, ns, xs, ys, ws, hs, ss)
        plot_all_compare(stats, ns, xs, ys, ws, hs, ss, stats2, ns2, xs2, ys2, ws2, hs2, ss2)
        plt.show()

    #output_file_path = os.path.join(os.path.split(args.annotations)[0], "stats.json")
    #with open(output_file_path, 'w+') as json_file:
    #    json_file.write(json.dumps(stats))

if __name__ == "__main__":
    main(args)
