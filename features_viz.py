#from cuml import 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import os

import csv

import seaborn as sns
import pandas as pd  
import matplotlib.pyplot as plt

from tqdm import tqdm

def get_article(a):
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

def create_prompts(a, b, relation, background):

    prompt0a = get_article(a)
    prompt0b = get_article(b)
    prompt1a = get_article(a) + " and " + get_article(b)
    prompt1b = get_article(b) + " and " + get_article(a)
    prompt2a = get_article(a) + get_relation(relation) + get_article(b)
    prompt2b = get_article(b) + get_relation(mirror_relation(relation)) + get_article(a)
    prompt3a = get_article(a) + " and " + get_article(b) + " in a " + background
    prompt3b = get_article(b) + " and " + get_article(a) + " in a " + background
    prompt4a = get_article(a) + get_relation(relation) + get_article(b) + " in a " + background
    prompt4b = get_article(b) + get_relation(mirror_relation(relation)) + get_article(a) + " in a " + background

    #prompts = [prompt0a, prompt0b, prompt1a, prompt1b, prompt2a, prompt2b, prompt3a, prompt3b, prompt4a, prompt4b]
    #levels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    prompts = [prompt1a, prompt2a, prompt3a, prompt4a]
    levels = [1, 2, 3, 4]

    return prompts, levels

def dict_list_add(dict, key, i):
    if key in dict:
        if (np.isscalar(i)):
            dict[key].append(i)            
        else:  
            dict[key] += i.tolist()
    else:
        if (np.isscalar(i)):
            dict[key] = [i]
        else:
            dict[key] = i.tolist()

path = 'E:/Source/EffortlessCVSystem/Data/coco_spatial_backgrounds/'

image_prompt_data = {}
prompts_by_level_data = {}

with open(os.path.join(path, 'pairs.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile)

    for i, row in enumerate(reader):
        a = row[0]
        b = row[1]
        relation = row[2]
        background = row[3].replace("-", " ").replace("_", " ")

        prompts, levels = create_prompts(a, b, relation, background)
    
        for j, prompt in enumerate(prompts):
            dict_list_add(image_prompt_data, (prompt, levels[j]), i)

            dict_list_add(prompts_by_level_data, levels[j], prompt)

image_features = np.load(os.path.join(path, 'image_features_background.npy'))
text_features = np.load(os.path.join(path, 'text_features_background.npy'))
text_features_dict = np.load(os.path.join(path, 'text_features_background_dict.npy'), allow_pickle=True).item()

image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)

#avg_per_level_similarity_table = {}

#for level in prompts_by_level_data.keys():

#    prompts = sorted(list(set(prompts_by_level_data[level])))

#    avg_similarity = {}
#    avg_similarity_mean_table = np.ndarray((len(prompts),len(prompts)))
#    avg_similarity_std_table = np.ndarray((len(prompts),len(prompts)))

#    for i, main_prompt in enumerate(tqdm(prompts)):
#        prompt_feature = text_features[text_features_dict[prompt],]

#        images = image_prompt_data[(main_prompt, level)]
#        image_feature_sel = image_features[images,:]

#        for j, prompt in enumerate(prompts):
#            prompt_feature = text_features[text_features_dict[prompt],]

#            sims = np.dot(image_feature_sel, prompt_feature)
#            sims_avg = np.mean(sims)
#            sims_std = np.std(sims)

#            avg_similarity[(main_prompt, prompt)] = (sims_avg, sims_std)
#            avg_similarity_mean_table[i,j] = sims_avg
#            avg_similarity_std_table[i,j] = sims_std

#    avg_per_level_similarity_table[level] = (avg_similarity, avg_similarity_mean_table, avg_similarity_std_table)

#    inds_sel = []
#    prompt_sel = []
#    for i, p in enumerate(prompts):
#        if (("bench" in p or "kite" in p) and not ("above" in p or "below" in p or "left" in p)):
#            if (level == 3 or level == 4):
#                if ("playground" in p or "downtown" in p):
#                    inds_sel.append(i)
#                    prompt_sel.append(p)
#            else:
#                inds_sel.append(i)
#                prompt_sel.append(p)

#    print(prompt_sel)

#    d = avg_similarity_mean_table[inds_sel,:]
#    d = d[:,inds_sel]

#    df_cm = pd.DataFrame(d, prompt_sel, prompt_sel)
#    sns.set(font_scale=0.8) # for label size
#    sns.heatmap(df_cm, annot=True, cbar=False) # font size
#    plt.show()

avg_simliarity = []
avg_per_level_similarity_table = {}

for prompt_level in image_prompt_data.keys():

    prompt, level = prompt_level

    prompt_feature = text_features[text_features_dict[prompt],]

    images = image_prompt_data[prompt_level]
    image_feature_sel = image_features[images,:]

    sims = np.dot(image_feature_sel, prompt_feature)    

    sims_avg = np.mean(sims)
    sims_std = np.std(sims)

    dict_list_add(avg_per_level_similarity_table, level, sims_avg)

avg_per_level_similarity = []

for level in avg_per_level_similarity_table.keys():

    sims_avg = np.mean(avg_per_level_similarity_table[level])
    sims_std = np.std(avg_per_level_similarity_table[level])

    avg_per_level_similarity.append([sims_avg,sims_std])

    print("level %d:\t%f\t%f" % (level, sims_avg, sims_std))

#image_data = {}

#with open(os.path.join(path, 'pairs.csv'), newline='') as csvfile:
#    reader = csv.reader(csvfile)

#    for i, row in enumerate(reader):
#        a = row[0]
#        b = row[1]
#        relation = row[2]
#        background = row[3].replace("-", " ").replace("_", " ")

#        prompts = [a, b, background]
    
#        for prompt in prompts:
#            dict_list_add(image_data, prompt, i)



#df = pd.DataFrame()

#x = []
#y = []
#label = []

##for i, relation in enumerate(relations):

##    key = ("dog", "bench")

##    x += image_features[image_data[key],0].tolist()
##    y  += image_features[image_data[key],1].tolist()

##    label += [key]*len(image_features[image_data[key]])


#key = ("dog", "sink")
#keys = prompt_data_keys[key]

#for key in keys:
#    for k in key:
#        x += text_features[prompt_data[k],0].tolist()
#        y += text_features[prompt_data[k],1].tolist()

#        key_disp = k

#        label += [key_disp]*len(text_features[prompt_data[k]])

#df["x"] = x
#df["y"] = y
#df["label"] = label

#sns.scatterplot(x="x", y="y", hue="label", data=df, palette=sns.color_palette("flare"))

##for i in range(df.shape[0]):
##    plt.text(x=df.x[i],y=df.y[i]+.1*np.random.rand()-.05,s=label[i], 
##              fontdict=dict(color='red',size=6),
##              bbox=dict(facecolor='yellow',alpha=0.5),)

#plt.show()