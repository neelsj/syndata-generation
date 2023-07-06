import os
import clip
import torch

from PIL import Image
import numpy as np

import csv

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

    prompts = [prompt0a, prompt0b, prompt1a, prompt1b, prompt2a, prompt2b, prompt3a, prompt3b, prompt4a, prompt4b]
    levels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    return prompts, levels

path = 'E:/Source/EffortlessCVSystem/Data/coco_spatial_backgrounds/'

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

clip.available_models()

model, preprocess = clip.load('ViT-L/14', device)

compute_image_feats = False
compute_prompt_feats = True
compute_tsne = False

batchSize = 256

if (compute_image_feats):

    rows = []
    with open(os.path.join(path, 'pairs.csv'), newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)

    image_features = np.zeros((len(rows), 768))

    for i in tqdm(range(0, len(rows), batchSize)):

        images = []

        num = min(batchSize, len(rows)-i)

        for j in range(num):
            row = rows[i+j]
            img_file = os.path.join(path, row[7])
            image = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
            images.append(image)
        
        images = torch.cat(images, dim=0)

        # Calculate features
        with torch.no_grad():
            image_feature = model.encode_image(images)

        image_features[i:i+num,:] = image_feature.cpu().numpy()

    np.save(os.path.join(path, 'image_features_background.npy'), image_features)

if(compute_prompt_feats):

    prompts = []
    prompts_dict = {}

    with open(os.path.join(path, 'pairs.csv'), newline='') as csvfile:   
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            a = row[0]
            b = row[1]
            relation = row[2]
            background = row[3].replace("-", " ").replace("_", " ")

            p, _ = create_prompts(a, b, relation, background)
            prompts += p

    prompts = sorted(list(set(prompts)))

    text_features = np.zeros((len(prompts), 768))

    print("%d prompts" % len(prompts))

    for i in tqdm(range(0, len(prompts), batchSize)):

        prompts_batch = []

        num = min(batchSize, len(prompts)-i)

        for j in range(num):
            prompt = prompts[i+j]           
            prompts_batch.append(prompt)

            prompts_dict[prompt] = i+j

        texts = clip.tokenize(prompts_batch).to(device)

        # Calculate features
        with torch.no_grad():
            text_feature = model.encode_text(texts)

        text_features[i:i+num,:] = text_feature.cpu().numpy()

    np.save(os.path.join(path, 'text_features_background.npy'), text_features)
    np.save(os.path.join(path, 'text_features_background_dict.npy'), prompts_dict)

if (compute_tsne):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    image_features = np.load(os.path.join(path, 'image_features_background.npy'))
    text_features = np.load(os.path.join(path, 'text_features_background.npy'))

    features = np.concatenate((image_features, text_features))

    print("pca")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features) 

    print("tsne")
    tsne = TSNE(n_components=2)
    features_tsne = tsne.fit_transform(features_pca)

    np.save(os.path.join(path, 'image_features_background_tsne.npy'), features_tsne[0:len(image_features),:])
    np.save(os.path.join(path, 'text_features_background_tsne.npy'), features_tsne[len(image_features)+1:,:])