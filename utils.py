import json
import gzip
import random
import pickle
from glob import glob
from tqdm import tqdm
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

from PIL import Image, UnidentifiedImageError
from datasets import Dataset

MAX_PROCESS = 32
asin2imgs = dict()


def load_ngrams(files):
    concepts = set()
    for filename in files:
        with open(filename, "rb") as f:
            for line in f.readlines():
                line = line.decode().strip().split(",")
                if len(line) != 2:
                    continue
                concept = line[0].replace("('", "").replace("'", "")
                count = int(line[1].replace(" ", "").replace(")", ""))
                if count > 1000:
                    concepts.add(concept)
    return concepts


def load_concepts(filename):
    concepts = set()
    with open(filename, "rb") as f:
        for line in f.readlines():
            line = line.decode().strip().split(",")
            line = [item.strip().lower() for item in line if item != ""]
            concepts.update(line)
    return concepts


def load_categories(filename):
    with open(filename, "rb") as f:
        return [l.decode().strip() for l in f.readlines()]


def load_products(filename):
    products = []
    with gzip.open(filename) as f:
        for line in f:
            products.append(json.loads(line.strip()))
    return products


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)



def open_image(filename):
    try:
        img = Image.open(filename)
        width, height = img.size
        if width < 224 or height < 224:
            return None
    except UnidentifiedImageError:
        return None

    img = img.resize((224, 224))
    return img


def get_imgs_for_product(p):
    asin = p["asin"]
    if asin in asin2imgs:
        # print(f"got {len(asin2imgs[asin])} images from cache")
        return asin2imgs[asin]
        
    imgs = []
    asin_pref = asin[:2]
    img_files = list(glob(f"./fashion/imgs/{asin_pref}/{asin}/*.jpg"))
    with ProcessPoolExecutor(max_workers=MAX_PROCESS) as exec:
        for img in exec.map(open_image, img_files):
            if img:
                imgs.append(img)

    asin2imgs[asin] = imgs
    return imgs


def load_concept_activation_dataset(positive_products, products, concept):
    dataset = dict(img=[], label=[])
    negative_products = []

    random.shuffle(products)
    for p in products:
        if concept not in p["concepts"]:
            if len(negative_products) == len(positive_products):
                break
            negative_products.append(p)

    products_for_imgs = positive_products + negative_products
    product_labels = [1] * len(positive_products) + [0] * len(negative_products)

    for p, l in zip(products_for_imgs, product_labels):
        p_imgs = get_imgs_for_product(p)
        dataset["img"] += p_imgs
        dataset["label"] += [l] * len(p_imgs)
    
    dataset = Dataset.from_dict(dataset)
    return dataset

