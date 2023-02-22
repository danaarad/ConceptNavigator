import re
import json
import torch
import pickle

import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize

from glob import glob
from tqdm import tqdm
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

from config import *


nltk.download('punkt')   

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f) 

def get_product_data(product):
    product_fields = []

    if "title" in product:
        product_fields.append(product["title"])

    if "description" in product:
        if isinstance(product["description"], str):
            product_fields.append(product["description"])
        elif isinstance(product["description"], list):
            product_fields.extend(product["description"])

    product_fields.extend(product.get("feature", []))
    # product_fields.extend(product.get("category", []))

    if any([isinstance(d, list) for d in product_fields]):
        print(product_fields)
        print(product)

    product_fields = [d.lower() for d in product_fields]
    product_fields = [d.replace(".", " . ") for d in product_fields]
    product_fields = [d.replace(",", " , ") for d in product_fields]
    product_fields = [d.replace("=", " = ") for d in product_fields]
    product_fields = [d.replace(":", " : ") for d in product_fields]
    product_fields = [d.replace("/", " : ") for d in product_fields]
    product_fields = [d.replace("\t", " \t ") for d in product_fields]
    product_fields = [d.replace("\n", " \n ") for d in product_fields]

    data = set()
    for field in product_fields:
        words = word_tokenize(field)
        two_grams = [" ".join(tup) for tup in ngrams(words, 2)]
        data.update(words)
        data.update(two_grams)
    return data



TRAIN = load_json("/home/daarad/concepts/fashion/concepts_data/train_with_ngram_concepts.json")
for product in tqdm(TRAIN):
    product["data"] = get_product_data(product)

DEV = load_json("/home/daarad/concepts/fashion/concepts_data/dev_with_ngram_concepts.json")
for product in tqdm(DEV):
    product["data"] = get_product_data(product)


def pluralize(noun):
    # from https://www.codespeedy.com/program-that-pluralize-a-given-word-in-python/
    if re.search('[sxz]$', noun):
        return re.sub('$', 'es', noun)
    elif re.search('[^aeioudgkprt]h$', noun):
        return re.sub('$', 'es', noun)
    elif re.search('[aeiou]y$', noun):
        return re.sub('y$', 'ies', noun)
    else:
        return noun + 's'


def get_image_paths(products, images_per_product=1):
    image_paths = []
    for product in products:
        asin = product["asin"]
        asin_prefix = asin[:2]
        prodcut_image_paths = list(glob(f"/home/daarad/concepts/fashion/imgs/{asin_prefix}/{asin}/*.jpg"))[:images_per_product]
        prodcut_image_paths = [p.replace(f"/home/daarad/concepts/fashion/imgs/{asin_prefix}/", "") for p in prodcut_image_paths]
        prodcut_image_paths = [p.replace(f".jpg", "") for p in prodcut_image_paths]
        prodcut_image_paths = ["./ui/product_images/" + p for p in prodcut_image_paths]
        image_paths.extend(prodcut_image_paths)
    return image_paths


class DefineRequest:
    def __init__(self, dataset, task, concept):
        self.dataset = dataset
        self.task = task
        self.concept = concept
        self.train_pos, self.train_neg = [], []
        self.dev_pos, self.dev_neg = [], []
        with open(f"/home/daarad/concepts/{dataset}/{task}_prediction/asin2embeddings.pkl", "rb") as f:
            self.asin2embeddings = pickle.load(f)
        self.concept_data = None

    def collect_pos_neg_samples(self, split):
        if split == "train":
            products = TRAIN
            pos, neg = self.train_pos, self.train_neg
        else:
            products = DEV
            pos, neg = self.dev_pos, self.dev_neg

        for product in tqdm(products):
            if "data" not in product:
                product["data"] = get_product_data(product)

            has_embeddings = True
            for model_embeddings in self.asin2embeddings.values():
                if product["asin"] not in model_embeddings:
                    has_embeddings = False
            if not has_embeddings:
                continue

            if self.concept in product["data"] or pluralize(self.concept) in product["data"]:
                pos.append(product)
            else:
                neg.append(product)
            
            if len(pos) >= MAX_POS_SAMPLES and len(neg) >= MAX_NEG_SAMPLES:
                return

    def is_request(self, dataset, task, concept):
        if dataset == self.dataset and task == self.task and concept == self.concept:
            return True
        else:
            return False

    def set_concept_data(self, cd):
        self.concept_data = cd
            