import json
import gzip
from tqdm import tqdm

import nltk
from nltk import ngrams
from nltk.corpus import stopwords

import itertools
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from utils import *

nltk.download('stopwords')


def collect_features_from_product(args):
    product, fileds_to_extract = args
    features = set()
    for field in fileds_to_extract:
        raw_features = product.get(field)
        if raw_features is None:
            continue

        if isinstance(raw_features, list):
            features.update(raw_features)
        elif isinstance(raw_features, str):
            features.add(raw_features)
    return features


def collect_features(products, fileds_to_extract):
    features = set()
    with ProcessPoolExecutor(max_workers=50) as executor:
        params = zip(products, itertools.repeat(fileds_to_extract))
        for feature_set in tqdm(executor.map(collect_features_from_product, params)):
            features.update(feature_set)
    return features


def preprocess_features(features, words_to_filter):
    stop_words = set(stopwords.words('english'))
    processed_features = []

    for feature in tqdm(features):
        feature = feature.strip().replace("\n", "").split(" ")
        feature = [w.lower() for w in feature if w != '']
        feature = [w for w in feature if w.isalpha()]
        feature = [w for w in feature if not w in stop_words]
        feature = [w for w in feature if not w in words_to_filter]
        feature = [w for w in feature if len(w) > 2]
        feature = [
            w.replace("<", " < ").replace(">", " > ").replace("/", " / ").replace(":", " : ")
            for w in feature
         ]
        feature = [
            w.replace(".", " . ").replace(",", " , ").replace("\n", " \n ").replace("\t", " \t ")
            for w in feature
         ]
        processed_features.append(feature)

    return processed_features


def generate_ngrams_from_features(features):
    one_grams = Counter()
    two_grams = Counter()
    three_grams = Counter()

    for feature in tqdm(features):
        one_grams.update(Counter(feature))
        if len(feature) >= 2:
            grams = [g for g in ngrams(feature, 2) if "" not in g]
            two_grams.update(Counter(grams))
        if len(feature) >= 3:
            grams = [g for g in ngrams(feature, 3) if "" not in g]
            three_grams.update(Counter(grams))

    return one_grams, two_grams, three_grams


def save_to_file(counter, filename):
    with open(filename, "wb") as f:
        lines = [f"{t}\n".encode() for t in counter.most_common(5000)]
        f.writelines(lines)


def generate_ngrams(products, filed_to_extract, words_to_filter, output_prefix):
    print(f"number of products: {len(products)}")
    features = collect_features(products, [filed_to_extract])
    features = preprocess_features(features, words_to_filter)
    print(f"number of unique features: {len(features)}")

    one_grams, two_grams, three_grams = generate_ngrams_from_features(features)

    print(f"saving {len(one_grams)} 1grams")
    save_to_file(one_grams, f"{output_prefix}/{filed_to_extract}/1grams.txt")

    print(f"saving {len(two_grams)} 2grams")
    save_to_file(two_grams, f"{output_prefix}/{filed_to_extract}/2grams.txt")

    print(f"saving {len(three_grams)} 3grams")
    save_to_file(three_grams, f"{output_prefix}/{filed_to_extract}/3grams.txt")

