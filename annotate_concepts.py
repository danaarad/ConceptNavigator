import re
import json
import gzip
import pprint
from tkinter.tix import IMMEDIATE
from tqdm import tqdm

import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize

from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

from utils import *


nltk.download('punkt')


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



def load_rules(filename):
    with open(filename, "r") as f:
        rules = json.load(f)

    expand_rules, any_rules, all_rules = [], [], []
    for rule in rules:
        if rule["type"] == "expand":
            expand_rules.append(rule)
        elif rule["type"] == "any":
            any_rules.append(rule)
        elif rule["type"] == "all":
            all_rules.append(rule)
        else:
            print('unknown rule type: ', rule)

    return expand_rules, any_rules, all_rules


def collect_concepts_from_list(concept_list, data):
    concepts = set()
    for concept in concept_list:
        if concept in data or pluralize(concept) in data:
            concepts.add(concept)
    return concepts


def collect_concepts_from_rule(rule, data):
    concepts = set()
    for inp in rule["input"]:
        if inp in data or pluralize(inp) in data:
            concepts.update(rule["output"])
    return concepts


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


def get_img_url_from_product(product):
    urls = []
    if "imageURLHighRes" in product:
        if isinstance(product["imageURLHighRes"], list):
            urls.extend(product["imageURLHighRes"])
        elif isinstance(product["imageURLHighRes"], str):
            urls.append(product["imageURLHighRes"])

    if "imageURL" in product:
        if isinstance(product["imageURL"], list):
            urls.extend(product["imageURL"])
        elif isinstance(product["imageURL"], str):
            urls.append(product["imageURL"])
    return urls


def collect_concepts_from_product(product_data, concepts, rules):
    expand_rules, any_rules, all_rules = rules
    product_concepts = set()

    product_concepts.update(collect_concepts_from_list(concepts, product_data))
    for rule in expand_rules:
        product_concepts.update(collect_concepts_from_rule(rule, product_data))

    for _ in range(5):
        for rule in any_rules:
            product_concepts.update(collect_concepts_from_rule(rule, product_data))
    return product_concepts


def process_raw_categories(all_categories, raw_product_categories):
    product_categories = []

    for category in raw_product_categories:
        category = category.strip().lower()
        if category in all_categories:
            product_categories.append(category)

    return product_categories


def process_product(args):
    # product, concepts, rules = args 
    # img_url = get_img_url_from_product(product)

    # # only products with images
    # if not img_url:
    #     return None

    # # only products with both categories and price:
    # if not (product.get("category") and product.get("price")):
    #     return None

    # product_data = get_product_data(product)
    # product_concepts = collect_concepts_from_product(product_data, concepts, rules)

    # data = dict(
    #     asin=product["asin"],
    #     title=product.get("title", ""),
    #     description=product.get("description", ""),
    #     feature=product.get("feature", []),
    #     img_url=img_url,
    #     raw_categories=product["category"],
    #     # categories=process_raw_categories(categories, product["category"]),
    #     price=product["price"],
    #     concepts=list(product_concepts)
    # )
    # return data

    product, concepts, rules = args 
    product_data = get_product_data(product)
    product_concepts = collect_concepts_from_product(product_data, concepts, rules)
    product["concepts"] = list(product_concepts)
    return product


def annotate_concepts_from_raw_product_file(concept_filename, products_filename, rules_filename, categories_file):
    products = load_products(products_filename)
    concepts = load_concepts(concept_filename)
    rules = load_rules(rules_filename)
    # categories = load_categories(categories_file)

    skipped_products = 0
    annotated_products = []

    print("processing products")
    args = zip(products, repeat(concepts), repeat(rules))
    with ProcessPoolExecutor(max_workers=50) as executor:
        for data in tqdm(executor.map(process_product, args)):
            if data: 
                annotated_products.append(data)
            else:
                skipped_products += 1

    print("saving products")
    with open("./electronics/products.json", "w") as f:
        json.dump(annotated_products, f)


def annotate_concepts_from_proccessed_product_file(concept_filename, product_filename, output_filename, rules_filename, ngram_files):
    products = load_json(product_filename)
    concepts = load_concepts(concept_filename)
    concepts.update(load_ngrams(ngram_files))
    rules = load_rules(rules_filename)

    skipped_products = 0
    annotated_products = []

    print(f"processing {len(products)} products with {len(concepts)} concepts")
    args = zip(products, repeat(concepts), repeat(rules))
    with ProcessPoolExecutor(max_workers=32) as executor:
        for data in tqdm(executor.map(process_product, args)):
            if data is not None: 
                annotated_products.append(data)
            else:
                skipped_products += 1

    print("saving products")
    with open(output_filename, "w") as f:
        json.dump(annotated_products, f)


