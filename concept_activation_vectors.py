import os
import random
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from concurrent.futures import ProcessPoolExecutor

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support

from sklearn.decomposition import PCA

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification, ViTConfig

import matplotlib.pyplot as plt

from utils import *
from process_concepts import plot_concepts


RANDOM_SEED = 42
BATCH_SIZE = 1024
CONCEPT_SAMPLES = 50
NUM_CLF_FOR_CONCEPT = 10
MAX_CONCEPTS_SAMPLES = 5000
MAX_PROCESS = 32


TASK = "category"
OUTPUT_DIR = f"./fashion/{TASK}_prediction/"


feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

MODELS = dict()

def test_transform(samples):
    transform = Compose(
            [
                Resize(feature_extractor.size),
                CenterCrop(feature_extractor.size),
                ToTensor(),
                Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
            ]
        )
    samples['pixel_values'] = [transform(image.convert("RGB")) for image in samples['img']]
    return samples


def collate_fn(samples):
    pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
    labels = torch.tensor([sample["label"] for sample in samples])
    return {"pixel_values": pixel_values, "label": labels}


def get_dataloader_for_product(product):
    p_imgs = get_imgs_for_product(product)
    dataset = dict(img=p_imgs, label=[0]*len(p_imgs))
    dataset = Dataset.from_dict(dataset)
    dataset.set_transform(test_transform)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader


def init_model(model_name, device):
    if model_name in MODELS:
        return MODELS[model_name]

    if model_name == "ft_vit":
        if TASK == "category":
            checkpoint = f"./fashion/category_prediction/checkpoint-343450"
        elif TASK == "price":
            checkpoint = "./fashion/price_prediction/checkpoint-103400"
        model = ViTForImageClassification.from_pretrained(checkpoint)
    elif model_name == "vit":
        vit_checkpoint = "google/vit-base-patch16-224-in21k"
        model = ViTForImageClassification.from_pretrained(vit_checkpoint)
    elif model_name == "random_vit":
        vitconfig = ViTConfig()
        model = ViTForImageClassification(vitconfig)
    else:
        raise Exception(f"Unknown model name: {model_name}")

    # model = model.to(device)
    model.config.output_hidden_states = True
    MODELS[model_name] = model

    return model


def generate_product_embeddings():
    np.random.seed(RANDOM_SEED)
    model_names = ["ft_vit", "vit", "random_vit"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    concept_train_products = load_json("./fashion/concepts_data/train.json")
    concept_dev_products = load_json("./fashion/concepts_data/dev.json")
    concept_test_products = load_json("./fashion/concepts_data/test.json")

    paths = glob(f"{OUTPUT_DIR}/asin2embeddings*")
    asin2embeddings = dict()
    for model_name in model_names:
        if model_name not in asin2embeddings:
            asin2embeddings[model_name] = dict()
    
    for path in paths:
        print(f"found {path}, resuming")
        with open(path, "rb") as f:
            partial_asin2embeddings = pickle.load(f)
            for k,v in partial_asin2embeddings.items():
                asin2embeddings[k].update(v)  
    
    print({k:len(v) for k,v in asin2embeddings.items()})
    return asin2embeddings
    # products = concept_train_products + concept_dev_products + concept_test_products
    print(f"train products: {len(concept_train_products)}")
    print(f"dev products: {len(concept_dev_products)}")
    print(f"test products: {len(concept_test_products)}")

    all_products = concept_train_products + concept_dev_products
    # all_products = concept_train_products + concept_dev_products + concept_test_products
    # all_products = concept_test_products
    remaining_products = [p for p in all_products if any([p["asin"] not in v for v in asin2embeddings.values()])]
    random.shuffle(remaining_products)
    print(f"{len(remaining_products)} products left to process.. ({len(all_products) - len(remaining_products)} done)")

    # return asin2embeddings
    
    split_size = 180
    product_splits = [remaining_products[i:i+split_size] for i in range(0, len(remaining_products), split_size)]

    # for products in [concept_dev_products, concept_test_products, concept_train_products]:
    for split_index, products in enumerate(product_splits):
        print(f"preparing dataloaders for {len(products)} products (split {split_index}/{len(product_splits)})")
        with ProcessPoolExecutor(max_workers=MAX_PROCESS) as exec:
            for dataloader, product in tqdm(zip(exec.map(get_dataloader_for_product, products), products)):
                product["dataloader"] = dataloader

        for model_name in model_names:
            if model_name not in asin2embeddings:
                asin2embeddings[model_name] = dict()
            
            done = len([product["asin"] for product in products if product["asin"] in asin2embeddings[model_name]])
            print(f"generating {model_name} embeddings for {len(products) - done} products ({done} done)")

            model = init_model(model_name, device)
            model = model.to(device)
            model.eval()

            for index, product in tqdm(enumerate(products)):
                asin = product["asin"]
                dataloader = product["dataloader"]
                embeddings = []

                if asin in asin2embeddings[model_name]:
                    continue

                if len(dataloader.dataset) == 0:
                    continue

                with torch.no_grad():
                    for inputs in dataloader:
                        pixel_values = inputs["pixel_values"].to(device)
                        outputs = model(pixel_values)
                        batch_embeddings = outputs.hidden_states
                        batch_embeddings = batch_embeddings[-1][: , 0, :].cpu()
                        embeddings.append(batch_embeddings)

                embeddings = torch.cat(embeddings, dim=0).numpy()
                asin2embeddings[model_name][asin] = embeddings
                
            model = model.cpu()
        print("saving embeddings...")
        print({k:len(v) for k,v in asin2embeddings.items()}) 
        with open(f"{OUTPUT_DIR}/asin2embeddings_{SPLIT_START_INDEX}.pkl", "wb") as f:
            pickle.dump(asin2embeddings, f)  
        

        for product in products:
            dataloader = product["dataloader"]
            product["dataloader"] = None
            del dataloader.dataset
            del dataloader

    return asin2embeddings


def load_concepts_data(output_dir):
    path = f"{output_dir}/concepts_activation_vectors.pkl"
    if os.path.exists(path):
        print(f"found {path}, resuming")
        with open(path, "rb") as f:
            concepts_data = pickle.load(f)
    else:
        concepts_data = dict()
    
    concepts_data = {k:v for k,v in concepts_data.items() if v}
    return concepts_data    


def get_pos_neg_asins_for_concept(products, concept, asins2embeddings):
    positive_products = []
    negative_products = []
    for product in products:
        asin = product["asin"]
        
        has_embeddings = True
        for model_embeddings in asins2embeddings.values():
            if asin not in model_embeddings:
                has_embeddings = False

        if not has_embeddings:
            continue

        if concept in product["concepts"]:
            positive_products.append(asin)
        else:
            negative_products.append(asin)
    
    random.shuffle(negative_products)
    negative_products = negative_products[:len(positive_products)]
    return positive_products, negative_products
    

def get_embeddings_and_ladels_from_asins(positive_asins, negative_asins, model_asins2embeddings):
    labels = []
    embeddings = []
    for asins_list, label in [(positive_asins, 1), (negative_asins, 0)]:
        for asin in asins_list:
            product_embeddings = model_asins2embeddings[asin]
            product_labels = [label] * len(product_embeddings)
            
            embeddings.append(product_embeddings)
            labels.extend(product_labels)

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.asarray(labels)
    return embeddings, labels


def fit_linear_clf(train_embeddings, train_labels, dev_embeddings, dev_labels):
    random_state = np.random.randint(10000)
    concept_classifier = Pipeline([
        ("scl", StandardScaler()),
        ("clf", SGDClassifier(max_iter=1000, tol=1e-3, random_state=random_state))
    ])
    concept_classifier.fit(train_embeddings, train_labels)

    train_predict = concept_classifier.predict(train_embeddings)
    clf_train_acc = (train_labels == train_predict).sum() / train_labels.shape[0]
    
    dev_predict = concept_classifier.predict(dev_embeddings)
    clf_dev_acc = (dev_labels == dev_predict).sum() / dev_labels.shape[0]

    return concept_classifier, clf_train_acc, clf_dev_acc


def generate_concept_activation_vector(args):
    concept, model_names, asins2embeddings, concept_train_products, concept_dev_products = args
    
    concept_data = dict()
    
    for model_name in model_names:
        concept_data[model_name] = dict()

        best_dev_acc = 0
        best_classifier = None
        train_accuracies = []
        dev_accuracies = []
        all_classifiers = []
        
        for clf_index in range(NUM_CLF_FOR_CONCEPT):
            train_positive_asins, train_negative_asins = get_pos_neg_asins_for_concept(concept_train_products, concept, asins2embeddings)
            dev_positive_asins, dev_negative_asins = get_pos_neg_asins_for_concept(concept_dev_products, concept, asins2embeddings)

            if not train_positive_asins or not dev_positive_asins:
                return None, None

            train_embeddings, train_labels = get_embeddings_and_ladels_from_asins(train_positive_asins, train_negative_asins, asins2embeddings[model_name])
            dev_embeddings, dev_labels = get_embeddings_and_ladels_from_asins(dev_positive_asins, dev_negative_asins, asins2embeddings[model_name])
            
            concept_classifier, clf_train_acc, clf_dev_acc = fit_linear_clf(train_embeddings, train_labels, dev_embeddings, dev_labels)
            # print(f"\t--- clf #{clf_index} {concept} {model_name}: train {clf_train_acc}, dev {clf_dev_acc}")

            all_classifiers.append(concept_classifier)
            dev_accuracies.append(clf_dev_acc)
            train_accuracies.append(clf_train_acc)

            if clf_dev_acc > best_dev_acc:
                best_dev_acc = clf_dev_acc
                best_classifier = clf_index
        
        for i in range(len(all_classifiers)):
            if i != best_classifier:
                all_classifiers[i] = None

        concept_data[model_name]["train_accuracies"] = train_accuracies
        concept_data[model_name]["dev_accuracies"] = dev_accuracies
        concept_data[model_name]["best_classifiers"] = best_classifier     
        concept_data[model_name]["all_classifiers"] = all_classifiers
    
    return concept, concept_data


def get_price_label(raw_price):
    if isinstance(raw_price, int) or isinstance(raw_price, float):
        return float(raw_price)
    elif raw_price.count("$") == 1:
        return float(raw_price.replace("$", "").replace(",", ""))
    elif raw_price.count("$") == 2:
        min_value, max_value = raw_price.split(" - ")
        min_value = float(min_value.replace("$", "").replace(",", ""))
        max_value = float(max_value.replace("$", "").replace(",", ""))
        return (min_value + max_value) / 2
    else:
        raise Exception(f"unknown format: {raw_price}")
        

def generate_concepts_data():
    np.random.seed(RANDOM_SEED)
    model_names = ["ft_vit", "vit", "random_vit"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    concept_train_products = load_json("./fashion/concepts_data/train.json")
    concept_dev_products = load_json("./fashion/concepts_data/dev_with_ngram_concepts.json")
    concept_test_products = load_json("./fashion/concepts_data/test_with_ngram_concepts.json")

    print(len(concept_train_products),len(concept_dev_products),len(concept_test_products))
    
    asin2embeddings = generate_product_embeddings()
    # concepts_data = load_concepts_data(OUTPUT_DIR)
    concepts_data = dict()

    concepts = set.union(*[set(p["concepts"]) for p in concept_train_products])
    concepts = concepts.difference(set(concepts_data.keys()))

    print(f"{len(concepts)} concepts ({len(concepts_data.keys())} processed)")
    
    args = zip(
        concepts,
        itertools.repeat(model_names),
        itertools.repeat(asin2embeddings),
        itertools.repeat(concept_train_products),
        itertools.repeat(concept_dev_products)
    )

    # plot_concepts(concepts_data, output_dir=output_dir)
    with ProcessPoolExecutor(max_workers=MAX_PROCESS) as exec:
        for concept, concept_data in tqdm(exec.map(generate_concept_activation_vector, args)):
            try:
                if concept_data:
                    concepts_data[concept] = concept_data
                    if len(concepts_data) % 500 == 0:
                        print("plotting all concepts so far...")
                        plot_concepts(concepts_data, output_dir=OUTPUT_DIR)
            finally:
                # print("saving concepts data...")
                with open(f"{OUTPUT_DIR}/concepts_activation_vectors.pkl", "wb") as f:
                    pickle.dump(concepts_data, f)

    plot_concepts(concepts_data, output_dir=OUTPUT_DIR)

                            



# generate_product_embeddings()
generate_concepts_data()