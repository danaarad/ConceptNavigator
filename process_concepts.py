import json
import math
import shutil
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from adjustText import adjust_text


import nltk
from nltk.corpus import wordnet

# from PyDictionary import PyDictionary

from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from scipy.cluster import hierarchy

from scipy import stats

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
from transformers import ViTForImageClassification

import seaborn as sns
import matplotlib.pyplot as plt

from utils import *


def process_concepts(concept_filename):
    nltk.download('omw-1.4')
    nltk.download('wordnet')

    concepts = load_concepts(concept_filename)
    output = []
    for concept in concepts:
        concept = concept.strip()

        # wordnet synonyms
        synonyms = []
        for syn in wordnet.synsets(concept):
            for i in syn.lemmas():
                synonyms.append(i.name())

        # pydictionary synonyms
        dictionary = PyDictionary()
        synonyms.extend(dictionary.synonym(concept) or [])

        synonyms = set(synonyms)

        print("-" * 50)
        print(concept)
        print(synonyms)
        print("-" * 50)

        output.append(
            dict(type="expand", input=list(synonyms), output=[concept]))
    with open("./synonyms.json", "w") as f:
        json.dump(output, f)


def concepts_stats():
    with open("./electronics/products.json", "rb") as f:
        products = json.load(f)

    # # concepts hist
    print("concepts hist")
    concepts = [p["concepts"] for p in products]
    concepts_len = [len(c) for c in concepts]

    plt.figure().clear()
    ax = sns.histplot(data=concepts_len).set(title="histogram of number of concepts per product")
    plt.savefig("./concepts_hist.png")

    # concepts to asin
    print("concepts to asin")
    concepts2asin = dict()
    for product in products:
        prod_concepts = product["concepts"]
        prod_concepts.sort()
        prod_concepts = tuple(prod_concepts)

        if prod_concepts not in concepts2asin:
            concepts2asin[prod_concepts] = []

        concepts2asin[prod_concepts].append(product["asin"])

    # print("concept count")
    products_count = [(k, len(v)) for k, v in concepts2asin.items()]
    products_count.sort(key=lambda x: -x[1]) # sort by count
    with open("./top_concepts.txt", "wb") as f:
        for count in products_count:
            line = f"{count}\n".encode()
            f.write(line)

    # prods without concepts
    print("prods without concepts")
    asin2prods = {p["asin"]: p for p in products}
    asin_without_concepts = concepts2asin[tuple()]
    prods_without_concepts = []
    for asin in asin_without_concepts:
        prod = asin2prods[asin]
        prods_without_concepts.append(prod)

    print(len(prods_without_concepts))
    with open("./electronics/prods_without_concepts.json", "w") as f:
        json.dump(prods_without_concepts[:1000], f)

    # plot concept counts
    all_concepts = load_concepts("./fashion/fashion_concepts")
    concepts_count = {c:0 for c in all_concepts}
    for product in products:
        prod_concepts = product["concepts"]
        for c in prod_concepts:
            if c in ["formal"]:
                continue
            concepts_count[c] += 1
    concepts_count = list(concepts_count.items())
    concepts_count.sort(key=lambda x: -x[1])
    concepts, counts = zip(*concepts_count)

    jumps = 15
    for i in range(0, len(concepts), jumps):
        concepts_part = concepts[i:i+jumps]
        counts_part = counts[i:i+jumps]

        plt.figure().clear()
        plt.suptitle('Number of products per concept')
        plt.bar(concepts_part, counts_part)
        plt.ylabel("# products")
        plt.xticks(rotation=45)
        plt.tick_params(labelsize=7)

        plt.savefig(f"./top_concepts_{i}.png")



def get_concept_samples():
    POSITIVE_SAMPLES = 5000
    train = load_json("./fashion/concepts_data/train.json")
    dev = load_json("./fashion/concepts_data/dev.json")
    test = load_json("./fashion/concepts_data/test.json")

    concepts = set.union(*[set(p["concepts"]) for p in train+dev+test])
    print(f"{len(concepts)} concepts")

    for products, split in zip([train, dev, test], ["train", "dev", "test"]):
        print("-"*50)
        print(f"{split}: {len(products)} products")
        concepts_to_samples = dict()
        for concept in concepts:
            positive_samples = []
            negative_samples = []
            random.shuffle(products)
            
            for product in products:
                if concept in product["concepts"]:
                    if len(positive_samples) >= POSITIVE_SAMPLES:
                        continue
                    positive_samples.append(product)        
                
                if len(positive_samples) == POSITIVE_SAMPLES:
                    break

            concepts_to_samples[concept] = dict(pos=positive_samples, neg=negative_samples)
        
        with open(f"./fashion/concepts_data/concepts_to_samples_{split}.json", "w") as f:
            json.dump(concepts_to_samples, f)


def plot_dendrogram(model, **kwargs):
    # from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    hierarchy.set_link_color_palette(['forestgreen', 'orange', 'orangered', 'darkturquoise', 'deeppink', 'royalblue'])
    hierarchy.dendrogram(
        linkage_matrix,
        color_threshold=350,
        above_threshold_color='grey',
        orientation="right",
        **kwargs
        )


def plot_cav_dendrogram(output_dir=None, concepts_data=None):
    if not output_dir:
        output_dir = "./fashion/category_prediction/"
    if not concepts_data:
        with open(f"{output_dir}/concepts_activation_vectors.pkl", "rb") as f:
            concepts_data = pickle.load(f)

    concepts_data = check_statistical_significance_accuracy(concepts_data, acc_threshold=0.9)
    concepts_data = check_statistical_significance_diff_from_vit(concepts_data)
    concepts, coefs, best_dev_accs = get_statistically_significant_concepts(concepts_data)

    plt.clf()
    plt.figure(figsize=(10,8))
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(coefs)
    plt.title("CAVs Hierarchical Clustering")
    
    # leaf_label_func = lambda x: concepts[x]
    # plot the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=24, labels=concepts) #leaf_label_func=leaf_label_func)
    plt.tick_params(axis='x', which='major', labelsize=7) #, rotation=80)
    plt.savefig(f"{output_dir}/concepts_dendrogram.png", dpi=300)


def get_concept_imgs_examples(concepts):
    train = load_json("./fashion/concepts_data/concepts_to_samples_train.json")
    all_products = load_json("./fashion/concepts_data/train.json")
    files_to_copy = []

    for concept in concepts:
        positive_products = train[concept]["pos"][:10]
        negative_products = []

        random.shuffle(all_products)
        for p in all_products:
            if concept not in p["concepts"]:
                if len(negative_products) == len(positive_products):
                    break
                negative_products.append(p)

        products_for_imgs = positive_products + negative_products
        product_labels = ["pos"] * len(positive_products) + ["neg"] * len(negative_products)

        for p, l in zip(products_for_imgs, product_labels):
                asin = p["asin"]
                asin_pref = asin[:2]
                img_files = list(glob(f"./fashion/imgs/{asin_pref}/{asin}/*.jpg"))
                copied = 0

                for f in img_files:
                    try:
                        img = Image.open(f)
                        width, height = img.size
                        if width < 224 or height < 224:
                            continue
                    except UnidentifiedImageError:
                        continue

                    img_name = f.replace(
                        f"./fashion/imgs/{asin_pref}/{asin}/",
                        ""
                    )

                    old_path = f
                    new_path = f"./fashion/concept_plots/{concept}/{l}_{asin}_{img_name}"
                    files_to_copy.append((old_path, new_path))
                    copied += 1
                    
                    if copied > 2:
                        break
                    
    for old_path, new_path in files_to_copy:
        shutil.copy(old_path, new_path)


def concept_scatter_plot_from_xy(x, y, concepts, dev_accs, output_dir, param_str):
    plt.clf()
    plt.figure(figsize=(10,8))
    # areas = [((acc - 30) * 15)**2 for acc in dev_acc]
    plt.scatter(x, y, alpha=0.3, cmap='rainbow', c=np.arctan2(x, y))

    texts = []
    for i, (concept, acc) in enumerate(zip(concepts, dev_accs)):
        if acc > 0.8:
            fontsize = round((acc - 0.2) * 12)  
            text = plt.annotate(concept, (x[i], y[i]), fontsize=fontsize)
            text.set_alpha((acc - 0.7)/ 0.4)
            texts.append(text)
    adjust_text(texts)

    plt.title("Concepts Scatter Plot")
    plt.savefig(f"{output_dir}/all_concepts_scatter_plot_{param_str}.png",  dpi=300)
    plt.close()


def is_statistically_significant(dev_accuracies, acc_threshold, pvalue_threshold, debug=True):
    result = stats.ttest_1samp(dev_accuracies, popmean=acc_threshold, alternative="greater")
    if result.pvalue < pvalue_threshold:
        print(f"greater then {acc_threshold} with confidence of {result.pvalue} > {pvalue_threshold}")
        return True
    else:
        print(f"NOT greater then {acc_threshold} with confidence of {result.pvalue} > {pvalue_threshold}")
        return False


def check_statistical_significance_accuracy(concepts_data, acc_threshold=0.9, pvalue_threshold=0.05, debug=True):
    for concept in concepts_data.keys():
        dev_accuracies = concepts_data[concept]["ft_vit"]["dev_accuracies"]

        ttest_result = stats.ttest_1samp(dev_accuracies, popmean=acc_threshold, alternative="greater")
        if ttest_result.pvalue < pvalue_threshold:
            if debug:
                print(f"we reject the null hyp (mean = {acc_threshold}) for concept {concept}, with confidence of {ttest_result.pvalue} < {pvalue_threshold}, mean > {acc_threshold}")
            concepts_data[concept]["gt_than_threshold"] = True
        else:
            if debug:
                print(f"we cannot reject the null hyp (mean = {acc_threshold}) for concept {concept}. pvalue {ttest_result.pvalue} > {pvalue_threshold}")
            concepts_data[concept]["gt_than_threshold"] = False
    return concepts_data


def check_statistical_significance_diff_from_vit(concepts_data, pvalue_threshold=0.05, debug=True):
    for concept in concepts_data.keys():
        if "gt_than_vit" in concepts_data[concept]:
            continue

        ft_vit_dev_accuracies = concepts_data[concept]["ft_vit"]["dev_accuracies"]
        vit_dev_accuracies = concepts_data[concept]["vit"]["dev_accuracies"]

        ttest_result = stats.ttest_ind(ft_vit_dev_accuracies, vit_dev_accuracies, alternative="greater")
        if ttest_result.pvalue < pvalue_threshold:
            if debug:
                print(f"we reject the null hyp (mean_ft = mean_vit) for concept {concept}, with confidence of {ttest_result.pvalue} < {pvalue_threshold}, mean_ft > mean_vit")
            concepts_data[concept]["gt_than_vit"] = True
        else:
            if debug:
                print(f"we cannot reject the null hyp (mean_ft = mean_vit) for concept {concept}. pvalue {ttest_result.pvalue} > {pvalue_threshold}")
            concepts_data[concept]["gt_than_vit"] = False
    return concepts_data


def get_statistically_significant_concepts(concepts_data):
    concepts = []
    coefs = []
    dev_acc = []

    for concept in concepts_data.keys():
        if concepts_data[concept]["gt_than_threshold"] and concepts_data[concept]["gt_than_vit"]: 
            concepts.append(concept)
        
            best_classifier_index = concepts_data[concept]["ft_vit"]["best_classifiers"]
            best_classifier = concepts_data[concept]["ft_vit"]["all_classifiers"][best_classifier_index]
            
            best_classifier_coefs = best_classifier.named_steps['clf'].coef_
            coefs.append(best_classifier_coefs)

            dev_accuracies = concepts_data[concept]["ft_vit"]["dev_accuracies"]
            best_classifier_accuracy = dev_accuracies[best_classifier_index]
            dev_acc.append(best_classifier_accuracy)

    coefs = np.concatenate(coefs, axis=0)
    return concepts, coefs, dev_acc


def plot_concepts(concepts_data=None, output_dir=None, ):
    if not output_dir:
        output_dir = "./"
    if not concepts_data:
        with open(f"{output_dir}/concepts_activation_vectors.pkl", "rb") as f:
            concepts_data = pickle.load(f)

    concepts_data = check_statistical_significance_accuracy(concepts_data, acc_threshold=0.6)
    concepts_data = check_statistical_significance_diff_from_vit(concepts_data)

    with open(f"{output_dir}/concepts_activation_vectors.pkl", "wb") as f:
            pickle.dump(concepts_data, f)

    concepts, coefs, best_dev_accs = get_statistically_significant_concepts(concepts_data)
    print(len(concepts))
    # T-SNE
    for perplexity in tqdm([4, 5, 7, 20, 30, 40, 50, 70, 100]):
        coefs = TSNE(
            n_components=2, learning_rate='auto',
            init='random', perplexity=perplexity
            ).fit_transform(coefs)
        coefs_x = coefs[:, 0]
        coefs_y = coefs[:, 1]
        concept_scatter_plot_from_xy(coefs_x, coefs_y, concepts, best_dev_accs, output_dir, f"p{perplexity}")

    # UMAP
    for n_neighbors in [15, 30, 50]:
        for min_dist in [0.01, 0.05, 0.1, 0.5, 1]:
            print(n_neighbors, min_dist)
            trans = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=123).fit(coefs)
            coefs_x = trans.embedding_[:, 0]
            coefs_y = trans.embedding_[:, 1]
            concept_scatter_plot_from_xy(coefs_x, coefs_y, concepts, best_dev_accs, output_dir, f"n{n_neighbors}_d{min_dist}")



def plot_concepts_correlation():
    embeddings_test_products = load_json("./fashion/embedding_data/test.json")
    concepts = set.union(*[set(p["concepts"]) for p in embeddings_test_products])
    concepts2asins = {c:[] for c in concepts}
    
    for product in embeddings_test_products:
        for concept in concepts2asins:
            value = 1 if concept in product["concepts"] else 0
            concepts2asins[concept].append(value)

    df = pd.DataFrame(concepts2asins)
    
    print_highly_correlated(df)

    plt.clf()
    plt.figure(figsize=(10,8))
    plt.matshow(df.corr())
    plt.title("Concepts Correlation (Embeddings Test Set)")
    plt.xticks(list(range(213)), [""] + df.columns.values, fontsize=3, rotation=90)
    plt.yticks(list(range(213)), [""] + df.columns.values, fontsize=3)
    plt.savefig(f"./fashion/concepts_data/concepts_correlation.png",  dpi=500)


def print_highly_correlated(df, threshold=0.5):
    corr_df = df.corr()
    correlated_features = np.where(np.abs(corr_df) > threshold)
    correlated_features = [(corr_df.iloc[x,y], x, y) for x, y in zip(*correlated_features) if x != y and x < y]
    s_corr_list = sorted(correlated_features, key=lambda x: -abs(x[0]))
    
    if s_corr_list == []:
        print("There are no highly correlated features with correlation above", threshold)
    else:
        with open("./fashion/concepts_data/concepts_correlation.txt", "wb") as f:
            for v, i, j in s_corr_list:
                output = "%s and %s = %.3f\n" % (corr_df.index[i], corr_df.columns[j], v)
                f.write(output.encode())



def get_tcav_score_for_class(concepts_data, asins2embeddings, class_test_products, output_file):
    # concept_test_products = load_json("./fashion/concepts_data/test_with_ngram_concepts.json")
    # class_test_products = [p for p in concept_test_products if class_to_test in p["short_categories"]]

    concepts, _, _ = get_statistically_significant_concepts(concepts_data)

    tcav_scores = []
    for concept in concepts:
        clf_index = concepts_data[concept]["ft_vit"]["best_classifiers"]
        clf = concepts_data[concept]["ft_vit"]["all_classifiers"][clf_index]
        
        embeddings = []
        # variations: x \in class can be for every input image or every product
        skipped = 0
        for product in class_test_products:
            asin = product["asin"]
            if asin not in asins2embeddings["ft_vit"]:
                skipped += 1
                continue
            product_embeddings = asins2embeddings["ft_vit"][asin]
            embeddings.append(product_embeddings)
            # predict = clf.predict(product_embeddings)
            # tcav_score = (predict > 0).sum() / product_embeddings.shape[0]
            # if tcav_score == 0:
            #     print(f"{asin}: {tcav_score}")
        embeddings = np.concatenate(embeddings, axis=0)
        predict = clf.predict(embeddings)
        tcav_score = (predict > 0).sum() / embeddings.shape[0]
        tcav_scores.append(tcav_score)

        # print(f"concept {concept}: {tcav_score}. skipped: {skipped}")    

    return concepts, tcav_scores

    # with open(output_file, "wb") as f:
    #     for concept, tcav in zip(concepts, tcav_scores):
    #         line = f"{concept},{tcav}\n".encode()
    #         f.write(line)



def get_model_prediction_for_class(model, products, class_idx):
    pass



def get_concept_diff(price_concepts_data, category_concepts_data):
    price_concepts, _, _ = get_statistically_significant_concepts(price_concepts_data)
    category_concepts, _, _ = get_statistically_significant_concepts(category_concepts_data)
    price_concepts = set(price_concepts)
    category_concepts = set(category_concepts)

    price_only = price_concepts.difference(category_concepts)
    print(f"price only: {len(price_only)}")
    for concept in price_only:
        print(f"\t{concept}")

    category_only = category_concepts.difference(price_concepts)
    print(f"price only: {len(category_only)}")
    for concept in category_only:
        print(f"\t{concept}")

    intersection = price_concepts.intersection(category_concepts)
    print(f"intersection: {len(intersection)}")
    for concept in intersection:
        print(f"\t{concept}")
