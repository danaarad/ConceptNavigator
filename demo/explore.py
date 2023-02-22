import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

import umap
from scipy import stats
from sklearn.manifold import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE

from config import SEED



def check_statistical_significance_for_new_concept(concept, new_concept_data, accuracy_threshold=0.9, pvalue_threshold=0.05, debug=True):
    new_concept_data = {concept: new_concept_data}
    new_concept_data = check_statistical_significance_accuracy(new_concept_data, accuracy_threshold, pvalue_threshold, debug)
    new_concept_data = check_statistical_significance_diff_from_vit(new_concept_data, pvalue_threshold, debug)
    return new_concept_data.pop(concept)


def check_statistical_significance_accuracy(concepts_data, accuracy_threshold=0.9, pvalue_threshold=0.05, debug=True):
    for concept in concepts_data.keys():
        dev_accuracies = concepts_data[concept]["ft_vit"]["dev_accuracies"]
        ttest_result = stats.ttest_1samp(dev_accuracies, popmean=accuracy_threshold, alternative="greater")
        if ttest_result.pvalue < pvalue_threshold:
            output = f"We reject the null hypothesis for concept {concept}. Mean concept accuracy > {accuracy_threshold} with confidence of {round(ttest_result.pvalue,3)} < {pvalue_threshold}."
            if debug:
                print(output)
            concepts_data[concept]["gt_than_threshold"] = True
            concepts_data[concept]["gt_than_threshold_output"] = output
        else:
            output = f"We cannot reject the null hypothesis for concept {concept}. P-value {round(ttest_result.pvalue,3)} > {pvalue_threshold}"
            if debug:
                print(output)
            concepts_data[concept]["gt_than_threshold"] = False
            concepts_data[concept]["gt_than_threshold_output"] = output
    return concepts_data


def check_statistical_significance_diff_from_vit(concepts_data, pvalue_threshold=0.05, debug=True):
    for concept in concepts_data.keys():
        if "gt_than_vit" in concepts_data[concept]:
            continue

        ft_vit_dev_accuracies = concepts_data[concept]["ft_vit"]["dev_accuracies"]
        vit_dev_accuracies = concepts_data[concept]["vit"]["dev_accuracies"]

        ttest_result = stats.ttest_ind(ft_vit_dev_accuracies, vit_dev_accuracies, alternative="greater")
        if ttest_result.pvalue < pvalue_threshold:
            output = f"We reject the null hypothesis for concept {concept}. Mean accuracy of the concept for downstream task > mean accuarcy without fine tuning, with confidence of {round(ttest_result.pvalue,3)} < {pvalue_threshold}."
            if debug:
                print(output)
            concepts_data[concept]["gt_than_vit"] = True
            concepts_data[concept]["gt_than_vit_output"] = output
        else:
            output = f"We cannot reject the null hypothesis for concept {concept}. P-value {round(ttest_result.pvalue,3)} > {pvalue_threshold}"
            if debug:
                print(output)
            concepts_data[concept]["gt_than_vit"] = False
            concepts_data[concept]["gt_than_vit_output"] = output
    return concepts_data


def get_statistically_significant_concepts(concepts_data, new_concept=None):
    concepts = []
    coefs = []
    dev_acc = []

    for concept in concepts_data.keys():
        new_concept_cond = new_concept and new_concept == concept
        verified_concept_cond = concepts_data[concept]["gt_than_threshold"] and concepts_data[concept]["gt_than_vit"]
        if new_concept_cond or verified_concept_cond: 
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


def create_concepts_image(path, alg, task, dataset, accuracy_threshold, perplexity=7, n_neighbors=15, min_dist=0.01, new_concept=None, new_concept_data=None, seed=None):
    with open(f"/home/daarad/concepts/{dataset}/{task}_prediction/concepts_activation_vectors.pkl", "rb") as f:
        concepts_data = pickle.load(f)

    if new_concept:
        concepts_data[new_concept] = new_concept_data
    
    concepts_data = check_statistical_significance_accuracy(concepts_data, accuracy_threshold=accuracy_threshold, debug=False)
    concepts_data = check_statistical_significance_diff_from_vit(concepts_data, debug=False)
    concepts, coefs, best_dev_accs = get_statistically_significant_concepts(concepts_data, new_concept)

    seed = seed or SEED

    print(f"running {alg}...")
    # T-SNE
    # if alg == "tsne":
    #     coefs = TSNE(
    #         n_components=2, learning_rate='auto',
    #         init='random', perplexity=perplexity, random_state=SEED
    #         ).fit_transform(coefs)
    #     coefs_x = coefs[:, 0]
    #     coefs_y = coefs[:, 1]

    if alg == "tsne":
        print("perplexity:", perplexity)
        embeddings = TSNE(n_jobs=2, 
            n_components=2, learning_rate='auto',
            perplexity=perplexity, init="random", random_state=seed
        ).fit_transform(coefs)
        coefs_x = embeddings[:, 0]
        coefs_y = embeddings[:, 1]
    # UMAP
    if alg == "umap":
        trans = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed).fit(coefs)
        coefs_x = trans.embedding_[:, 0]
        coefs_y = trans.embedding_[:, 1]
    
    print("creating figure...")
    plt.clf()
    plt.figure(figsize=(10,8))

    # Selecting the axis-X making the bottom and top axes False.
    plt.tick_params(axis='x', which='both', bottom=False,
                    top=False, labelbottom=False)
    
    # Selecting the axis-Y making the right and left axes False
    plt.tick_params(axis='y', which='both', right=False,
                    left=False, labelleft=False)

    plt.scatter(coefs_x, coefs_y, alpha=0.3, cmap='rainbow', c=np.arctan2(coefs_x, coefs_y))

    texts = []
    new_added = False
    for i, (concept, acc) in enumerate(zip(concepts, best_dev_accs)):
        if concept == new_concept and not new_added:
            fontsize = 14 
            text = plt.annotate(concept, (coefs_x[i], coefs_y[i]), fontsize=fontsize)
            texts.append(text)
            new_added = True
        elif acc > 0.8:
            fontsize = round((acc - 0.2) * 12)  
            text = plt.annotate(concept, (coefs_x[i], coefs_y[i]), fontsize=fontsize)
            text.set_alpha((acc - 0.7)/ 0.4)
            texts.append(text)

    adjust_text(texts, force_text=(0.15, 0.27))

    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
