from glob import glob
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import *
from define import TRAIN, DEV


MODEL_NAMES = ["ft_vit", "vit"]


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
    random_state = np.random.randint(SEED)
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


def generate_concept_activation_vector(curr_state):
    concept_data = dict()
    
    for model_name in MODEL_NAMES:
        concept_data[model_name] = dict()

        best_dev_acc = 0
        best_classifier = None
        train_accuracies = []
        dev_accuracies = []
        all_classifiers = []
        
        for clf_index in range(NUM_CLF_FOR_CONCEPT):
            random.shuffle(curr_state.train_neg)
            train_positive_asins = [p["asin"] for p in curr_state.train_pos]
            train_negative_asins = [p["asin"] for p in curr_state.train_neg[:len(curr_state.train_pos)]]
            
            random.shuffle(curr_state.dev_neg)
            dev_positive_asins = [p["asin"] for p in curr_state.dev_pos]
            dev_negative_asins = [p["asin"] for p in curr_state.dev_neg[:len(curr_state.dev_pos)]]
            
            if not train_positive_asins or not dev_positive_asins:
                return None, None

            train_embeddings, train_labels = get_embeddings_and_ladels_from_asins(train_positive_asins, train_negative_asins, curr_state.asin2embeddings[model_name])
            dev_embeddings, dev_labels = get_embeddings_and_ladels_from_asins(dev_positive_asins, dev_negative_asins, curr_state.asin2embeddings[model_name])
            
            concept_classifier, clf_train_acc, clf_dev_acc = fit_linear_clf(train_embeddings, train_labels, dev_embeddings, dev_labels)

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
    
    return concept_data

