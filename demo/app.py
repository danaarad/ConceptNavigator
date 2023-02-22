import os
import sys
import json
from flask import Flask, request, send_from_directory, send_file

from config import *
from explore import create_concepts_image, check_statistical_significance_for_new_concept
from define import DefineRequest, get_image_paths
from train import generate_concept_activation_vector

app = Flask(__name__)
recent_requests = []
MAX_REQUESTS = 10

################################
######### STATIC PAGES #########
################################

@app.route('/')
def return_homepage():
    return send_from_directory(UI_FILES_DIR, "index.html")

@app.route('/explore')
def return_explore():
    return send_from_directory(UI_FILES_DIR, "explore.html")

    
@app.route('/define')
def return_define():
    return send_from_directory(UI_FILES_DIR, "define.html")


@app.route('/ui/<path:filename>')
def return_ui_file(filename):
    return send_from_directory(UI_FILES_DIR, filename)


@app.route('/ui/product_images/<asin>/<image_id>')
def return_product_image(asin, image_id):
    asin_prefix = asin[:2]
    filename = f"{PRODUCT_IMAGES_DIR}/{asin_prefix}/{asin}/{image_id}.jpg"
    return send_file(filename, mimetype='image/jpeg')


################################
######### EXPLORE ##############
################################


@app.route('/get_concept_image', methods=["POST"])
def get_concept_image():
    print(request.json)
    dataset = request.json.get('dataset')
    if dataset not in ["fashion", "electronics"]:
        return EMPTY_RESPONSE
    
    task = request.json.get('task')
    if task not in ["price", "category"]:
        return EMPTY_RESPONSE

    method = request.json.get('method')
    accuracy_threshold = int(request.json.get('accuracy_threshold')) / 100
    if not method or not accuracy_threshold:
        return EMPTY_RESPONSE

    str_acc = str(accuracy_threshold).replace(".", "_")
    if method == "tsne":
        perplexity = int(request.json['perplexity'])
        path = f"./ui/images/{dataset}/{task}/concpets_p{perplexity}_accth{str_acc}.png"
        if not os.path.exists(path):
            create_concepts_image(path, method, task, dataset, accuracy_threshold, perplexity=perplexity)
        return json.dumps(dict(img_src=path))
    if method == "umap":
        n_neighbors = int(request.json['neighbors'])
        min_dist = float(request.json['distance'])
        str_min_dist = str(min_dist).replace(".", "_")
        path = f"./ui/images/{dataset}/{task}/concpets_n{n_neighbors}_d{str_min_dist}_accth{str_acc}.png"
        if not os.path.exists(path):
            create_concepts_image(path, method, task, dataset, accuracy_threshold, n_neighbors=n_neighbors, min_dist=min_dist)
        return json.dumps(dict(img_src=path))
    else:
        return EMPTY_RESPONSE

################################
########## DEFINE ##############
################################

@app.route('/define/<concept>', methods=["POST"])
def define_concept(concept):
    global recent_requests

    concept = concept.replace("_", " ").lower()
    dataset = request.form.get('dataset')
    task = request.form.get('task')
    if not dataset or not task:
        return EMPTY_RESPONSE
    
    existing_requests = [r for r in recent_requests if r.is_request(dataset, task, concept)]
    if not existing_requests:
        current_define_request = DefineRequest(dataset, task, concept)
        if len(recent_requests) == MAX_REQUESTS:
            recent_requests = recent_requests[1:]
        recent_requests.append(current_define_request)
        current_define_request.collect_pos_neg_samples("train")
        current_define_request.collect_pos_neg_samples("dev")
    else:
        current_define_request = existing_requests[-1]

    if not current_define_request.train_pos or not current_define_request.dev_pos:
        return json.dumps(dict(error=2))

    random.shuffle(current_define_request.train_pos)
    random.shuffle(current_define_request.train_neg)
    pos_products_to_send = current_define_request.train_pos[:8]
    pos_paths = get_image_paths(pos_products_to_send)
    neg_products_to_send = current_define_request.train_neg[:8]
    neg_paths = get_image_paths(neg_products_to_send)

    return json.dumps(dict(pos=pos_paths, neg=neg_paths))


@app.route('/define/<concept>/train', methods=["POST"])
def train_cavs(concept):
    global recent_requests
    print(request.form)
    concept = concept.replace("_", " ").lower()
    dataset = request.form.get('dataset')
    task = request.form.get('task')
    accuracy_threshold = int(request.form.get('accuracy_threshold')) / 100
    
    if not recent_requests:
        return EMPTY_RESPONSE

    current_define_request = recent_requests[-1]
    if not current_define_request.is_request(dataset, task, concept):
        return EMPTY_RESPONSE

    print(f"Training CAVs for concept {concept}, task {task}, dataset {dataset}")
    concept_data = generate_concept_activation_vector(current_define_request)
    concept_data = check_statistical_significance_for_new_concept(concept, concept_data, accuracy_threshold=accuracy_threshold)
    current_define_request.set_concept_data(concept_data)
    recent_requests[-1] = current_define_request
    
    accuracies = current_define_request.concept_data["ft_vit"]["dev_accuracies"]
    mean_accuracy = round(sum([round(a*100) for a in accuracies]) / len(accuracies), 2)
    accuracies_response = [f"<th scope='col'>{round(a*100)}%</th>" for a in accuracies]
    accuracies_response = "\n".join(accuracies_response)

    atest_color = "darkgreen" if current_define_request.concept_data["gt_than_threshold"] else "darkred"
    atest_pass = "Passed." if current_define_request.concept_data["gt_than_threshold"] else "Failed."
    atest_output = current_define_request.concept_data["gt_than_threshold_output"]
    accuracy_test_reponse = f"<p style='color: {atest_color};'><b>Accuracy Test: {atest_pass}</b><br>{atest_output}</p>"

    stest_color = "darkgreen" if current_define_request.concept_data["gt_than_vit"] else "darkred"
    stest_pass = "Passed." if current_define_request.concept_data["gt_than_vit"] else "Failed."
    stest_output = current_define_request.concept_data["gt_than_vit_output"]
    specificity_test_respose = f"<p style='color: {stest_color};'><b>Task Utility Test: {stest_pass}</b><br>{stest_output}</p>"
    
    return json.dumps(
        dict(
            accuracies=accuracies_response,
            mean_accuracy=mean_accuracy,
            accuray_test=accuracy_test_reponse,
            specificity_test=specificity_test_respose
            )
        )
    
@app.route('/define/<concept>/scatter_plot', methods=["POST"])
def new_scatter_plot(concept):
    global recent_requests
  
    concept = concept.replace("_", " ").lower()
    dataset = request.form.get('dataset')
    task = request.form.get('task')
    
    if not recent_requests:
        return EMPTY_RESPONSE

    current_define_request = recent_requests[-1]
    if not current_define_request.is_request(dataset, task, concept):
        return EMPTY_RESPONSE

    method = "tsne"
    perplexity = 15
    accuracy_threshold = 0.9
    str_acc = str(accuracy_threshold).replace(".", "_")

    print(current_define_request.concept_data)

    path = f"./ui/images/{dataset}/{task}/{concept}_p{perplexity}_accth{str_acc}.png"
    if not os.path.exists(path):
        create_concepts_image(
            path, method, task, dataset,
             accuracy_threshold, perplexity=perplexity,
             new_concept=concept, 
             new_concept_data=current_define_request.concept_data
        )
    
    return json.dumps(dict(img_src=path))

########################################


if __name__ == '__main__':
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = 11234
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)