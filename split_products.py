import json
import random
from datasets import ClassLabel

from utils import *


def split_products():
    with open("./fashion/products_short_categories.json", "r") as f:
       products = json.load(f) 

    asins = set()
    unique_products = []
    for product in products:
        if product["asin"] not in asins:
            unique_products.append(product)
            asins.add(product["asin"])


    print(f"{len(asins)} unique products")
    random.shuffle(unique_products)

    products_for_embeddings = unique_products[:500000]
    dev_products = unique_products[500000:600000]
    test_products = unique_products[600000:]

    with open("./fashion/train.json", "w") as f:
        json.dump(products_for_embeddings, f)


    with open("./fashion/dev.json", "w") as f:
        json.dump(dev_products, f)


    with open("./fashion/test.json", "w") as f:
        json.dump(test_products, f)


def save_batch(curr_batch, batch_size, imgs, category_labels, price_labels, split):
    batch_idx = [i for i in range(len(imgs))]
    random.shuffle(batch_idx)

    imgs = [imgs[i] for i in batch_idx]
    category_labels = [category_labels[i] for i in batch_idx]
    price_labels = [price_labels[i] for i in batch_idx]
    
    batch_range = f"{curr_batch * batch_size}_{(curr_batch+1)*batch_size}"
    imgs_filename = f"./fashion/embedding_data/{split}_data_files/idx_{batch_range}_imgs.pkl"
    category_labels_filename = f"./fashion/embedding_data/{split}_data_files/idx_{batch_range}_labels_category.pkl"
    price_labels_filename = f"./fashion/embedding_data/{split}_data_files/idx_{batch_range}_labels_price.pkl"
    
    with open(imgs_filename, "wb") as f:
        pickle.dump(imgs, f)
    
    with open(category_labels_filename, "wb") as f:
        pickle.dump(category_labels, f)
    
    with open(price_labels_filename, "wb") as f:
        pickle.dump(price_labels, f)


def get_category_label(product, classlabel, label2id):
    if classlabel:
        label_inds = [classlabel.str2int(c) for c in product["short_categories"]]
        category_label = [0] * len(classlabel.names)
    elif label2id:
        label_inds = [label2id[c] for c in product["short_categories"]]
        category_label = [0] * len(label2id)
    else:
        raise Exception("classlabel or label2id must be provided")

    for ind in label_inds:
        category_label[ind] = 1
    return category_label


def get_price_label(product):
    raw_price = product["price"]
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
        

def save_dataset_files(classlabel=None, label2id=None, split="train"):
    batch_size = 200000
    curr_batch = 0
    imgs = []
    category_labels = []
    price_labels = []
    
    products = load_json(f"./fashion/embedding_data/{split}.json")
    random.shuffle(products)

    for product in tqdm(products):
        asin = product["asin"]
        asin_pref = asin[:2]
        img_files = list(glob(f"./fashion/imgs/{asin_pref}/{asin}/*.jpg"))

        category_label = get_category_label(product, classlabel, label2id)
        price_label = get_price_label(product)

        if price_label > 200:
            continue
        for f in img_files:
            try:
                img = Image.open(f)
                width, height = img.size
                if width < 224 or height < 224:
                    continue
                img = img.resize((224, 224))

            except UnidentifiedImageError:
                continue
            except OSError:
                continue
            
            imgs.append(img)
            category_labels.append(category_label)
            price_labels.append(price_label)

            if len(category_labels) == batch_size:
                save_batch(curr_batch, batch_size, imgs, category_labels, price_labels, split)
                curr_batch += 1
                for image in imgs:
                    image.close()
                imgs, category_labels, price_labels = [], [], []
    
    if imgs:
        save_batch(curr_batch, batch_size, imgs, category_labels, price_labels, split)
        for image in imgs:
            image.close()
