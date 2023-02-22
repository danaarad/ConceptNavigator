import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

from datasets import ClassLabel

from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
from transformers import TrainingArguments, Trainer

from utils import *


OUTPUT_PREFIX = "./fashion/category_prediction/"
BATCH_SIZE = 32


class CategoryPredictionDataset(Dataset):
    def __init__(self, file_batch_size=500000, lenght=None, transform=None, split=None):
        self.min_idx = 0
        self.max_idx = self.min_idx + file_batch_size
        self.file_batch_size = file_batch_size
        self.len = lenght
        self._split = split
        self.transform = transform

        self.imgs, self.labels = self._load_pickles(split)
        
    def _load_pickles(self, split):
        with open(f"./fashion/embedding_data/{self._split}_data_files/idx_{self.min_idx}_{self.max_idx}_imgs.pkl", "rb") as f:
            imgs = pickle.load(f)
        with open(f"./fashion/embedding_data/{self._split}_data_files/idx_{self.min_idx}_{self.max_idx}_labels.pkl", "rb") as f:
            labels = pickle.load(f)
        return imgs, labels

    def __len__(self):
        if not self.len:
            raise Exception("len is not set!")
        return self.len

    def __getitem__(self, idx):
        if idx < self.min_idx or idx >= self.max_idx:
            new_batch_idx = idx // self.file_batch_size
            self.min_idx = new_batch_idx * self.file_batch_size
            self.max_idx = self.min_idx + self.file_batch_size
            
            self.imgs = pickle.load(open(f"./fashion/embedding_data/{self._split}_data_files/idx_{self.min_idx}_{self.max_idx}_imgs.pkl", "rb"))
            self.labels = pickle.load(open(f"./fashion/embedding_data/{self._split}_data_files/idx_{self.min_idx}_{self.max_idx}_labels.pkl", "rb"))

        idx = idx % self.file_batch_size
        sample = dict(img=self.imgs[idx], label=self.labels[idx]) 
        sample = self.transform(sample)
        return sample


def collate_fn(samples):
    pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
    labels = torch.tensor([sample["label"] for sample in samples]).float()
    return {"pixel_values": pixel_values, "labels": labels}
    

def compute_metrics(eval_pred):
    y_pred, y_true = eval_pred
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    
    y_pred = y_pred.sigmoid() > 0.5
    
    accuracy_thresh = (y_pred==y_true.bool()).float().mean().item()
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true, y_pred, average="macro")
    
    return dict(
        accuracy_thresh=accuracy_thresh,
        precision=precision,
        recall=recall,
        fbeta_score=fbeta_score,
        support=support
    )


def seed_worker(_):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).
    Args:
        seed (`int`): The seed to set.
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        # train_sampler = self._get_train_sampler()
        train_sampler = range(len(train_dataset))

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
            # shuffle=False
        )

def main():
    train_products = load_json("./fashion/embedding_data/train.json")
    dev_products = load_json("./fashion/embedding_data/dev.json")
    
    categories = set.union(*[set(p["short_categories"]) for p in train_products])
    print(f"{len(categories)} categories")

    classlabel = ClassLabel(names=categories) 
    id2label = {id:label for id, label in enumerate(classlabel.names)}
    label2id = {label:id for id,label in id2label.items()}

    # train_dataset = save_category_prediction_dataset_files(train_products, classlabel=classlabel, split="train")
    # dev_dataset = save_category_prediction_dataset_files(dev_products, classlabel=classlabel, split="dev")
    
    print("torch.cuda.is_available(): ", torch.cuda.is_available())

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    def _train_transform(sample):
        transform = Compose(
                [
                    RandomResizedCrop(feature_extractor.size),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
                ]
            )
        sample['pixel_values'] = transform(sample['img'].convert("RGB"))
        return sample


    def _dev_transform(sample):
        transform = Compose(
                [
                    Resize(feature_extractor.size),
                    CenterCrop(feature_extractor.size),
                    ToTensor(),
                    Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
                ]
            )
        sample['pixel_values'] = transform(sample['img'].convert("RGB"))
        return sample
    
    train_dataset = CategoryPredictionDataset(split="train", lenght=2198037, transform=_train_transform)
    dev_dataset = CategoryPredictionDataset(split="dev", lenght=440177, transform=_dev_transform)

    print(f"{len(train_dataset)} train imgs, {len(dev_dataset)} dev imgs")

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  num_labels=len(id2label),
                                                  id2label=id2label,
                                                  label2id=label2id,
                                                  force_download=True,
                                                  problem_type="multi_label_classification")

    args = TrainingArguments(
        OUTPUT_PREFIX,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir='logs',
        remove_unused_columns=False,
    )

    trainer = MultilabelTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )
    print(f"device: {args.device}")
    trainer.train()




if __name__ == "__main__":
    main()

