import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)


from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
from transformers import TrainingArguments, Trainer

from sklearn.metrics import mean_absolute_error
from utils import *


OUTPUT_PREFIX = "./fashion/price_prediction/"
MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 64
        

class PricePredictionDataset(Dataset):
    def __init__(self, file_batch_size=200000, lenght=None, transform=None, split=None):
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
        with open(f"./fashion/embedding_data/{self._split}_data_files/idx_{self.min_idx}_{self.max_idx}_labels_price.pkl", "rb") as f:
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
            with open(f"./fashion/embedding_data/{self._split}_data_files/idx_{self.min_idx}_{self.max_idx}_imgs.pkl", "rb") as f:
                self.imgs = pickle.load(f)
            with open(f"./fashion/embedding_data/{self._split}_data_files/idx_{self.min_idx}_{self.max_idx}_labels_price.pkl", "rb") as f:
                self.labels = pickle.load(f)
                
        idx = idx % self.file_batch_size
        sample = dict(img=self.imgs[idx], label=self.labels[idx]) 
        sample = self.transform(sample)
        return sample



def get_price_label(raw_price):
    # raw_price = product["price"]
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
        

def collate_fn(samples):
    pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
    labels = torch.tensor([get_price_label(sample["label"]) for sample in samples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    y_pred, y_true = eval_pred
    print(y_pred.shape, y_true.shape)
    
    y_pred = torch.from_numpy(y_pred.squeeze())
    y_true = torch.from_numpy(y_true)
    
    y_acc_10 = torch.abs(torch.subtract(y_pred, y_true)) <= (y_true * 0.1)
    y_acc_20 = torch.abs(torch.subtract(y_pred, y_true)) <= (y_true * 0.2)
    y_acc_50 = torch.abs(torch.subtract(y_pred, y_true)) <= (y_true * 0.5)
    accuracy_10 = (y_acc_10.bool()).float().mean().item()
    accuracy_20 = (y_acc_20.bool()).float().mean().item()
    accuracy_50 = (y_acc_50.bool()).float().mean().item()
    
    mae = mean_absolute_error(y_true, y_pred)

    return dict(
        accuracy_10=accuracy_10,
        accuracy_20=accuracy_20,
        accuracy_50=accuracy_50,
        mae=mae
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


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = torch.squeeze(outputs.logits)
        loss = torch.nn.functional.mse_loss(logits, labels)
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
    # train_products = load_json("./fashion/embedding_data/train.json")
    # dev_products = load_json("./fashion/embedding_data/dev.json")
    
    print("torch.cuda.is_available(): ", torch.cuda.is_available())

    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
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
    
    train_dataset = PricePredictionDataset(split="train", lenght=2117418, transform=_train_transform)
    dev_dataset = PricePredictionDataset(split="dev", lenght=100000, transform=_dev_transform)

    print(f"{len(train_dataset)} train imgs, {len(dev_dataset)} dev imgs")
    
    model = ViTForImageClassification.from_pretrained(MODEL_CHECKPOINT,
                                                  num_labels=1,
                                                  force_download=True
                                                  )



    args = TrainingArguments(
        OUTPUT_PREFIX,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=50,
        weight_decay=0.01,
        logging_dir='logs',
        remove_unused_columns=False,
        gradient_accumulation_steps=4
    )

    trainer = RegressionTrainer(
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

