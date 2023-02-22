import json
import torch
import random
import numpy as np

UI_FILES_DIR = "./ui"
PRODUCT_IMAGES_DIR = "/home/daarad/concepts/fashion/imgs/"

SEED = 501
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EMPTY_RESPONSE = json.dumps(dict())

NUM_CLF_FOR_CONCEPT = 10
MAX_POS_SAMPLES = 5000
MAX_NEG_SAMPLES = 10000