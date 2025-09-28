import pandas as pd
import numpy as np
import os

def process():
    data_dir = os.path.join("/data", "cityscapes-image-pairs", "cityscapes_data")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    train_fns = os.listdir(train_dir)
    val_fns = os.listdir(val_dir)
    return data_dir, train_dir, val_dir, train_fns, val_fns