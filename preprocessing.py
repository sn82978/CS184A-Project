'''
Preprocess the data such that the 3 classes that regard "other" are dropped and we combine 
the normal cells into one category.

We also perform the split into 70% training, 15% test, 15% validation so that it is the same.
'''

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Relative paths starting inside the CS184A-Project repository
DATA_PATH = "DATA/bone_marrow_cell_dataset"
DEST_PATH = "DATA_PP"

## Other:
# ART = Artefact, NIF = Not identifiable, OTH = other cell
OTHER_CATEGORIES = ["ART", "NIF", "OTH"] 

## Normal:
# BAS (Basophil), EBO (Erythroblast), EOS (Eosinophil), LYT (Lymphocyte), MMZ (Metamyelocyte), MON (Monocyte), 
# MYB (Myelocyte), NGB (Band neutrophil), NGS (Segmented neutrophil), PEB (Proerythroblast), PLM (Plasma cell), 
# PMO (Promyelocyte)
NORMAL_CATEGORIES = ["BAS", "EBO", "EOS", "LYT", "MMZ", "MON", "MYB", "NGB", "NGS", "PEB", "PLM", "PMO"]

# We leave the abnormal categories as they are

records = []

for subdir in os.listdir(DATA_PATH):
    src_dir = os.path.join(DATA_PATH, subdir)
    if not os.path.isdir(src_dir) or subdir in OTHER_CATEGORIES:
        continue

    label = "normal" if subdir in NORMAL_CATEGORIES else subdir
    dest_dir = os.path.join(DEST_PATH, label)
    os.makedirs(dest_dir, exist_ok=True)

    ## Walks through any subdirs within our classes and copies them to avoid irregularities in dataset structure.
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                src_file = os.path.join(root, fname)
                dst_file = os.path.join(dest_dir, fname)
                shutil.copy2(src_file, dst_file)
                # norm to forward slashes so CSV paths look like "DATA_PP/normal/img.jpg"
                rel_filepath = dst_file.replace("\\", "/")
                records.append({"filepath": rel_filepath, "label": label})

## The below code is what performs the split. I already did it, so do not run it again, it will overwrite the original split.
# Just use the .csv files in DATA_PP, which already have the information on which patients' images ended up in which sets.
# We were not able to split by patient since there is no patient data or IDs or anything like that to split upon.
# That means that our metrics might be better than they actually are since patients' data could be spread among the training,
# testing, and validation sets.

# df = pd.DataFrame(records)

# train_val_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
# train_df, val_df = train_test_split(train_val_df, test_size=0.15 / 0.85, stratify=train_val_df["label"], random_state=42)

# train_df.to_csv(os.path.join(DEST_PATH, "train.csv"), index=False)
# val_df.to_csv(os.path.join(DEST_PATH, "val.csv"), index=False)
# test_df.to_csv(os.path.join(DEST_PATH, "test.csv"), index=False)