import os
import json

import matplotlib
matplotlib.use("Agg") # headless: no display on the server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # tolerate files that are cut off at the end

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import ViTConfig, ViTForImageClassification

# config
PRETRAINED = True # False = train from scratch
AUGMENT = True
EPOCHS = 10
BATCH_SIZE = 32
LR = 3e-5 if PRETRAINED else 3e-4
IMG_SIZE = 224

ROOT = os.environ.get("DATA_ROOT", "/Volumes/Crucial X9/CS184A/CS184A-Project")
DATA_CSV_LOC = os.path.join(ROOT, "DATA_PP")
OUT_DIR = os.environ.get("OUT_DIR", ".")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 8))
os.makedirs(OUT_DIR, exist_ok=True)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
use_amp = device == "cuda" # bf16 autocast: faster + less memory on L4/A100
print("device:", device, flush=True)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

# data loader
train_df = pd.read_csv(os.path.join(DATA_CSV_LOC, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_CSV_LOC, "val.csv"))
test_df = pd.read_csv(os.path.join(DATA_CSV_LOC, "test.csv"))

classes = sorted(train_df["label"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
print(class_to_idx)
print(train_df["label"].value_counts())

normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize,
])

aug_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    normalize,
])


class CellDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = [os.path.join(ROOT, p) for p in df["filepath"]]
        self.labels = [class_to_idx[l] for l in df["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        for k in range(10):  # try this image, then the next few
            j = (i + k) % len(self.paths)
            try:
                img = Image.open(self.paths[j]).convert("RGB")
                return self.transform(img), self.labels[j]
            except (OSError, ValueError) as e:
                print(f"WARNING: bad image {self.paths[j]}: {e}", flush=True)
        raise RuntimeError(f"Too many consecutive bad images starting at index {i}")


train_set = CellDataset(train_df, aug_transform if AUGMENT else eval_transform)
val_set = CellDataset(val_df, eval_transform)
test_set = CellDataset(test_df, eval_transform)

loader_kwargs = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=(device == "cuda"),
    persistent_workers=NUM_WORKERS > 0,
)
train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_set, **loader_kwargs)
test_loader = DataLoader(test_set, **loader_kwargs)

# model
NUM_CLASSES = len(classes) # 7

if PRETRAINED:
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=NUM_CLASSES)
else:
    config = ViTConfig(image_size=IMG_SIZE, patch_size=16, hidden_size=384, num_hidden_layers=6, num_attention_heads=6, intermediate_size=1536, num_labels=NUM_CLASSES)
    model = ViTForImageClassification(config)

model = model.to(device)

# class weights because of the class imbalance
counts = np.bincount(train_set.labels, minlength=NUM_CLASSES)
counts = np.maximum(counts, 1)  # avoid division by zero
weights = torch.tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float).to(device)

loss_fn = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)


def forward(x):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        logits = model(pixel_values=x).logits
    return logits.float()  # compute the loss in fp32


def evaluate(loader, desc="Eval"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, leave=False, mininterval=30):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = forward(x)
            total_loss += loss_fn(out, y).item() * len(y)
            all_preds += out.argmax(1).cpu().tolist()
            all_labels += y.cpu().tolist()

    return total_loss / len(all_labels), all_preds, all_labels


# training
best_f1 = 0
train_losses, val_losses, val_f1s = [], [], []
ckpt_path = os.path.join(OUT_DIR, "best_vit.pt")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    print(f"\nStarting epoch {epoch+1}/{EPOCHS}", flush=True)
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}", mininterval=30)

    for x, y in progress:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(forward(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        progress.set_postfix(batch_loss=f"{loss.item():.4f}")

    train_loss = total_loss / len(train_set)

    print("Training complete. Starting validation...", flush=True)
    val_loss, preds, labels = evaluate(val_loader, desc="Validation")
    val_f1 = f1_score(labels, preds, average="macro")
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_f1s.append(val_f1)

    print(
        f"Epoch {epoch+1}: "
        f"train loss {train_loss:.4f}, "
        f"val loss {val_loss:.4f}, "
        f"val macro F1 {val_f1:.4f}",
        flush=True,
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved new best model (val macro F1 {best_f1:.4f})", flush=True)

    # save history every epoch so partial runs still leave results behind
    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses, "val_macro_f1": val_f1s}, f, indent=2)

# curve
plt.figure()
plt.plot(train_losses, label="train loss")
plt.plot(val_losses, label="val loss")
plt.plot(val_f1s, label="val macro F1")
plt.xlabel("epoch")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "output_loss.png"), dpi=150)
plt.close()

# final test (only once) - no hyperparam tuning for this.
print("\nEvaluating best checkpoint on the test set...", flush=True)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
test_loss, test_preds, test_labels = evaluate(test_loader, desc="Test")
test_f1 = f1_score(test_labels, test_preds, average="macro")
print(f"Test loss {test_loss:.4f}, test macro F1 {test_f1:.4f}", flush=True)

names = [str(c) for c in classes]
report = classification_report(test_labels, test_preds, target_names=names,
                               labels=list(range(NUM_CLASSES)), digits=4,
                               zero_division=0)
print(report)
with open(os.path.join(OUT_DIR, "test_report.txt"), "w") as f:
    f.write(f"test loss {test_loss:.4f}\ntest macro F1 {test_f1:.4f}\n\n{report}")

cm = confusion_matrix(test_labels, test_preds, labels=list(range(NUM_CLASSES)))
pd.DataFrame(cm, index=names, columns=names).to_csv(
    os.path.join(OUT_DIR, "confusion_matrix.csv"))

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(NUM_CLASSES))
ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_yticks(range(NUM_CLASSES))
ax.set_yticklabels(names)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)

print(f"\nDone. Results are in {OUT_DIR}", flush=True)