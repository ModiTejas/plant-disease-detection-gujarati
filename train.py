import os
import sys
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# ---------------- CONFIGURATION ----------------
DATASET_DIR = Path("Plant Disease Dataset")
MODEL_SAVE_DIR = Path("models")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
DICT_FILE = Path("disease_dictionary.json")

BATCH_SIZE = 32
IMAGE_SIZE = 224
STAGE1_EPOCHS = 5      # Warmup classifier head with frozen backbone & frozen BatchNorm
STAGE2_MAX_EPOCHS = 15 # Fine-tune entire model with early stopping
EARLY_STOP_PATIENCE = 5

STAGE1_LR = 1e-3
STAGE2_LR = 1e-4

# ---------------- DATASET UTILS ----------------
class PlantLeafDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def scan_dataset(root_dir: Path):
    classes = []
    for crop_dir in sorted(root_dir.iterdir()):
        if crop_dir.is_dir() and not crop_dir.name.startswith('.'):
            for disease_dir in sorted(crop_dir.iterdir()):
                if disease_dir.is_dir() and not disease_dir.name.startswith('.'):
                    classes.append(f"{crop_dir.name}___{disease_dir.name}")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG'}

    image_paths = []
    labels = []
    for crop_dir in sorted(root_dir.iterdir()):
        if crop_dir.is_dir() and not crop_dir.name.startswith('.'):
            for disease_dir in sorted(crop_dir.iterdir()):
                if disease_dir.is_dir() and not disease_dir.name.startswith('.'):
                    cls_name = f"{crop_dir.name}___{disease_dir.name}"
                    cls_idx = class_to_idx[cls_name]
                    for img in disease_dir.rglob("*"):
                        if img.suffix in valid_exts:
                            image_paths.append(str(img))
                            labels.append(cls_idx)

    return classes, class_to_idx, image_paths, labels

# ---------------- EVALUATION HELPER ----------------
def evaluate(model, data_loader, criterion, device, scaler=None):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    total_samples = len(all_targets)
    avg_loss = running_loss / total_samples
    acc = accuracy_score(all_targets, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )

    return avg_loss, acc, precision * 100, recall * 100, f1 * 100, all_targets, all_preds

# ---------------- MAIN TRAINING ROUTINE ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"  TRAINING PLATFORM: {device}")
    if device.type == "cuda":
        print(f"  GPU MODEL:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM CAPACITY:    {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 70)

    # 1. Dataset Scan
    classes, class_to_idx, image_paths, labels = scan_dataset(DATASET_DIR)
    num_classes = len(classes)
    total_images = len(image_paths)

    print(f"\n[1/6] Dataset Discovery:")
    print(f"  Total Images Found: {total_images}")
    print(f"  Total Classes:      {num_classes}")

    if num_classes != 37:
        raise RuntimeError(f"Expected 37 classes, but found {num_classes} in dataset!")

    # Verify dictionary
    if DICT_FILE.exists():
        with open(DICT_FILE, "r", encoding="utf-8") as f:
            dict_data = json.load(f)
        dict_classes = set(dict_data.get("classes", dict_data).keys())
        missing = set(classes) - dict_classes
        if missing:
            print(f"  [WARNING] Missing dictionary mappings: {missing}")
        else:
            print("  [SUCCESS] All 37 classes match disease_dictionary.json!")

    # Save class mapping
    with open(MODEL_SAVE_DIR / "class_indices.json", "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "classes": classes}, f, indent=4)

    # 2. Stratified 70 / 15 / 15 Split
    print("\n[2/6] Performing Stratified Split (70% Train | 15% Val | 15% Test)...")
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.30, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )

    print(f"  Training Set:   {len(train_paths):>6} images ({len(train_paths)/total_images*100:.1f}%)")
    print(f"  Validation Set: {len(val_paths):>6} images ({len(val_paths)/total_images*100:.1f}%)")
    print(f"  Test Set:       {len(test_paths):>6} images ({len(test_paths)/total_images*100:.1f}%)")

    # 3. Transforms & DataLoaders
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = PlantLeafDataset(train_paths, train_labels, train_transform)
    val_ds = PlantLeafDataset(val_paths, val_labels, eval_transform)
    test_ds = PlantLeafDataset(test_paths, test_labels, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 4. Model Setup
    print("\n[3/6] Initializing MobileNetV3-Large (Pretrained)...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # =========================================================================
    # STAGE 1: Train Classifier Head Only (Backbone + BatchNorm FROZEN)
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"  STAGE 1: Training Classifier Head ({STAGE1_EPOCHS} Epochs, Backbone + BatchNorm Frozen)")
    print("=" * 70)

    for param in model.features.parameters():
        param.requires_grad = False

    optimizer_stage1 = torch.optim.AdamW(model.classifier.parameters(), lr=STAGE1_LR, weight_decay=1e-4)

    best_val_f1 = 0.0

    for epoch in range(1, STAGE1_EPOCHS + 1):
        t0 = time.time()
        
        # Keep backbone (including BatchNorm) in eval mode, classifier in train mode
        model.features.eval()
        model.classifier.train()
        
        train_loss = 0.0

        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer_stage1.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer_stage1)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer_stage1.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion, device, scaler)
        elapsed = time.time() - t0

        print(f"Stage 1 - Epoch [{epoch:02d}/{STAGE1_EPOCHS:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Val Macro F1: {val_f1:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "stage": 1,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "class_to_idx": class_to_idx,
            }, MODEL_SAVE_DIR / "best_model.pth")

    # =========================================================================
    # STAGE 2: Fine-Tuning Entire Model (with Early Stopping)
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"  STAGE 2: Fine-Tuning Full Model (Max {STAGE2_MAX_EPOCHS} Epochs, Early Stopping Patience={EARLY_STOP_PATIENCE})")
    print("=" * 70)

    # Unfreeze all backbone layers and set entire model to train mode
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer_stage2 = torch.optim.AdamW(model.parameters(), lr=STAGE2_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=STAGE2_MAX_EPOCHS)

    patience_counter = 0

    for epoch in range(1, STAGE2_MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer_stage2.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer_stage2)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer_stage2.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion, device, scaler)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Stage 2 - Epoch [{epoch:02d}/{STAGE2_MAX_EPOCHS:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Val Macro F1: {val_f1:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "stage": 2,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "class_to_idx": class_to_idx,
            }, MODEL_SAVE_DIR / "best_model.pth")
            print(f"  >>> [SAVED] New Best Val Macro F1: {val_f1:.2f}% (Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  --- No improvement for {patience_counter}/{EARLY_STOP_PATIENCE} epochs")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n[!] Early stopping triggered at Stage 2 Epoch {epoch}.")
                break

    # =========================================================================
    # UNBIASED TEST EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION ON INDEPENDENT TEST SET (15%)")
    print("=" * 70)

    best_ckpt = torch.load(MODEL_SAVE_DIR / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc, test_prec, test_rec, test_f1, test_targets, test_preds = evaluate(
        model, test_loader, criterion, device, scaler
    )

    print(f"\n  Independent Test Set Results:")
    print(f"  -----------------------------")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_acc:.2f}%")
    print(f"  Macro F1:  {test_f1:.2f}%")
    print(f"  Precision: {test_prec:.2f}%")
    print(f"  Recall:    {test_rec:.2f}%")

    print("\n" + "=" * 70)
    print("  PER-CLASS DETAILED CLASSIFICATION REPORT")
    print("=" * 70)
    report_text = classification_report(test_targets, test_preds, target_names=classes, digits=2)
    print(report_text)

    # Save test report to JSON
    report_dict = classification_report(test_targets, test_preds, target_names=classes, output_dict=True)
    report_dict["summary"] = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "best_val_macro_f1": best_ckpt["val_macro_f1"],
        "best_val_accuracy": best_ckpt["val_acc"]
    }
    with open(MODEL_SAVE_DIR / "test_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    print(f"Report saved to: {MODEL_SAVE_DIR / 'test_evaluation_report.json'}")
    print("=" * 70)

if __name__ == '__main__':
    main()