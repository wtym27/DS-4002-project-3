#!/usr/bin/env python
# coding: utf-8

# In[4]:

"""The purpose of this program is to train a Resnet50 CNN on the images of wildlife, and to evaluate the output.
It is recommended to first move this program into the parent folder, DS-4002-project-3, such that data is a child folder, and to run it as a slurm job. """

#Import statements
import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Variables
CSV_PATH = 'data/nacti_metadata.csv'
DATA_DIR = 'data/'
TARGET_SIZE = (336, 448)
TRAIN_SPLIT_RATIO = 0.8

# Data Augmentation for training
train_transforms = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Standard transforms for testing (no augmentation)
test_transforms = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_labels(csv_path):
    #Loads data into a dictionary for fast labelling.
    df = pd.read_csv(csv_path, low_memory=False)
    return dict(zip(df['filename'], df['common_name']))

class StreamingCameraTrapDataset(Dataset):
    #Stores only file paths in RAM. Streams and processes images on-the-fly.
    def __init__(self, filepaths, labels, label_to_idx, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
        # Encode string labels to integers based on the master mapping
        self.encoded_labels = [label_to_idx[lbl] for lbl in self.labels]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Image is only opened when requested by the DataLoader
        img_path = self.filepaths[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
        except Exception as e:
            # Fallback to a black image tensor if file is corrupt
            img = torch.zeros((3, TARGET_SIZE[0], TARGET_SIZE[1]))

        label = self.encoded_labels[idx]
        return img, torch.tensor(label, dtype=torch.long)

class NACTI_ResNet50(nn.Module):
    # Defines resnet 50 cnn
    def __init__(self, num_classes):
        super(NACTI_ResNet50, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
        #defines the CNN layers
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    # 1. Parse Directory and Gather Data
    label_dict = load_labels(CSV_PATH)
    all_filepaths = []
    all_labels = []

    print("Scanning directories for images...")
    for i in range(0, 4):
        part_folder = f"part{i}"
        part_path = os.path.join(DATA_DIR, part_folder)

        if not os.path.exists(part_path):
            continue

        for root, _, files in os.walk(part_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, DATA_DIR).replace('\\', '/')

                    label = label_dict.get(rel_path, "unlabeled")
                    all_filepaths.append(full_path)
                    all_labels.append(label)

    if len(all_filepaths) == 0:
        print("No images found. Check your DATA_DIR path.")
        exit()

    # 2. Create Train/Test Split and Mapping
    unique_labels = sorted(list(set(all_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Found {len(all_filepaths)} images across {num_classes} classes.")

    # Shuffle and split data
    combined = list(zip(all_filepaths, all_labels))
    random.seed(42)  # For reproducibility
    random.shuffle(combined)

    split_idx = int(len(combined) * TRAIN_SPLIT_RATIO)
    train_data = combined[:split_idx]
    test_data = combined[split_idx:]

    train_paths = [x[0] for x in train_data]
    train_labels = [x[1] for x in train_data]
    test_paths = [x[0] for x in test_data]
    test_labels = [x[1] for x in test_data]

    print(f"Training samples: {len(train_paths)} | Testing samples: {len(test_paths)}")

    # 3. Initialize Datasets and Dataloaders
    train_dataset = StreamingCameraTrapDataset(train_paths, train_labels, label_to_idx, transform=train_transforms)
    test_dataset = StreamingCameraTrapDataset(test_paths, test_labels, label_to_idx, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

    # 4. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = NACTI_ResNet50(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') 

    epochs = 5
    results_log = []

    # 5. Training and Testing Loop
    for epoch in range(epochs):
        #Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        # Keep track of time
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
        for inputs, labels in pbar_train:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            current_train_loss = train_loss / (train_total / inputs.size(0))
            current_train_acc = 100 * train_correct / train_total
            pbar_train.set_postfix({"Loss": f"{current_train_loss:.4f}", "Acc": f"{current_train_acc:.2f}%"})

        #Testing
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            pbar_test = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", unit="batch")
            for inputs, labels in pbar_test:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                current_test_loss = test_loss / (test_total / inputs.size(0))
                current_test_acc = 100 * test_correct / test_total
                pbar_test.set_postfix({"Loss": f"{current_test_loss:.4f}", "Acc": f"{current_test_acc:.2f}%"})

        #Logging
        log_entry = (f"Epoch {epoch+1} | "
                     f"Train Loss: {current_train_loss:.4f} | Train Acc: {current_train_acc:.2f}% | "
                     f"Test Loss: {current_test_loss:.4f} | Test Acc: {current_test_acc:.2f}%")

        print(f"\n{log_entry}\n")
        results_log.append(log_entry)

    # 6. Save Results to TXT
    with open("output/output_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Training and Testing Results\n")
        f.write("============================\n")
        for log in results_log:
            f.write(log + "\n")
        print("Results successfully saved to output_metrics.txt")


# In[ ]:




