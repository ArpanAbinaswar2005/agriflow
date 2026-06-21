import os
import sys
import random
from datetime import datetime

import torch
import torchvision.models as models
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

# 1. Add /content/agriflow to sys.path
sys.path.insert(0, "/content/agriflow")

# 2. Imports done above

# 3. Set device to cuda when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 4. Load ImageFolder from /content/freshness44 with required transforms
data_dir = "/content/freshness44"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = ImageFolder(root=data_dir, transform=transform)
print(f"Full dataset size: {len(full_dataset)} classes: {full_dataset.classes}")

# 5. Take 25% random subset
total_n = len(full_dataset)
if total_n == 0:
    raise SystemExit("Dataset at /content/freshness44 is empty or not found.")
subset_n = max(1, int(total_n * 0.25))
indices = list(range(total_n))
random.seed(42)
random.shuffle(indices)
subset_indices = indices[:subset_n]
subset = Subset(full_dataset, subset_indices)
print(f"Subset size (25%): {len(subset)}")

# 6. Split 70/15/15 into train/val/test
train_n = int(len(subset) * 0.70)
val_n = int(len(subset) * 0.15)
test_n = len(subset) - train_n - val_n
if train_n <= 0 or val_n < 0 or test_n < 0:
    raise SystemExit("Not enough samples in subset to split into train/val/test.")
train_set, val_set, test_set = random_split(subset, [train_n, val_n, test_n], generator=torch.Generator().manual_seed(42))

batch_size = 32
num_workers = 4
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# 7. Build MobileNetV3-Large with DEFAULT weights and replace classifier
try:
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
except Exception:
    weights = None
model = models.mobilenet_v3_large(weights=weights)

# determine num_classes
if hasattr(full_dataset, "classes"):
    num_classes = len(full_dataset.classes)
else:
    num_classes = 2

# replace last classifier layer
if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
    try:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    except Exception:
        # fallback: replace whole classifier
        model.classifier = nn.Sequential(nn.Linear(model.classifier[-1].in_features, num_classes))
else:
    # generic fallback
    model.fc = nn.Linear(getattr(model, "fc", nn.Linear(1280, num_classes)).in_features, num_classes)

model = model.to(device)

# 8. Train 15 epochs with Adam and StepLR
num_epochs = 15
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_val_acc = 0.0
save_path = "/content/drive/MyDrive/agriflow/models/stage1_real.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    running_corrects = 0
    running_total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()
        running_total += labels.size(0)

    train_acc = running_corrects / running_total if running_total > 0 else 0.0

    # Validation
    model.eval()
    val_corrects = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            val_corrects += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_corrects / val_total if val_total > 0 else 0.0

    # step scheduler
    scheduler.step()

    # 9. Print train and val accuracy every 3 epochs
    if (epoch + 1) % 3 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    # 10. Save best val accuracy model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        try:
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} at epoch {epoch+1} with val acc {best_val_acc:.4f}")
        except Exception as e:
            print(f"Failed to save model: {e}")

# 11. Print final best val accuracy
print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
