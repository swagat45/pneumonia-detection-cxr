import os, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

def get_loaders(data_root, batch_size=32, num_workers=2, img_size=224, aug=True):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip() if aug else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(10) if aug else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        norm
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_root, "val"), transform=test_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_root, "test"), transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.classes

def class_weights_from_loader(loader, device="cpu"):
    counts = np.zeros(2, dtype=np.int64)
    for _, y in loader:
        if isinstance(y, torch.Tensor):
            y_np = y.numpy()
        else:
            y_np = np.array(y)
        for c in range(2):
            counts[c] += (y_np == c).sum()
    total = counts.sum()
    weights = total / (2.0 * counts + 1e-9)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def metrics_from_logits(logits, labels):
    import numpy as np, torch
    if logits.shape[1] == 2:
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:,1]
    else:
        probs = torch.tensor(logits).numpy()
    y = np.array(labels)
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(y, preds)
    pr_auc = average_precision_score(y, probs)
    rocauc = roc_auc_score(y, probs)
    return f1, pr_auc, rocauc, probs, preds

def plot_pr(y, scores, out_path):
    p, r, _ = precision_recall_curve(y, scores)
    plt.figure(); plt.step(r, p, where="post")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curve")
    plt.grid(True, alpha=0.3); plt.savefig(out_path, bbox_inches="tight"); plt.close()

def plot_roc(y, scores, out_path):
    fpr, tpr, _ = roc_curve(y, scores)
    plt.figure(); plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.grid(True, alpha=0.3); plt.savefig(out_path, bbox_inches="tight"); plt.close()

def plot_confusion(y, preds, out_path):
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(); im = ax.imshow(cm)
    import numpy as np
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha='center', va='center')
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); plt.title("Confusion Matrix")
    plt.colorbar(im); plt.savefig(out_path, bbox_inches="tight"); plt.close()

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
