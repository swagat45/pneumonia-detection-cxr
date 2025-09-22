import argparse, os, torch
import torch.nn as nn
from torchvision import models
from utils import get_loaders, metrics_from_logits, plot_pr, plot_roc, plot_confusion, save_json

@torch.no_grad()
def main(args):
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, classes = get_loaders(args.data_root, batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size, aug=False)

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device); model.eval()

    all_logits, all_labels = [], []
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    import torch as T
    logits = T.cat(all_logits, dim=0).numpy()
    labels = T.cat(all_labels, dim=0).numpy()

    f1, pr_auc, rocauc, probs, preds = metrics_from_logits(logits, labels)
    print({"f1": f1, "pr_auc": pr_auc, "roc_auc": rocauc})

    plot_pr(labels, probs, "outputs/pr_curve_test.png")
    plot_roc(labels, probs, "outputs/roc_curve_test.png")
    plot_confusion(labels, preds, "outputs/confusion_matrix_test.png")
    save_json({"f1": f1, "pr_auc": pr_auc, "roc_auc": rocauc}, "outputs/metrics_test.json")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=2)
    main(ap.parse_args())
