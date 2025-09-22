import argparse, os, torch, numpy as np, matplotlib.pyplot as plt
from torchvision import models, datasets, transforms
import torch.nn.functional as F

def overlay(img_t, cam):
    img = img_t.permute(1,2,0).cpu().numpy()
    heat = cam.squeeze(0).cpu().numpy()
    heat_rgb = np.stack([heat, np.zeros_like(heat), np.zeros_like(heat)], axis=-1)
    out = 0.6*img + 0.4*heat_rgb
    out = np.clip(out, 0, 1)
    return out

def gradcam_from_block(model, x, block):
    feats = []
    def hook(m, i, o): feats.append(o)
    h = block.register_forward_hook(hook)
    logits = model(x)
    cls = logits.argmax(1)
    score = logits[torch.arange(logits.size(0)), cls]
    model.zero_grad(); score.sum().backward()
    fmap = feats[0]  # [B,C,H,W]
    grad = fmap.grad if hasattr(fmap, "grad") and fmap.grad is not None else torch.ones_like(fmap)
    weights = grad.mean(dim=(2,3), keepdim=True)
    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam_min = cam.view(cam.size(0), -1).min(dim=1, keepdim=True)[0].view(-1,1,1,1)
    cam_max = cam.view(cam.size(0), -1).max(dim=1, keepdim=True)[0].view(-1,1,1,1)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    h.remove()
    return cam

def main(args):
    os.makedirs("outputs/gradcam", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ds = datasets.ImageFolder(os.path.join(args.data_root, "test"), transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device); model.eval()

    block = model.layer4[-1]  # last block
    count = 0
    for x, y in loader:
        x = x.to(device)
        cam = gradcam_from_block(model, x, block)
        vis = overlay(x[0].cpu(), cam[0].cpu())
        import matplotlib.pyplot as plt
        plt.figure(); plt.imshow(vis); plt.title("Grad-CAM"); plt.axis("off")
        out_path = os.path.join("outputs/gradcam", f"gradcam_{count}.png")
        plt.savefig(out_path, bbox_inches="tight"); plt.close()
        count += 1
        if count >= args.n: break
    print(f"Saved {count} Grad-CAM images to outputs/gradcam/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--img_size", type=int, default=224)
    main(ap.parse_args())
