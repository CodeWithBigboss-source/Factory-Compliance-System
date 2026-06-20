"""
model_loader.py
----------------
Loads the already-trained ResNet18 binary classifier (model.pth) and exposes
a single predict_frame() function used by the detection engine.

IMPORTANT: This module does NOT train or fine-tune anything. It assumes
model.pth was saved either as:
    (a) a full model object (torch.save(model, path)), or
    (b) a state_dict (torch.save(model.state_dict(), path))
and handles both cases automatically.

Classes: ['safe', 'unsafe']  (binary, as trained)
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

CLASS_NAMES = ["safe", "unsafe"]

# Standard ImageNet normalization — matches typical ResNet18 transfer-learning
# preprocessing. If your training script used different mean/std, update here
# to match exactly, otherwise inference accuracy will silently degrade.
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_resnet18_binary():
    """Recreate the ResNet18 architecture with a 2-class output head."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))
    return model


def load_model(model_path, device="cpu"):
    """
    Load model.pth robustly. Handles both:
      - Saved full model: torch.save(model, "model.pth")
      - Saved state_dict:  torch.save(model.state_dict(), "model.pth")
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model.pth not found at {model_path}. Place your trained model "
            f"file at this exact path before running detection."
        )

    loaded = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(loaded, nn.Module):
        model = loaded
    else:
        # It's a state_dict (an OrderedDict of tensors)
        model = _build_resnet18_binary()
        model.load_state_dict(loaded)

    model.to(device)
    model.eval()
    return model


class SafeUnsafeClassifier:
    """Thin wrapper used by the detection engine for per-frame inference."""

    def __init__(self, model_path="models/model.pth", device="cpu"):
        self.device = device
        self.model = load_model(model_path, device=device)

    @torch.no_grad()
    def predict_frame(self, pil_image: Image.Image):
        """
        Args:
            pil_image: a PIL.Image (RGB) of a single video frame.
        Returns:
            dict: {"label": "safe"|"unsafe", "confidence": float, "probs": {"safe": float, "unsafe": float}}
        """
        tensor = PREPROCESS(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        label_idx = int(torch.argmax(probs).item())
        return {
            "label": CLASS_NAMES[label_idx],
            "confidence": float(probs[label_idx].item()),
            "probs": {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))},
        }


if __name__ == "__main__":
    # Quick smoke test — run this directly to confirm model.pth loads correctly
    # before wiring it into the full pipeline.
    import sys
    clf = SafeUnsafeClassifier(model_path="models/model.pth")
    print("✅ Model loaded successfully.")
    print(f"Architecture: {clf.model.__class__.__name__}")
    print(f"Classes: {CLASS_NAMES}")

    if len(sys.argv) > 1:
        test_img = Image.open(sys.argv[1])
        result = clf.predict_frame(test_img)
        print(f"\nTest prediction on {sys.argv[1]}:")
        print(result)