"""AlpacaVision AI -- Wrapper del clasificador EfficientNet-B0 (timm)."""

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


_TRANSFORM_CENTER = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# TTA: 5 crops (center + 4 corners) x 2 (original + hflip) = 10 augmentations
_TTA_TRANSFORMS = [
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.FiveCrop(224),  # returns tuple of 5 crops
    ]),
]


def _build_model(num_classes: int, arch: str = "efficientnet_b0") -> torch.nn.Module:
    try:
        import timm
    except ImportError:
        raise ImportError("pip install timm")
    return timm.create_model(arch, pretrained=False, num_classes=num_classes)


class AnomalyClassifier:
    """Clasificador binario de anomalias para ojos o extremidades."""

    def __init__(
        self,
        model_path: Union[str, Path],
        task: str,          # 'eyes' | 'legs'
        device: str = "cpu",
        use_tta: bool = True,
        arch: str = "efficientnet_b0",
    ):
        assert task in ("eyes", "legs"), "task debe ser 'eyes' o 'legs'"
        self.task    = task
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_tta = use_tta

        model_path = Path(model_path)
        cn_path = model_path.parent / "class_names.json"
        if cn_path.exists():
            self.class_names = json.loads(cn_path.read_text())
        else:
            self.class_names = ["anomaly", "normal"]

        self.anomaly_idx = self.class_names.index("anomaly") if "anomaly" in self.class_names else 0

        # Auto-detect arch from metadata if available
        meta_path = model_path.parent / "arch.txt"
        if meta_path.exists():
            arch = meta_path.read_text().strip()

        self.model = _build_model(num_classes=len(self.class_names), arch=arch)
        state = torch.load(str(model_path), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _to_pil(self, image) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image[..., ::-1]).convert("RGB")  # BGR->RGB
        return image.convert("RGB")

    def _predict_single(self, img: Image.Image) -> np.ndarray:
        """Retorna probabilidades para una imagen (sin TTA)."""
        tensor = _TRANSFORM_CENTER(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        return probs

    def _predict_tta(self, img: Image.Image) -> np.ndarray:
        """Retorna probabilidades promediadas con TTA (10 augmentaciones)."""
        all_probs = []

        # 1) center crop + hflip
        for t in (_TRANSFORM_CENTER,):
            tensor = t(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                all_probs.append(F.softmax(self.model(tensor), dim=1)[0].cpu().numpy())
            # horizontal flip
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            tensor = t(flipped).unsqueeze(0).to(self.device)
            with torch.no_grad():
                all_probs.append(F.softmax(self.model(tensor), dim=1)[0].cpu().numpy())

        # 2) FiveCrop (4 corners + center) + hflip
        five_crop_t = transforms.Compose([transforms.Resize(256), transforms.FiveCrop(224)])
        to_tensor   = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        for crop in five_crop_t(img):
            tensor = to_tensor(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                all_probs.append(F.softmax(self.model(tensor), dim=1)[0].cpu().numpy())
            flipped = crop.transpose(Image.FLIP_LEFT_RIGHT)
            tensor = to_tensor(flipped).unsqueeze(0).to(self.device)
            with torch.no_grad():
                all_probs.append(F.softmax(self.model(tensor), dim=1)[0].cpu().numpy())

        return np.mean(all_probs, axis=0)  # average over 12 augmentations

    def predict(self, image: Union[str, Path, np.ndarray, Image.Image]) -> dict:
        """Clasifica un recorte de region anatomica."""
        img   = self._to_pil(image)
        probs = self._predict_tta(img) if self.use_tta else self._predict_single(img)

        cls_id     = int(np.argmax(probs))
        is_anomaly = cls_id == self.anomaly_idx
        anomaly_p  = float(probs[self.anomaly_idx])

        return {
            "is_anomaly":    is_anomaly,
            "anomaly_prob":  anomaly_p,
            "class_id":      cls_id,
            "class_name":    self.class_names[cls_id],
            "confidence":    float(probs[cls_id]),
            "probabilities": {name: float(p) for name, p in zip(self.class_names, probs)},
        }
