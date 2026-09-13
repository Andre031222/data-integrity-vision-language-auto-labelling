"""AlpacaVision AI -- Grad-CAM visualization for EfficientNet classifiers."""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class GradCAM:
    """Grad-CAM for timm EfficientNet models (hooks last Conv2d layer)."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        if target_layer is None:
            target_layer = self._last_conv(model)

        self._hooks = [
            target_layer.register_forward_hook(self._fwd_hook),
            target_layer.register_backward_hook(self._bwd_hook),
        ]

    @staticmethod
    def _last_conv(model: nn.Module) -> nn.Module:
        last = next((m for m in reversed(list(model.modules())) if isinstance(m, nn.Conv2d)), None)
        if last is None:
            raise ValueError("No Conv2d layer found in model.")
        return last

    def _fwd_hook(self, _mod, _inp, output):
        self._activations = output.detach()

    def _bwd_hook(self, _mod, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        logits = self.model(x)
        cls = target_class if target_class is not None else int(logits.argmax(1))
        self.model.zero_grad()
        logits[0, cls].backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        lo, hi = cam.min(), cam.max()
        return (cam - lo) / (hi - lo + 1e-8)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def generate_gradcam(
    model: nn.Module,
    image: Union[str, Path, np.ndarray, Image.Image],
    target_class: Optional[int] = None,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Grad-CAM overlay."""
    if isinstance(image, (str, Path)):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image[..., ::-1])
    else:
        pil_img = image.convert("RGB")

    orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = orig_bgr.shape[:2]

    gcam = GradCAM(model)
    heatmap = gcam(_TRANSFORM(pil_img).unsqueeze(0), target_class=target_class)
    gcam.remove_hooks()

    heatmap_u8 = (cv2.resize(heatmap, (w, h)) * 255).astype(np.uint8)
    overlay = cv2.addWeighted(orig_bgr, 1 - alpha,
                              cv2.applyColorMap(heatmap_u8, colormap), alpha, 0)
    return overlay, heatmap_u8


def save_gradcam(
    model: nn.Module,
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    target_class: Optional[int] = None,
) -> None:
    overlay, _ = generate_gradcam(model, image_path, target_class=target_class)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
