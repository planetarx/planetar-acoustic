"""SAI → image-tensor → CV classifier.

The SAI is a 2-D map (cochlear-channel × time-lag) which already looks like
an image. We log-compress, normalise, and tile it into a 3-channel input
that any vanilla ImageNet-pretrained backbone (ResNet, ViT, EfficientNet)
can ingest without retraining the stem.

Three backends:

  * ``mock``   — deterministic hash of the SAI image. No torch needed.
                 Lets the rest of the pipeline run end-to-end in tests / CI.
                 The output explicitly carries ``model_id="mock"`` so nobody
                 mistakes this for a trained model.
  * ``torch``  — load a torchvision / timm model checkpoint. Optional dep.
  * ``onnx``   — load an exported model via onnxruntime. Optional dep.

The intended training corpus is DeepShip + ShipsEar with optional WAB
mammal clips as background-class negatives, but training lives in a
separate notebook / repo; this module only does inference.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from planetar_acoustic.sai.sai import SAI

log = logging.getLogger(__name__)


DEFAULT_CLASSES: tuple[str, ...] = (
    "cargo",
    "passenger",
    "tanker",
    "tug",
    "fishing",
    "background",
)


@dataclass(frozen=True)
class Hypothesis:
    class_: str            # most-likely label
    score: float           # softmax probability of class_
    top_k: list[tuple[str, float]]
    model_id: str          # "mock" | "torch:<sha>" | "onnx:<sha>"
    embedding: np.ndarray | None  # pre-logit features, shape (D,) float32


def sai_to_image(sai: SAI, *, target_hw: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Convert a SAI to a 3-channel HWC float32 image in [0, 1].

    Channels carry mildly different views so the CNN's first conv has more to
    chew on than a grey replication:
        R = log-compressed SAI
        G = per-channel z-normalised SAI
        B = horizontal-gradient magnitude (lag-direction edges)
    """
    img = sai.image.astype(np.float32, copy=False)
    # Log-compression: signal dynamic range is enormous.
    log_img = np.log1p(np.maximum(img, 0.0))
    log_img -= log_img.min()
    if log_img.max() > 0:
        log_img /= log_img.max()

    z = img - img.mean(axis=1, keepdims=True)
    z_std = z.std(axis=1, keepdims=True) + 1e-6
    z = z / z_std
    # Squash to [0, 1] via logistic — robust to outliers.
    z = 1.0 / (1.0 + np.exp(-z))

    # Lag-direction edges (|∂/∂lag|), normalised.
    grad = np.abs(np.diff(log_img, axis=1, prepend=log_img[:, :1]))
    if grad.max() > 0:
        grad = grad / grad.max()

    stacked = np.stack([log_img, z, grad], axis=-1)  # (C_ch, lag, 3)

    # Resize to target. We use simple area-averaging via numpy; this stays
    # pure-numpy so we don't drag in PIL just for inference.
    return _resize_hwc(stacked, target_hw[0], target_hw[1])


def _resize_hwc(img: np.ndarray, h_out: int, w_out: int) -> np.ndarray:
    """Pure-numpy bilinear resize of an HWC float image."""
    h_in, w_in, c = img.shape
    if (h_in, w_in) == (h_out, w_out):
        return img.astype(np.float32, copy=False)
    ys = np.linspace(0, h_in - 1, h_out)
    xs = np.linspace(0, w_in - 1, w_out)
    y0 = np.floor(ys).astype(np.int64); y1 = np.minimum(y0 + 1, h_in - 1)
    x0 = np.floor(xs).astype(np.int64); x1 = np.minimum(x0 + 1, w_in - 1)
    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)
    a = img[y0[:, None], x0[None, :], :]
    b = img[y0[:, None], x1[None, :], :]
    cc = img[y1[:, None], x0[None, :], :]
    d = img[y1[:, None], x1[None, :], :]
    top = a + (b - a) * wx[None, :, None]
    bot = cc + (d - cc) * wx[None, :, None]
    return (top + (bot - top) * wy[:, None, None]).astype(np.float32, copy=False)


class SAIClassifier:
    """Inference wrapper. Backend chosen by file extension / `kind`."""

    def __init__(
        self,
        model: str | Path | None = None,
        classes: Sequence[str] = DEFAULT_CLASSES,
        kind: Literal["auto", "mock", "torch", "onnx"] = "auto",
        device: str = "cpu",
    ):
        self.classes = tuple(classes)
        self.device = device
        self._kind, self._impl, self._model_id = self._load(model, kind)
        log.info("SAIClassifier ready: kind=%s classes=%d model_id=%s",
                 self._kind, len(self.classes), self._model_id)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def kind(self) -> str:
        return self._kind

    def predict(self, sai: SAI) -> Hypothesis:
        img = sai_to_image(sai)
        logits, embedding = self._impl(img)
        # Stable softmax.
        z = logits - logits.max()
        exp = np.exp(z)
        probs = exp / exp.sum()
        order = np.argsort(probs)[::-1]
        top = [(self.classes[i], float(probs[i])) for i in order[: min(5, len(self.classes))]]
        best = int(order[0])
        return Hypothesis(
            class_=self.classes[best],
            score=float(probs[best]),
            top_k=top,
            model_id=self._model_id,
            embedding=embedding,
        )

    # --- backends ---------------------------------------------------------

    def _load(self, model: str | Path | None, kind: str):
        if kind == "auto":
            if model is None:
                kind = "mock"
            else:
                suffix = Path(model).suffix.lower()
                kind = "onnx" if suffix == ".onnx" else "torch"
        if kind == "mock":
            return "mock", self._mock_impl, "mock"
        if kind == "torch":
            return "torch", *self._load_torch(Path(model))  # type: ignore[arg-type]
        if kind == "onnx":
            return "onnx", *self._load_onnx(Path(model))  # type: ignore[arg-type]
        raise ValueError(f"unknown kind: {kind}")

    def _mock_impl(self, img: np.ndarray):
        """Deterministic SAI→logits via hashing. Same SAI ⇒ same prediction."""
        h = hashlib.sha256(img.tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        logits = rng.normal(0.0, 1.0, size=len(self.classes)).astype(np.float32)
        # A faint mean signal so log-energy clips bias slightly toward
        # non-"background" — keeps mock output non-trivial in smoke tests.
        if "background" in self.classes:
            bg = self.classes.index("background")
            logits[bg] -= 0.5 * float(np.log1p(img.mean()) * 10)
        return logits, None

    def _load_torch(self, path: Path):
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "torch backend requested but torch is not installed. "
                "Install with `pip install -e .[ml]`."
            ) from e
        ckpt = torch.load(str(path), map_location=self.device)
        model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.eval()
        model_id = "torch:" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]

        def impl(img: np.ndarray):
            x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = model(x)
                if isinstance(out, dict):
                    logits = out["logits"]
                    emb = out.get("embedding")
                else:
                    logits = out
                    emb = None
            logits_np = logits.squeeze(0).cpu().numpy().astype(np.float32)
            emb_np = emb.squeeze(0).cpu().numpy().astype(np.float32) if emb is not None else None
            return logits_np, emb_np

        return impl, model_id

    def _load_onnx(self, path: Path):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnx backend requested but onnxruntime is not installed. "
                "Install with `pip install -e .[ml]`."
            ) from e
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        model_id = "onnx:" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]

        def impl(img: np.ndarray):
            x = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
            outs = sess.run(None, {in_name: x})
            logits = outs[0].squeeze(0).astype(np.float32)
            emb = outs[1].squeeze(0).astype(np.float32) if len(outs) > 1 else None
            return logits, emb

        return impl, model_id
