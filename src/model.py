"""Phase 1 - baseline CNN for 1x64x64 hotspot / non-hotspot clips.

    python -m src.model            # prints the architecture, parameter count and a shape check

Design notes
------------
* Input is a single-channel coverage map in [0, 1] (see data.py `to_float`). No pretrained
  weights: the domain (binary layout polygons) has nothing in common with ImageNet.
* Four conv stages (64 -> 32 -> 16 -> 8 -> 4) each of two 3x3 conv + BN + ReLU followed by a
  2x2 max-pool, then global average pooling and a single logit. BCE-with-logits is used in
  train.py so the model emits one logit, not a 2-way softmax.
* Global average pooling (rather than flatten + FC) keeps the head tiny and makes the model
  robust to the small translations that the random-crop-free augmentation in train.py does
  not cover.
* `MODELS` is a registry so later phases can add variants without touching train.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ref_model import RefCNN, keras_param_count


def _stage(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class HotspotCNN(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), in_ch: int = 1, dropout: float = 0.3):
        super().__init__()
        stages, c = [], in_ch
        for w in widths:
            stages.append(_stage(c, w))
            c = w
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, 1))
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x))).squeeze(1)  # (N,) logits


# name -> (class, kwargs, expected Keras-convention parameter count or None). RefCNN needs the
# input side (`size`) to size its flatten; build_model passes it only to classes that take it.
MODELS = {
    "cnn": (HotspotCNN, dict(widths=(32, 64, 128, 256)), None),
    "cnn_small": (HotspotCNN, dict(widths=(16, 32, 64, 128)), None),
    "ref": (RefCNN, dict(in_ch=3, padding="same"), 12_873),     # reference paper: 3 x 150 x 150
    "ref64": (RefCNN, dict(in_ch=1, padding="same"), 7_857),    # same net on the repo's 1 x 64 x 64
}


def build_model(name: str = "cnn", size: int | None = None, **overrides) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'; choose from {sorted(MODELS)}")
    cls, kw, expect = MODELS[name]
    kw = {**kw, **overrides}
    if cls is RefCNN and size is not None:
        kw["size"] = size
    m = cls(**kw)
    if expect is not None:
        got = keras_param_count(m)
        assert got == expect, f"{name}: Keras-convention params {got:,} != expected {expect:,} (size={size})"
    return m


def in_channels(m: nn.Module) -> int:
    return next(mod for mod in m.modules() if isinstance(mod, nn.Conv2d)).in_channels


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, size in (("cnn", 64), ("cnn_small", 64), ("ref", 150), ("ref64", 64)):
        m = build_model(name, size=size)
        out = m(torch.zeros(2, in_channels(m), size, size))
        print(f"{name:10s} in={in_channels(m)}x{size}x{size} params={count_params(m):,} "
              f"(keras {keras_param_count(m):,})  out={tuple(out.shape)}")
    print(build_model("cnn"))
