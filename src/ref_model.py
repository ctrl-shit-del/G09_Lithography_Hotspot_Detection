"""Reference baseline CNN re-implemented from the supplied paper (Keras description):

    block   = Conv2D(12, 3x3, elu) -> Conv2D(12, 3x3, elu) -> Conv2D(12, 3x3, linear)
              -> BatchNorm(momentum 0.99, eps 1e-3) -> elu -> MaxPool(2, 2)
    network = block -> MaxPool(5, 5) -> block -> Flatten -> Dropout(0.3) -> Dense(10) -> Dense(1, sigmoid)
    training: Nadam, 10 epochs, no augmentation, unweighted BCE, threshold 0.5
    paper's reported parameter count: 12,873

    python -m src.ref_model                     # count params for the repo's 1x64x64 input and check
    python -m src.ref_model --in-ch 3 --size 150 --padding same

Input-size derivation (why the reference is run at 3 x 150 x 150 with 'same' padding)
    The description gives the parameter count (12,873) but neither the input size, the number
    of input channels nor the padding. All three are pinned down by that count:

      * conv1  : in_ch*12*9 + 12          =   336 (in_ch = 3)   /   120 (in_ch = 1)
      * conv2-6: 5 * (12*12*9 + 12)       = 6,540
      * BN x 2 : 2 * 4 * 12 (Keras counts gamma, beta, moving mean, moving var) = 96
      * dense  : (12*h*h)*10 + 10 + (10 + 1) = 1,200*h*h + 21   (h = feature side after block 2)
      ------------------------------------------------------------------------------------
      total    = 6,993 + 1,200*h*h   (in_ch = 3)   =>   h*h = 49,  h = 7   (12,873 - 6,993 = 5,880)
                 6,777 + 1,200*h*h   (in_ch = 1)   =>   h*h = 5.08 (not an integer: impossible)

    So the reference took a 3-channel input (a grey clip duplicated into RGB, as Keras image
    loaders do by default) and its second block flattens to 12 x 7 x 7 = 588. Working the
    spatial chain backwards with 'same' padding (convs keep size, pools floor-divide):
    7 <- MaxPool(2) <- 15 <- MaxPool(5) <- 75..79 <- MaxPool(2) <- 150..159. With 'valid'
    padding no input in 8..512 px gives h = 7 with 3 channels (`matching_configs`). 150 px is
    the smallest 'same'-padding input that reproduces the count and is what the repo uses.
    The 1-channel, 64 px twin ('ref64': 64 -> 32 -> 6 -> 3, flatten 108) has 7,857 params.

Keras-to-torch notes
    * Keras "Total params" counts BatchNorm's moving mean/var (non-trainable) as well as
      gamma/beta, i.e. 4 per channel. `keras_param_count` reproduces that convention;
      `count_params` from model.py would report 2 per channel.
    * Keras BN momentum 0.99 == torch momentum 0.01 (torch's is the update weight).
    * Dense(1, sigmoid) + binary_crossentropy == Dense(1) logits + BCEWithLogitsLoss; the model
      returns logits so train.py's loss / metrics path is shared unchanged.
    * Padding is not stated in the description. Keras' default is 'valid'; both are supported
      because the flatten size (hence the parameter count) depends on it.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn

TARGET_PARAMS = 12_873


def _block(cin: int, pad: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, 12, 3, padding=pad), nn.ELU(),
        nn.Conv2d(12, 12, 3, padding=pad), nn.ELU(),
        nn.Conv2d(12, 12, 3, padding=pad),                      # linear
        nn.BatchNorm2d(12, momentum=0.01, eps=1e-3), nn.ELU(),
        nn.MaxPool2d(2, 2),
    )


class RefCNN(nn.Module):
    def __init__(self, in_ch: int = 1, size: int = 64, padding: str = "valid", dropout: float = 0.3):
        super().__init__()
        pad = 1 if padding == "same" else 0
        self.features = nn.Sequential(_block(in_ch, pad), nn.MaxPool2d(5, 5), _block(12, pad))
        with torch.no_grad():
            f = self.features(torch.zeros(1, in_ch, size, size)).numel()
        self.flat_dim = f
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(f, 10), nn.Linear(10, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def keras_param_count(m: nn.Module) -> int:
    """Keras 'Total params': trainable parameters + BatchNorm running statistics."""
    n = sum(p.numel() for p in m.parameters())
    n += sum(b.numel() for name, b in m.named_buffers() if name.endswith(("running_mean", "running_var")))
    return n


def breakdown(m: nn.Module) -> list[tuple[str, int]]:
    rows = []
    for name, mod in m.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            rows.append((f"{name} {mod.__class__.__name__}", sum(p.numel() for p in mod.parameters())))
        elif isinstance(mod, nn.BatchNorm2d):
            rows.append((f"{name} BatchNorm2d (gamma,beta + running mean,var)", 4 * mod.num_features))
    return rows


def feature_hw(size: int, padding: str) -> int:
    """Spatial side after block -> MaxPool(5) -> block, for a square input."""
    s = size
    conv = (lambda v: v) if padding == "same" else (lambda v: v - 2)
    for _ in range(3):
        s = conv(s)
    s //= 2
    s //= 5
    for _ in range(3):
        s = conv(s)
    s //= 2
    return max(s, 0)


def analytic_total(in_ch: int, hw: int) -> int:
    convs = (in_ch * 12 * 9 + 12) + 2 * (12 * 12 * 9 + 12) + 3 * (12 * 12 * 9 + 12)
    bn = 2 * 4 * 12
    dense = (12 * hw * hw * 10 + 10) + (10 + 1)
    return convs + bn + dense


def matching_configs():
    """Every (in_ch, padding, input size) in a sane range whose Keras param count equals the target."""
    out = []
    for in_ch in (1, 3):
        for padding in ("valid", "same"):
            for size in range(8, 513):
                hw = feature_hw(size, padding)
                if hw > 0 and analytic_total(in_ch, hw) == TARGET_PARAMS:
                    out.append((in_ch, padding, size, hw))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-ch", type=int, default=1)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--padding", choices=["valid", "same"], default="valid")
    a = ap.parse_args(argv)

    if feature_hw(a.size, a.padding) <= 0:
        print(f"RefCNN(in_ch={a.in_ch}, size={a.size}, padding={a.padding}): network does not fit - the feature "
              f"map is empty before the second block (minimum input is "
              f"{min(s for s in range(8, 513) if feature_hw(s, a.padding) > 0)} px with {a.padding} padding)")
        m = None
    else:
        m = RefCNN(a.in_ch, a.size, a.padding)
    total = keras_param_count(m) if m is not None else -1
    if m is not None:
        print(f"RefCNN(in_ch={a.in_ch}, size={a.size}, padding={a.padding}): flatten = 12 x {int(math.isqrt(m.flat_dim // 12))}^2 = {m.flat_dim}")
        print(f"  Keras-convention total params = {total:,}   (target {TARGET_PARAMS:,})")
        if total == TARGET_PARAMS:
            print("  MATCH")
            return 0
        print("  NO MATCH - per-layer breakdown:")
        for name, n in breakdown(m):
            print(f"    {name:60s} {n:8,d}")
        print(f"    {'TOTAL':60s} {total:8,d}")
    print("\n  configurations that reproduce 12,873 exactly (in_ch, padding, input size, feature side):")
    cfgs = matching_configs()
    if not cfgs:
        print("    none in 8..512 px")
    for in_ch, padding, size, hw in cfgs:
        print(f"    in_ch={in_ch} padding={padding:5s} input={size:3d}x{size:<3d} -> flatten 12x{hw}x{hw}={12 * hw * hw}")
    for in_ch in (1, 3):
        for padding in ("valid", "same"):
            print(f"  in_ch={in_ch} padding={padding}: 12,873 requires flatten side^2 = "
                  f"{(TARGET_PARAMS - analytic_total(in_ch, 0)) / 120:.2f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
